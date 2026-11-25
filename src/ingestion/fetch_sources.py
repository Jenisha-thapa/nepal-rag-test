"""
Phase 1: Source acquisition.

Reads config/domains.yaml and fetches every configured government source,
saving the raw content to data/raw/<domain_id>/ along with a metadata
sidecar file. The metadata captured here (source URL, issuing authority,
document type, fetch timestamp, content hash) is exactly what the proposal's
"Data Collection and Preparation" section calls for, and it's what later
lets the generation component cite a source and avoid claiming real-time
currency for information it can't verify (see proposal's Ethical section).

This script is intentionally conservative:
- it does not guess at pages that aren't explicitly listed in domains.yaml
- it respects each site's robots.txt (Disallow rules and Crawl-delay)
- it verifies TLS certificates
- it never overwrites a previous fetch silently — it versions by timestamp
- every failure is recorded, not swallowed, so gaps in the corpus are visible
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.robotparser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

from src.utils import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "domains.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw"
USER_AGENT = "Mozilla/5.0 (compatible; NepalSewaAI/2.0; research project)"
REQUEST_TIMEOUT_SECONDS = 30


@dataclass
class FetchResult:
    """Everything downstream components need to know about one fetch attempt."""

    domain_id: str
    domain_name_en: str
    source_name: str
    source_url: str
    document_type: str
    fetched_at: str  # ISO 8601, UTC
    status: str  # "ok" | "http_error" | "request_error"
    http_status_code: int | None
    content_type: str | None
    content_length_bytes: int | None
    sha256: str | None
    saved_path: str | None
    error_message: str | None = None
    retrieved_by: str = "crawler"  # "crawler" | "manual-download" (see import_downloads)


def load_domain_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Domain config not found at {config_path}. "
            "Run this from the repo root, or pass --config explicitly."
        )
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(text: str, max_length: int = 80) -> str:
    """Turn a name or URL path into a filesystem-safe slug."""
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^\w\-.]+", "-", text, flags=re.UNICODE)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:max_length] if text else "source"


def guess_extension(content_type: str | None) -> str:
    if not content_type:
        return ".bin"
    content_type = content_type.split(";")[0].strip().lower()
    return {
        "text/html": ".html",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
    }.get(content_type, ".bin")


class PoliteFetcher:
    """HTTP client that follows each host's robots.txt (Disallow + Crawl-delay) and verifies TLS."""

    def __init__(self, min_delay: float = 1.0) -> None:
        self.min_delay = min_delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request: dict[str, float] = {}

    def _rules(self, url: str) -> tuple[str, urllib.robotparser.RobotFileParser]:
        parts = urlparse(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.session.get(host + "/robots.txt", timeout=REQUEST_TIMEOUT_SECONDS)
                lines = r.text.splitlines() if r.ok and "<html" not in r.text[:500].lower() else []
            except requests.RequestException:
                lines = []
            rp.parse(lines)
            rp.modified()
            if not lines:
                rp.allow_all = True  # no robots.txt -> no restrictions
            self._robots[host] = rp
        return host, self._robots[host]

    def get(self, url: str, retries: int = 1) -> requests.Response:
        host, rp = self._rules(url)
        if not rp.can_fetch(USER_AGENT, url):
            raise PermissionError(f"disallowed by {host}/robots.txt")
        delay = max(self.min_delay, float(rp.crawl_delay(USER_AGENT) or 0))
        for attempt in range(retries + 1):
            wait = self._last_request.get(host, -1e9) + delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            finally:
                self._last_request[host] = time.monotonic()
            if response.status_code not in (429, 503) or attempt == retries:
                return response
            # server busy / rate-limited: back off (honour Retry-After) and try once more
            retry_after = response.headers.get("Retry-After", "")
            backoff = float(retry_after) if retry_after.isdigit() else 30.0
            logger.info("HTTP %s from %s — retrying in %.0fs", response.status_code, host, backoff)
            time.sleep(backoff)
        return response


def fetch_one_source(
    domain_id: str,
    domain_name_en: str,
    source: dict[str, Any],
    output_dir: Path,
    fetcher: PoliteFetcher | None = None,
) -> FetchResult:
    """Fetch a single configured source and persist raw content + metadata."""
    source_name = source["name"]
    source_url = source["url"]
    document_type = source.get("type", "unknown")
    fetched_at = datetime.now(timezone.utc).isoformat()
    fetcher = fetcher or PoliteFetcher()

    try:
        response = fetcher.get(source_url)
    except (requests.RequestException, PermissionError) as exc:
        logger.warning("Request failed for %s (%s): %s", source_name, source_url, exc)
        return FetchResult(
            domain_id=domain_id,
            domain_name_en=domain_name_en,
            source_name=source_name,
            source_url=source_url,
            document_type=document_type,
            fetched_at=fetched_at,
            status="request_error",
            http_status_code=None,
            content_type=None,
            content_length_bytes=None,
            sha256=None,
            saved_path=None,
            error_message=str(exc),
        )

    content_type = response.headers.get("Content-Type")
    status = "ok" if response.ok else "http_error"

    saved_path: str | None = None
    sha256: str | None = None
    content_length_bytes: int | None = None

    if response.ok:
        domain_dir = output_dir / domain_id
        domain_dir.mkdir(parents=True, exist_ok=True)

        ext = guess_extension(content_type)
        timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{slugify(source_name)}__{timestamp_tag}{ext}"
        file_path = domain_dir / filename

        file_path.write_bytes(response.content)
        # repo-relative, forward slashes: the record stays valid on any OS / clone location
        try:
            saved_path = file_path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            saved_path = file_path.as_posix()
        sha256 = hashlib.sha256(response.content).hexdigest()
        content_length_bytes = len(response.content)

        logger.info("Saved %s -> %s (%d bytes)", source_url, file_path, content_length_bytes)
    else:
        logger.warning(
            "HTTP %s for %s (%s)", response.status_code, source_name, source_url
        )

    result = FetchResult(
        domain_id=domain_id,
        domain_name_en=domain_name_en,
        source_name=source_name,
        source_url=source_url,
        document_type=document_type,
        fetched_at=fetched_at,
        status=status,
        http_status_code=response.status_code,
        content_type=content_type,
        content_length_bytes=content_length_bytes,
        sha256=sha256,
        saved_path=saved_path,
    )

    if saved_path:
        meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
        meta_path.write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return result


def _already_fetched(output_dir: Path) -> set[str]:
    """Source URLs that already have a successful fetch on disk."""
    urls = set()
    for meta in output_dir.glob("*/*.meta.json"):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("status") == "ok":
                urls.add(m["source_url"])
        except (json.JSONDecodeError, KeyError):
            continue
    return urls


def run(config_path: Path, output_dir: Path, domain_filter: str | None, delay_seconds: float,
        only_missing: bool = False) -> list[FetchResult]:
    config = load_domain_config(config_path)
    domains = config.get("domains", [])
    skip = _already_fetched(output_dir) if only_missing else set()

    if domain_filter:
        domains = [d for d in domains if d["id"] == domain_filter]
        if not domains:
            raise ValueError(f"No domain with id '{domain_filter}' found in {config_path}")

    jobs_by_host: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for domain in domains:
        domain_name_en = domain.get("name_en", domain["id"])
        logger.info("Domain '%s': %d configured source(s)", domain_name_en, len(domain.get("sources", [])))
        for source in domain.get("sources", []):
            if source["url"] in skip:
                continue
            host = urlparse(source["url"]).netloc
            jobs_by_host.setdefault(host, []).append((domain["id"], domain_name_en, source))

    # Round-robin across hosts so one site's crawl-delay doesn't stall the others.
    fetcher = PoliteFetcher(min_delay=delay_seconds)
    results: list[FetchResult] = []
    queues = list(jobs_by_host.values())
    while any(queues):
        for queue in queues:
            if queue:
                domain_id, domain_name_en, source = queue.pop(0)
                results.append(fetch_one_source(domain_id, domain_name_en, source, output_dir, fetcher))
    return results


def print_summary(results: list[FetchResult]) -> None:
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]

    print(f"\n{'=' * 60}")
    print(f"Fetch summary: {len(ok)} succeeded, {len(failed)} failed (of {len(results)} total)")
    print(f"{'=' * 60}")
    if failed:
        print("\nFailed sources (fix these before Phase 2):")
        for r in failed:
            reason = r.error_message or f"HTTP {r.http_status_code}"
            print(f"  - [{r.domain_id}] {r.source_name}: {reason}")


def import_downloads(inbox: Path, config_path: Path = DEFAULT_CONFIG_PATH,
                     output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[FetchResult]:
    """Register files downloaded by hand in a browser (e.g. Acts on a file server that refuses
    scripted downloads) as crawl records, exactly as if the crawler had fetched them.

    Each file in `inbox` is matched to a configured source URL by content: a HEAD request to every
    configured PDF source that has no successful download yet gives its size and ETag (an MD5 of
    the file on these servers), so file names don't matter. Matched files are saved to
    data/raw/<domain>/ with a .meta.json (`retrieved_by: "manual-download"`); unmatched files are
    left in the inbox and reported."""
    config = load_domain_config(config_path)
    have = _already_fetched(output_dir)
    candidates = [(d["id"], d.get("name_en", d["id"]), s) for d in config.get("domains", [])
                  for s in d.get("sources", [])
                  if s["url"].lower().split("?")[0].endswith(".pdf") and s["url"] not in have]
    files = sorted(p for p in inbox.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not files:
        print(f"No PDF files in {inbox}")
        return []

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    by_md5: dict[str, tuple] = {}
    by_size: dict[int, list[tuple]] = {}
    for job in candidates:
        try:
            head = session.head(job[2]["url"], timeout=REQUEST_TIMEOUT_SECONDS, allow_redirects=True)
        except requests.RequestException as exc:
            logger.warning("HEAD failed for %s: %s", job[2]["url"], exc)
            continue
        etag = head.headers.get("ETag", "").strip('"').lower()
        if re.fullmatch(r"[0-9a-f]{32}", etag):
            by_md5[etag] = job
        if head.headers.get("Content-Length", "").isdigit():
            by_size.setdefault(int(head.headers["Content-Length"]), []).append(job)
        time.sleep(0.5)

    results = []
    for path in files:
        content = path.read_bytes()
        job = by_md5.get(hashlib.md5(content).hexdigest())
        if job is None and len(by_size.get(len(content), [])) == 1:
            job = by_size[len(content)][0]
        if job is None:
            print(f"  ? {path.name}: does not match any configured PDF source — left in {inbox}")
            continue
        domain_id, domain_name_en, source = job
        domain_dir = output_dir / domain_id
        domain_dir.mkdir(parents=True, exist_ok=True)
        timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        file_path = domain_dir / f"{slugify(source['name'])}__{timestamp_tag}.pdf"
        file_path.write_bytes(content)
        result = FetchResult(
            domain_id=domain_id, domain_name_en=domain_name_en, source_name=source["name"],
            source_url=source["url"], document_type=source.get("type", "unknown"),
            fetched_at=datetime.now(timezone.utc).isoformat(), status="ok", http_status_code=200,
            content_type="application/pdf", content_length_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            saved_path=file_path.resolve().relative_to(REPO_ROOT).as_posix(),
            retrieved_by="manual-download",
        )
        file_path.with_suffix(".pdf.meta.json").write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
        path.unlink()
        results.append(result)
        print(f"  ✓ {path.name} -> [{domain_id}] {source['name']}")
    missing = [s["name"] for _, _, s in candidates if s["url"] not in {r.source_url for r in results}]
    print(f"\nImported {len(results)} file(s). Still not downloaded ({len(missing)}): " + "; ".join(missing))
    return results


def fetch_all(domain: str | None = None, delay_seconds: float = 1.0, only_missing: bool = False) -> list[FetchResult]:
    """Fetch sources in config/domains.yaml into data/raw/ and print a summary.
    only_missing=True skips sources that already have a successful download."""
    results = run(DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_DIR, domain, delay_seconds, only_missing)
    print_summary(results)
    return results

"""
Hand-written knowledge base: data/chunks/new_chunks.json (or data/chunks/<domain>.yaml) -> retrieval chunks.

new_chunks.json holds every domain in one file: {"domains": [{"domain_id", "sources", "gaps", "chunks": [...]}]}.
If it is absent, each data/chunks/<domain>.yaml holds one domain's chunks. Chunk fields are the same:

    domain_id: identity_civil_registration
    chunks:
      - id: nid-documents                  # unique slug -> chunk_id "identity_civil_registration::nid-documents"
        url: https://donidcr.gov.np/...    # page the facts come from (must be a fetched source)
        section: राष्ट्रिय परिचयपत्र — आवश्यक कागजात
        type: documents                    # faq | procedure | documents | fees | eligibility | contact | notice | about | ...
        date_bs: ३ भदौ, २०८३                # optional — date stated on the source
        text: |
          ...

Metadata that is not written by hand is filled in automatically: source name, fetch time and
document type come from the crawl record (data/raw/**/*.meta.json) of that URL, authority from
config/domains.yaml, plus language, token count and a content hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from src.chunking.structure_aware import _language
from src.processing.html_extract import first_bs_date
from src.utils import REPO_ROOT, get_settings, load_yaml

logger = logging.getLogger(__name__)

KB_DIR = REPO_ROOT / "data" / "chunks"
NEW_KB_FILE = "new_chunks.json"
REQUIRED = ("id", "url", "section", "type", "text")


def _norm_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


def _crawl_records() -> dict[str, dict[str, Any]]:
    """Latest successful crawl record (.meta.json) per source URL."""
    raw_dir = Path(get_settings()["paths"]["raw_dir"])
    records: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(raw_dir.glob("*/*.meta.json")):  # sorted -> newer timestamps win
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "ok":
            # "<source>__<timestamp>.html.meta.json" -> "<source>__<timestamp>"; derived from the
            # file name (not the stored saved_path) so it is the same on Windows, Linux and macOS
            meta["doc_id"] = Path(meta_path.name[: -len(".meta.json")]).stem
            records[_norm_url(meta["source_url"])] = meta
    return records


def _kb_files(kb_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """(file, {domain_id, chunks}) per domain. data/chunks/new_chunks.json — all domains in one
    file — is used when present; otherwise one data/chunks/<domain>.yaml per domain."""
    combined = kb_dir / NEW_KB_FILE
    if combined.exists():
        data = json.loads(combined.read_text(encoding="utf-8"))
        return [(combined, d) for d in data.get("domains", [])]
    return [(path, load_yaml(path) or {}) for path in sorted(kb_dir.glob("*.yaml"))]


def load_knowledge_base(
    count_tokens: Callable[[str], int],
    max_tokens: int,
    kb_dir: Path = KB_DIR,
    crawl_records: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records = _crawl_records() if crawl_records is None else crawl_records
    domain_cfg = load_yaml(REPO_ROOT / "config" / "domains.yaml").get("domains", [])
    authorities = {d["id"]: d.get("authority", "") for d in domain_cfg}
    # display name of each source as configured in domains.yaml (preferred over the crawl record's)
    source_names = {_norm_url(s["url"]): s["name"] for d in domain_cfg for s in d.get("sources", [])}

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path, cfg in _kb_files(kb_dir):
        domain_id = cfg.get("domain_id") or path.stem
        for position, entry in enumerate(cfg.get("chunks") or []):
            missing = [k for k in REQUIRED if not entry.get(k)]
            if missing:
                raise ValueError(f"{path.name}: chunk {entry.get('id', position)} is missing {missing}")
            chunk_id = f"{domain_id}::{entry['id']}"
            if chunk_id in seen_ids:
                raise ValueError(f"{path.name}: duplicate chunk id '{entry['id']}'")
            seen_ids.add(chunk_id)

            meta = records.get(_norm_url(entry["url"]))
            if meta is None:
                logger.warning("%s: %s cites %s, which has not been fetched (run `nepal-rag fetch`)",
                               path.name, entry["id"], entry["url"])
                meta = {}
            text = entry["text"].strip()
            tokens = count_tokens(text)
            if tokens > max_tokens:
                logger.warning("%s: %s has %d tokens (> %d) — the embedder will truncate it; split it",
                               path.name, entry["id"], tokens, max_tokens)
            source_name = (entry.get("source") or source_names.get(_norm_url(entry["url"]))
                           or meta.get("source_name") or entry["url"])
            section = entry["section"]
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": meta.get("doc_id", "manual"),
                "domain_id": domain_id,
                "source_name": source_name,
                "source_url": entry["url"],
                "authority": authorities.get(domain_id, ""),
                "document_type": meta.get("document_type", ""),
                "page_title": source_name,
                "section": section,
                "content_type": entry["type"],
                "language": _language(text),
                "date_bs": entry.get("date_bs") or first_bs_date(text),
                "fetched_at": meta.get("fetched_at", ""),
                "position": position,
                "token_count": tokens,
                "content_hash": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
                "text": text,
                "search_text": f"{source_name} › {section}\n{text}",
            })
    logger.info("Knowledge base: %d hand-written chunks from %s", len(chunks), kb_dir)
    return chunks

"""
Document processing: raw HTML/PDF -> structured blocks -> structure-aware chunks.

    data/raw/<domain>/<source>__<timestamp>.html (+ .meta.json)
      -> data/processed/<domain>/<source>__<timestamp>.json   (title, date, typed blocks)
      -> data/processed/all_chunks.json                       (chunks + metadata used for retrieval)
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pymupdf  # PyMuPDF

from src.processing.html_extract import Block, extract_html
from src.utils import REPO_ROOT, get_settings, get_token_counter, load_json, load_yaml, normalize_nepali_text, save_json

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = ("*.html", "*.htm", "*.pdf", "*.txt")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a (text-based) PDF using PyMuPDF."""
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            return "\n\n".join(page.get_text() for page in doc if page.get_text().strip())
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", pdf_path, exc)
        return ""


def extract_text(file_path: Path) -> str:
    """Plain text of a non-HTML source (HTML goes through html_extract instead)."""
    if file_path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(file_path)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    """Normalize Nepali Unicode, strip control characters, collapse whitespace."""
    text = normalize_nepali_text(text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _latest_raw_files(raw_dir: Path) -> list[Path]:
    """Each fetch saves `<source>__<timestamp>.<ext>`; keep only the newest file per source."""
    files = [f for ext in RAW_EXTENSIONS for f in raw_dir.rglob(ext) if f.stat().st_size > 0]
    latest: dict[tuple[str, str], Path] = {}
    for f in sorted(files, key=lambda p: p.name):  # timestamp suffix sorts chronologically
        latest[(str(f.parent), f.name.split("__")[0])] = f
    return sorted(latest.values())


def _authorities() -> dict[str, str]:
    """domain_id -> issuing authority, from config/domains.yaml."""
    cfg = load_yaml(REPO_ROOT / "config" / "domains.yaml")
    return {d["id"]: d.get("authority", "") for d in cfg.get("domains", [])}


def process_document(file_path: Path, raw_dir: Path, authorities: dict[str, str] | None = None) -> dict[str, Any]:
    """Turn one raw file (+ its .meta.json sidecar) into a structured document dict."""
    meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
    meta = load_json(meta_path) if meta_path.exists() else {}
    rel = file_path.relative_to(raw_dir)
    domain_id = rel.parts[0] if len(rel.parts) > 1 else "unknown"

    if file_path.suffix.lower() in (".html", ".htm"):
        page = extract_html(file_path.read_text(encoding="utf-8", errors="ignore"))
        title, page_date, blocks = page.title, page.date_bs, page.blocks
    else:  # PDF / text: paragraphs become text blocks
        raw_text = extract_text(file_path)
        blocks = [Block("text", clean_text(p)) for p in re.split(r"\n\s*\n", raw_text) if clean_text(p)]
        title, page_date = meta.get("source_name", file_path.stem), ""

    return {
        "doc_id": file_path.stem,
        "domain_id": domain_id,
        # Human-readable name, used in citations (e.g. "DoNIDCR — Social Security Allowance FAQs")
        "source_name": meta.get("source_name") or file_path.stem,
        "source_url": meta.get("source_url", ""),
        "authority": (authorities or {}).get(domain_id, ""),
        "document_type": meta.get("document_type") or file_path.suffix.lstrip("."),
        "page_title": title,
        "page_date_bs": page_date,
        "fetched_at": meta.get("fetched_at", ""),
        "blocks": [asdict(b) for b in blocks],
        "cleaned_text": " ".join(b.text for b in blocks),
    }


def build_chunks(verbose: bool = False) -> list[dict[str, Any]]:
    """Write data/processed/all_chunks.json from the configured knowledge base:
    `manual` = hand-written chunks in data/chunks/*.yaml (default),
    `auto`   = structure-aware chunking of the crawled pages."""
    settings = get_settings()
    mode = settings["chunking"].get("knowledge_base", "manual")
    if mode == "manual":
        from src.chunking.knowledge_base import load_knowledge_base
        chunks = load_knowledge_base(get_token_counter(), settings["chunking"]["max_chunk_tokens"])
    elif mode == "auto":
        chunks = process_all(verbose=verbose)
    else:
        raise ValueError(f"Unknown chunking.knowledge_base '{mode}' (use 'manual' or 'auto')")

    out = Path(settings["paths"]["processed_dir"]) / "all_chunks.json"
    save_json(chunks, out)
    if verbose:
        per_domain: dict[str, int] = {}
        for c in chunks:
            per_domain[c["domain_id"]] = per_domain.get(c["domain_id"], 0) + 1
        print(f"  Knowledge base ({mode}): {len(chunks)} chunks {per_domain} -> {out}")
    return chunks


def process_all(verbose: bool = False) -> list[dict[str, Any]]:
    """Automatic mode: extract every crawled page into blocks and chunk them by structure."""
    settings = get_settings()
    raw_dir = Path(settings["paths"]["raw_dir"])
    processed_dir = Path(settings["paths"]["processed_dir"])

    if not raw_dir.exists():
        logger.error("Raw directory not found: %s — run `fetch` first", raw_dir)
        return []

    docs = []
    authorities = _authorities()
    for file_path in _latest_raw_files(raw_dir):
        doc = process_document(file_path, raw_dir, authorities)
        if len(doc["cleaned_text"]) < settings["processing"]["min_chunk_chars"]:
            logger.warning("Skipping %s: too little text after cleaning", doc["source_name"])
            continue
        save_json(doc, processed_dir / doc["domain_id"] / f"{doc['doc_id']}.json")
        docs.append(doc)
        if verbose:
            print(f"  Processed: {doc['source_name']} -> {len(doc['blocks'])} blocks, {len(doc['cleaned_text'])} chars")

    logger.info("Processed %d documents", len(docs))

    from src.chunking.structure_aware import chunk_documents
    return chunk_documents(
        docs,
        count_tokens=get_token_counter(),
        max_tokens=settings["chunking"]["max_chunk_tokens"],
        min_tokens=settings["chunking"]["min_chunk_tokens"],
        verbose=verbose,
    )

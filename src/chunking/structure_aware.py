"""
Structure-aware chunking: typed blocks (from html_extract) -> retrieval chunks with metadata.

Rules
  * FAQ: one chunk per question + its complete answer (tables and lists included).
    Long answers are split, and every part repeats the question so it stays self-contained.
  * Other content: grouped by heading; tiny sections (e.g. contact cards) are merged
    until they are worth a chunk; nothing crosses a top-level (h1/h2) boundary once
    the chunk is big enough.
  * Size: measured with the embedding model's tokenizer (`max_tokens`), so no chunk
    is silently truncated by the embedder.
  * Every chunk gets a `search_text` = "page title › section\\n text" — embedding and
    BM25 both see where the passage comes from.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from src.processing.html_extract import Block, first_bs_date

FAQ_Q = re.compile(r"^(?:[०-९0-9]+\s*[.)]\s*)?(?:प्रश्न|question)\s*[:：]\s*(.*)$", re.IGNORECASE)
FAQ_A = re.compile(r"^(?:उत्तर|answer)\s*[:：]\s*(.*)$", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[।?!])\s+|(?<=\.)\s+(?=[A-Zऀ-ॿ])")


@dataclass
class Unit:
    kind: str                      # "faq" | "text"
    path: list[str]                # heading trail at the start of the unit
    question: str = ""
    lines: list[Block] = field(default_factory=list)


def _section_label(path: list[str], title: str) -> str:
    parts = [p for p in path if p and p != title]
    return " › ".join(parts[-3:])


def _render(blocks: list[Block]) -> str:
    """Join blocks: wrapped paragraph lines flow together; list items / table rows become bullets."""
    out: list[str] = []
    for b in blocks:
        if b.kind in ("item", "row"):
            out.append(f"\n- {b.text}")
        elif b.kind == "heading":
            out.append(f"\n{b.text}\n")
        else:
            out.append(("" if not out or out[-1].endswith("\n") else " ") + b.text)
    return re.sub(r"\n{2,}", "\n", "".join(out)).strip()


def _build_units(blocks: list[Block], count: Callable[[str], int], min_tokens: int) -> list[Unit]:
    units: list[Unit] = []
    stack: list[tuple[int, str]] = []
    cur = Unit("text", [])
    state = ""  # for FAQ units: "q" while reading the question, "a" while reading the answer

    def path() -> list[str]:
        return [t for _, t in stack]

    def close() -> None:
        nonlocal cur
        if cur.lines or cur.question:
            units.append(cur)
        cur = Unit("text", path())

    for b in blocks:
        q = FAQ_Q.match(b.text) if b.kind == "text" else None
        a = FAQ_A.match(b.text) if b.kind == "text" else None

        if b.kind == "heading":
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            # new chunk at a heading, unless the current one is still too small to stand alone
            if cur.kind == "faq" or b.level <= 2 or count(_render(cur.lines)) >= min_tokens:
                close()
                state = ""
            cur.lines.append(b)
            if len(cur.lines) == 1:
                cur.path = path()
        elif q:
            close()
            cur, state = Unit("faq", path(), question=q.group(1)), "q"
        elif a:
            if cur.kind != "faq":
                # "question line" + "उत्तर:" without a numbered marker: promote the last line
                prev = cur.lines.pop() if cur.lines and cur.lines[-1].kind == "text" else None
                close()
                cur = Unit("faq", path(), question=prev.text if prev else "")
            state = "a"
            if a.group(1):
                cur.lines.append(Block("text", a.group(1)))
        elif cur.kind == "faq" and state == "q":
            cur.question = f"{cur.question} {b.text}".strip()
        else:
            cur.lines.append(b)
    close()
    return units


def _split_to_budget(blocks: list[Block], count: Callable[[str], int], budget: int) -> list[list[Block]]:
    """Greedy-pack blocks into groups whose rendered text fits the budget; split oversize blocks by sentence."""
    groups: list[list[Block]] = [[]]
    for b in blocks:
        pieces = [b]
        if count(b.text) > budget:
            sentences = [s for s in SENTENCE_SPLIT.split(b.text) if s.strip()]
            pieces, buf = [], ""
            for s in sentences:
                if buf and count(f"{buf} {s}") > budget:
                    pieces.append(Block(b.kind, buf))
                    buf = s
                else:
                    buf = f"{buf} {s}".strip()
            if buf:
                pieces.append(Block(b.kind, buf))
        for p in pieces:
            if groups[-1] and count(_render(groups[-1] + [p])) > budget:
                groups.append([])
            groups[-1].append(p)
    return [g for g in groups if g]


def _merge_small(units: list[Unit], count: Callable[[str], int], min_tokens: int, max_tokens: int) -> list[Unit]:
    """Merge consecutive small text units (e.g. contact cards) under the same top-level section."""
    merged: list[Unit] = []
    for u in units:
        prev = merged[-1] if merged else None
        if (prev and u.kind == prev.kind == "text"
                and prev.path[:1] == u.path[:1]
                and count(_render(prev.lines)) < min_tokens
                and count(_render(prev.lines + u.lines)) <= max_tokens):
            prev.lines.extend(u.lines)
        else:
            merged.append(u)
    return merged


def _language(text: str) -> str:
    dev = len(re.findall(r"[ऀ-ॿ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    if dev + lat == 0:
        return "unknown"
    share = dev / (dev + lat)
    return "ne" if share > 0.8 else "en" if share < 0.2 else "mixed"


def chunk_document(
    doc: dict[str, Any],
    count_tokens: Callable[[str], int],
    max_tokens: int = 350,
    min_tokens: int = 40,
) -> list[dict[str, Any]]:
    """Chunk one processed document (with `blocks`) into retrieval chunks with metadata."""
    title = doc.get("page_title") or doc.get("source_name", "")
    blocks = [Block(**b) if isinstance(b, dict) else b for b in doc.get("blocks", [])]
    units = _merge_small(_build_units(blocks, count_tokens, min_tokens), count_tokens, min_tokens, max_tokens)

    chunks: list[dict[str, Any]] = []
    for unit in units:
        if unit.kind == "faq":
            header = f"प्रश्न: {unit.question}\nउत्तर:"
            groups = _split_to_budget(unit.lines, count_tokens, max_tokens - count_tokens(header))
            texts = [f"{header} {_render(g)}" if i == 0 else f"{header} (जारी) {_render(g)}"
                     for i, g in enumerate(groups)] or [header]
            kind_of = lambda g: "faq"  # noqa: E731
        else:
            groups = _split_to_budget(unit.lines, count_tokens, max_tokens)
            texts = [_render(g) for g in groups]

            def kind_of(g: list[Block]) -> str:
                rows = sum(b.kind == "row" for b in g)
                items = sum(b.kind == "item" for b in g)
                return "table" if rows > len(g) / 2 else "list" if items > len(g) / 2 else "text"

        for g, text in zip(groups or [[]], texts):
            # skip a lone heading with no body, and "Download" / "Login" leftovers
            # (a run of several headings — e.g. a list of notice titles — is content)
            if unit.kind == "text" and ((len(g) == 1 and g[0].kind == "heading") or count_tokens(text) < 8):
                continue
            section = _section_label(unit.path, title)
            position = len(chunks)
            chunks.append({
                "chunk_id": f"{doc['doc_id']}::{position:03d}",
                "doc_id": doc["doc_id"],
                "domain_id": doc["domain_id"],
                "source_name": doc["source_name"],
                "source_url": doc["source_url"],
                "authority": doc.get("authority", ""),
                "document_type": doc.get("document_type", ""),
                "page_title": title,
                "section": section,
                "content_type": kind_of(g),
                "language": _language(text),
                # only a date written in the chunk itself; the page's publication date does not
                # apply to notice lists or other sections on the same page
                "date_bs": first_bs_date(text),
                "fetched_at": doc.get("fetched_at", ""),
                "position": position,
                "token_count": count_tokens(text),
                "content_hash": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
                "text": text,
                "search_text": f"{title} › {section}\n{text}" if section else f"{title}\n{text}",
            })
    return chunks


def chunk_documents(
    docs: list[dict[str, Any]],
    count_tokens: Callable[[str], int],
    max_tokens: int = 350,
    min_tokens: int = 40,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in docs:
        chunks = []
        for c in chunk_document(doc, count_tokens, max_tokens, min_tokens):
            # identical text across pages (site-wide notice tickers, shared footers) is indexed once
            if c["content_hash"] in seen:
                continue
            seen.add(c["content_hash"])
            chunks.append(c)
        all_chunks.extend(chunks)
        if verbose:
            kinds: dict[str, int] = {}
            for c in chunks:
                kinds[c["content_type"]] = kinds.get(c["content_type"], 0) + 1
            print(f"  Chunked: {doc['source_name']} -> {len(chunks)} chunks {kinds}")
    return all_chunks

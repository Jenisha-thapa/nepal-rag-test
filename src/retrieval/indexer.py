"""
Retrieval — BM25 (lexical), dense (sentence-transformers + ChromaDB) and hybrid.

Both retrievers index each chunk's `search_text` ("page title › section\\n text"), so a
passage is found by its context as well as its body. All chunk metadata is stored in
ChromaDB, which enables metadata filters such as `where={"domain_id": ...}`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.utils import get_settings, load_json

logger = logging.getLogger(__name__)

# Chunk fields stored as ChromaDB metadata (must be str / int / float / bool).
METADATA_FIELDS = (
    "doc_id", "domain_id", "source_name", "source_url", "authority", "document_type", "page_title",
    "section", "content_type", "language", "date_bs", "fetched_at", "position", "token_count", "content_hash",
)


@dataclass
class RetrievalResult:
    """A single retrieved passage with its metadata."""
    chunk_id: str
    domain_id: str
    source_name: str
    source_url: str
    title: str
    section: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# Startup progress, read by GET /api/health so the web page can show a progress bar.
PROGRESS: dict[str, Any] = {"stage": "starting", "done": 0, "total": 0}


def set_progress(stage: str, done: int = 0, total: int = 0) -> None:
    PROGRESS.update(stage=stage, done=done, total=total)


def _result(chunk_id: str, meta: dict[str, Any], text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        domain_id=meta.get("domain_id", ""),
        source_name=meta.get("source_name", ""),
        source_url=meta.get("source_url", ""),
        title=meta.get("page_title", ""),
        section=meta.get("section", ""),
        text=text,
        score=float(score),
        metadata={k: meta.get(k) for k in METADATA_FIELDS if k in meta},
    )


def load_chunks() -> list[dict[str, Any]]:
    path = Path(get_settings()["paths"]["processed_dir"]) / "all_chunks.json"
    return load_json(path) if path.exists() else []


# ── BM25 ───────────────────────────────────────────────────────────────────────
class BM25Retriever:
    """BM25 lexical retriever over Devanagari + Latin word tokens."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.index: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[\wऀ-ॿ]+", text.lower())

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        corpus = [self._tokenize(c.get("search_text") or c.get("text", "")) for c in chunks]
        if not any(corpus):
            raise ValueError(
                f"BM25: none of the {len(chunks)} chunks has any 'text' to index — "
                "all_chunks.json probably holds documents, not chunks. Re-run `uv run nepal-rag process`."
            )
        self.index = BM25Okapi(corpus, k1=self.k1, b=self.b)
        logger.info("BM25 index built with %d chunks", len(chunks))

    def search(self, query: str, top_k: int = 5, domain: str | None = None) -> list[RetrievalResult]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        scores = self.index.get_scores(self._tokenize(query))
        if domain:
            scores = np.where([c.get("domain_id") == domain for c in self.chunks], scores, -np.inf)
        results = []
        for idx in np.argsort(scores)[::-1][:top_k]:
            if not np.isfinite(scores[idx]) or scores[idx] <= 0:
                continue
            c = self.chunks[idx]
            results.append(_result(c["chunk_id"], c, c["text"], scores[idx]))
        return results


# ── Dense (ChromaDB) ───────────────────────────────────────────────────────────
class DenseRetriever:
    """Dense retriever: sentence-transformers embeddings stored in a persistent ChromaDB collection."""

    def __init__(
        self,
        model_name: str,
        persist_dir: str | None = None,
        collection_name: str = "nepal_gov_services",
        max_seq_length: int = 512,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        set_progress("model")
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.collection = None

    def _client(self):
        import chromadb
        if self.persist_dir:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            return chromadb.PersistentClient(path=self.persist_dir)
        return chromadb.Client()

    def _fingerprint(self, chunks: list[dict[str, Any]]) -> str:
        """Changes whenever the model, any chunk text or any stored metadata changes."""
        h = hashlib.sha1(f"{self.model_name}|{self.passage_prefix}|{self.model.max_seq_length}".encode())
        for c in chunks:
            meta = {k: c.get(k) for k in METADATA_FIELDS}
            h.update(f"{c['chunk_id']}|{c.get('search_text', '')}|{json.dumps(meta, sort_keys=True, ensure_ascii=False)}".encode())
        return h.hexdigest()

    def build_index(self, chunks: list[dict[str, Any]], force: bool = False) -> None:
        """Load the stored ChromaDB index; embed the chunks only when there is no complete index.

        The vector store is built ONCE (`uv run nepal-rag build` / `index --force`, which pass
        force=True) and then only loaded: the server, `ask`, `chat` and `eval` never re-embed.
        If the chunks changed since the last build, the existing index is still used and a
        warning says to run the build. Missing or half-built (interrupted) index -> built once."""
        client = self._client()
        fingerprint = self._fingerprint(chunks)

        if not force:
            try:
                existing = client.get_collection(self.collection_name)
            except Exception:
                existing = None  # no index yet -> build it once
            stored = (existing.metadata or {}).get("fingerprint") if existing is not None else None
            if existing is not None and stored not in (None, "incomplete") and existing.count() > 0:
                set_progress("reuse")
                self.collection = existing
                if stored != fingerprint:
                    logger.warning("The vector index is older than the chunks (%d stored, %d now) — using it anyway. "
                                   "Run `uv run nepal-rag build` to update it.", existing.count(), len(chunks))
                else:
                    logger.info("Loaded ChromaDB index (%d chunks, %s)", existing.count(), self.model_name)
                return
            logger.info("No complete vector index found — building it once.")

        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        # The fingerprint is stored only after every chunk is added, so an interrupted build
        # (server reload, Ctrl+C) is rebuilt next time instead of being reused half-empty.
        metadata = {"hnsw:space": "cosine", "embedding_model": self.model_name, "fingerprint": "incomplete"}
        self.collection = client.create_collection(name=self.collection_name, metadata=metadata)

        # Embedded and stored in batches so progress can be logged (and shown in the web page).
        logger.info("Embedding %d chunks with %s ...", len(chunks), self.model_name)
        step = 32
        for i in range(0, len(chunks), step):
            batch = chunks[i:i + step]
            embeddings = self.model.encode([self.passage_prefix + (c.get("search_text") or c["text"]) for c in batch],
                                           batch_size=16, normalize_embeddings=True, show_progress_bar=False)
            self.collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=embeddings.tolist(),
                documents=[c["text"] for c in batch],
                metadatas=[{k: c.get(k) if c.get(k) is not None else "" for k in METADATA_FIELDS} for c in batch],
            )
            done = min(i + step, len(chunks))
            set_progress("embedding", done, len(chunks))
            logger.info("Embedding chunks: %d/%d (%d%%)", done, len(chunks), 100 * done // len(chunks))
        self.collection.modify(metadata={k: v for k, v in {**metadata, "fingerprint": fingerprint}.items()
                                         if k != "hnsw:space"})
        logger.info("Dense index built with %d chunks", len(chunks))

    def is_current(self, chunks: list[dict[str, Any]]) -> bool:
        """True if the stored index was built from exactly these chunks with this model."""
        try:
            existing = self._client().get_collection(self.collection_name)
        except Exception:
            return False
        return (existing.metadata or {}).get("fingerprint") == self._fingerprint(chunks) and existing.count() > 0

    def search(self, query: str, top_k: int = 5, domain: str | None = None) -> list[RetrievalResult]:
        if self.collection is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        embedding = self.model.encode([self.query_prefix + query], normalize_embeddings=True)
        res = self.collection.query(
            query_embeddings=embedding.tolist(),
            n_results=max(1, min(top_k, self.collection.count())),
            where={"domain_id": domain} if domain else None,
        )
        out = []
        for i, chunk_id in enumerate(res["ids"][0] if res["ids"] else []):
            out.append(_result(chunk_id, res["metadatas"][0][i], res["documents"][0][i], 1.0 - res["distances"][0][i]))
        return out


# ── Hybrid ─────────────────────────────────────────────────────────────────────
class HybridRetriever:
    """Combines BM25 and dense rankings.

    fusion="rrf"      Reciprocal Rank Fusion: score = Σ weight / (k + rank). Uses ranks only, so a
                      passage the dense model ranks first still scores well when BM25 (which needs
                      exact word overlap) misses it.
    fusion="weighted" weighted sum of max-normalized scores.
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, bm25_weight: float = 0.4,
                 dense_weight: float = 0.6, fusion: str = "rrf", rrf_k: int = 60) -> None:
        self.bm25, self.dense = bm25, dense
        self.bm25_weight, self.dense_weight = bm25_weight, dense_weight
        self.fusion, self.rrf_k = fusion, rrf_k

    def build_index(self, chunks: list[dict[str, Any]], force: bool = False) -> None:
        set_progress("bm25")
        self.bm25.build_index(chunks)  # in memory, takes a second
        self.dense.build_index(chunks, force=force)

    def search(self, query: str, top_k: int = 5, domain: str | None = None) -> list[RetrievalResult]:
        merged: dict[str, dict[str, Any]] = {}
        depth = max(top_k * 4, 20)
        for weight, results in ((self.bm25_weight, self.bm25.search(query, depth, domain)),
                                (self.dense_weight, self.dense.search(query, depth, domain))):
            top = max((r.score for r in results), default=0.0)
            for rank, r in enumerate(results, 1):
                entry = merged.setdefault(r.chunk_id, {"result": r, "score": 0.0})
                if self.fusion == "rrf":
                    entry["score"] += weight / (self.rrf_k + rank)
                else:
                    entry["score"] += weight * (r.score / top if top > 0 else 0.0)
        final = []
        for entry in merged.values():
            entry["result"].score = entry["score"]
            final.append(entry["result"])
        final.sort(key=lambda r: r.score, reverse=True)
        return final[:top_k]


def make_retriever():
    """Build the retriever configured in settings.yaml (bm25 | dense | hybrid)."""
    cfg = get_settings()["retrieval"]
    bm25 = BM25Retriever(k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"])
    if cfg["mode"] == "bm25":
        return bm25
    dense = DenseRetriever(
        model_name=cfg["embedding_model"],
        persist_dir=cfg["chromadb"]["persist_dir"],
        collection_name=cfg["chromadb"]["collection_name"],
        max_seq_length=cfg.get("embedding_max_tokens", 512),
        query_prefix=cfg.get("query_prefix", ""),
        passage_prefix=cfg.get("passage_prefix", ""),
    )
    if cfg["mode"] == "dense":
        return dense
    return HybridRetriever(bm25, dense, cfg["hybrid"]["bm25_weight"], cfg["hybrid"]["dense_weight"],
                           fusion=cfg["hybrid"].get("fusion", "rrf"), rrf_k=cfg["hybrid"].get("rrf_k", 60))


def build_index(force_rebuild: bool = False) -> None:
    """Build the persistent ChromaDB vector store (data/index/chroma) — only when needed.

    The index is embedded once and then reused: if it already matches the chunks (fingerprint
    of model + chunk texts + metadata) nothing is embedded. `force_rebuild` re-embeds anyway."""
    chunks = load_chunks()
    if not chunks:
        logger.error("No chunks found. Run `uv run nepal-rag process` first.")
        return

    retriever = make_retriever()
    if isinstance(retriever, BM25Retriever):
        retriever.build_index(chunks)
        return
    dense = retriever.dense if isinstance(retriever, HybridRetriever) else retriever
    if not force_rebuild and dense.is_current(chunks):
        print(f"Vector index is already up to date ({len(chunks)} chunks) — nothing to embed.")
        return

    # A rebuild starts from an empty folder: Chroma keeps the files of dropped collections on
    # disk, so deleting the folder avoids orphaned segments piling up. Release our own handle first.
    import gc

    from chromadb.api.client import SharedSystemClient
    persist_dir = Path(get_settings()["retrieval"]["chromadb"]["persist_dir"])
    dense.collection = None
    SharedSystemClient.clear_system_cache()
    gc.collect()
    if persist_dir.exists():
        try:
            shutil.rmtree(persist_dir)
        except PermissionError as exc:  # Windows: files are open in another process
            raise SystemExit(
                f"Cannot rebuild: {persist_dir} is in use by another process — stop the running "
                f"server (`fastapi dev/run`, Ctrl+C) and run the build again.\n({exc})"
            ) from exc
    dense.build_index(chunks, force=True)
    logger.info("Index built (mode=%s, %d chunks)", get_settings()["retrieval"]["mode"], len(chunks))

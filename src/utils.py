"""Shared utilities for the Nepal RAG Agent."""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Repo root = the folder containing src/ and config/ (independent of the current directory).
REPO_ROOT = Path(__file__).resolve().parent.parent

# Load API keys / LLM settings from .env once, for both the CLI and the FastAPI app.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save data to a YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data to a JSON file."""
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def normalize_nepali_text(text: str) -> str:
    """Normalize Nepali/Unicode text for consistent processing.

    - NFC normalization
    - Remove zero-width characters
    - Normalize whitespace: collapse all whitespace (spaces, newlines, tabs) to single spaces
    - Remove control characters
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Replace all whitespace (spaces, newlines, tabs) with single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 100) -> str:
    """Truncate text for display."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


@lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    """Load and cache config/settings.yaml. Relative paths are made absolute (repo-root based)."""
    settings = load_yaml(REPO_ROOT / "config" / "settings.yaml")
    for key, value in settings.get("paths", {}).items():
        settings["paths"][key] = str(REPO_ROOT / value)
    chroma = settings.get("retrieval", {}).get("chromadb", {})
    if chroma.get("persist_dir"):
        chroma["persist_dir"] = str(REPO_ROOT / chroma["persist_dir"])
    return settings


def find_repo_root() -> Path:
    """Return the repository root (folder containing config/ and src/)."""
    return REPO_ROOT


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the project."""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level.upper(), format=log_format, handlers=handlers)


@lru_cache(maxsize=1)
def get_token_counter():
    """Token counter using the embedding model's own tokenizer, so chunk budgets match
    what the model actually reads. Falls back to a character heuristic if unavailable."""
    name = get_settings()["retrieval"]["embedding_model"]
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.model_max_length = 10**9  # we only count; silence "sequence too long" warnings
        return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception as exc:  # pragma: no cover
        logger.warning("Tokenizer for %s unavailable (%s) — using approximate token counts", name, exc)
        return count_tokens_approx


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough heuristic for Devanagari + English)."""
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    english = len(re.findall(r"[a-zA-Z]+", text))
    numbers = len(re.findall(r"\d+", text))
    return devanagari + int(english * 1.3) + numbers

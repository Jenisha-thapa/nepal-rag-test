"""Core unit tests: HTML extraction, chunking, knowledge base loader, retrieval, raw-file selection, API."""

from pathlib import Path

import pytest

from src.processing.html_extract import Block, extract_html
from src.utils import count_tokens_approx, normalize_nepali_text

COUNT = count_tokens_approx  # fast, deterministic token counter for tests


# ── utils ──────────────────────────────────────────────────────────────────────
def test_normalize_removes_zero_width_and_collapses_whitespace():
    assert normalize_nepali_text("नेपाली​") == "नेपाली"
    assert normalize_nepali_text("  Hello\n\n  World  ") == "Hello World"


# ── HTML extraction ────────────────────────────────────────────────────────────
PAGE = """
<html><head><title>FAQ | Dept</title></head><body>
<header class="header-7"><nav><a href="/">गृह पृष्ठ</a></nav>
  <div class="marquee-slider"><h2>महत्त्वपूर्ण सूचना</h2><h3>स्वास्थ्य बीमा नवीकरण सम्बन्धमा सूचना</h3></div>
</header>
<main>
  <h2>प्राय सोधिने प्रश्नहरु</h2>
  <p>१ . प्रश्न:</p><p>भत्ता दर कति छ ?</p><p>उत्तर:</p><p>भत्ता दर देहाय अनुसार छ।</p>
  <table>
    <tr><th>क्र.सं.</th><th>लाभान्वित वर्ग</th><th>मासिक रकम (रु.)</th></tr>
    <tr><td>१</td><td>जेष्ठ नागरिक</td><td>४०००</td></tr>
  </table>
  <p>२ . प्रश्न:</p><p>निवेदन कहाँ दिने ?</p><p>उत्तर:</p><p>वडा कार्यालयमा निवेदन दिनुपर्छ।</p>
  <ul class="sidebar__nav"><li>मेनु</li></ul>
</main>
<footer>© Government of Nepal</footer>
</body></html>
"""


def test_extract_keeps_structure_and_drops_chrome():
    page = extract_html(PAGE)
    texts = [b.text for b in page.blocks]
    assert page.title == "FAQ"
    assert any(b.kind == "heading" and b.text == "प्राय सोधिने प्रश्नहरु" for b in page.blocks)
    # table row is self-contained "header: value" text, serial-number column dropped
    assert "लाभान्वित वर्ग: जेष्ठ नागरिक; मासिक रकम (रु.): ४०००" in texts
    # nav / footer / sidebar menu removed
    assert not any("गृह पृष्ठ" in t or "Government of Nepal" in t or t == "मेनु" for t in texts)
    # notice ticker inside <header> is kept as content
    assert "स्वास्थ्य बीमा नवीकरण सम्बन्धमा सूचना" in texts


def test_accordion_faq_questions_are_kept():
    """Regression: FAQ questions inside <h2><button> were dropped with the buttons."""
    html = ('<main><h2 class="accordion__header"><button>स्वास्थ्य बीमा गर्दा कति शुल्क लाग्छ ?</button></h2>'
            '<div class="accordion__body"><p>प्रति परिवार रु.३,५००/- लाग्छ।</p></div>'
            '<button>भाषा परिवर्तन गर्नुहोस्</button></main>')
    blocks = extract_html(html).blocks
    assert any(b.kind == "heading" and "कति शुल्क लाग्छ" in b.text for b in blocks)
    assert not any("भाषा परिवर्तन" in b.text for b in blocks)  # UI buttons still dropped


# ── chunking ───────────────────────────────────────────────────────────────────
def _doc(blocks):
    return {"doc_id": "doc", "domain_id": "health_social_welfare", "source_name": "Test FAQ",
            "source_url": "https://example.gov.np", "page_title": "FAQ", "blocks": blocks}


def test_faq_question_and_full_answer_form_one_chunk():
    from src.chunking.structure_aware import chunk_document
    chunks = chunk_document(_doc(extract_html(PAGE).blocks), COUNT, max_tokens=400, min_tokens=10)
    faq = [c for c in chunks if c["content_type"] == "faq"]
    assert len(faq) == 2
    assert faq[0]["text"].startswith("प्रश्न: भत्ता दर कति छ ?")
    assert "४०००" in faq[0]["text"]  # the table belongs to the answer
    assert faq[1]["text"].startswith("प्रश्न: निवेदन कहाँ दिने ?")


def test_chunk_metadata_and_search_text():
    from src.chunking.structure_aware import chunk_document
    c = chunk_document(_doc(extract_html(PAGE).blocks), COUNT, max_tokens=400, min_tokens=10)[0]
    for key in ("chunk_id", "domain_id", "source_url", "section", "content_type", "language",
                "token_count", "content_hash", "text", "search_text"):
        assert key in c
    assert c["search_text"].startswith("FAQ")  # page title is part of what gets embedded
    assert c["language"] == "ne"


def test_long_faq_answer_is_split_and_repeats_question():
    from src.chunking.structure_aware import chunk_document
    blocks = [Block("text", "१ . प्रश्न:"), Block("text", "लामो प्रश्न ?"), Block("text", "उत्तर:")]
    blocks += [Block("item", f"बुँदा नम्बर {i} को विस्तृत विवरण यहाँ छ।") for i in range(40)]
    chunks = chunk_document(_doc(blocks), COUNT, max_tokens=120, min_tokens=10)
    assert len(chunks) > 1
    assert all(c["text"].startswith("प्रश्न: लामो प्रश्न ?") for c in chunks)
    assert all(c["token_count"] <= 120 for c in chunks)


# ── hand-written knowledge base ────────────────────────────────────────────────
KB_YAML = """
domain_id: land_property
chunks:
  - id: officials
    url: https://dolma.gov.np/
    section: पदाधिकारी
    type: contact
    text: |
      सूचना अधिकारी: छत्र बहादुर देवकोटा, 9764398636
"""


def test_knowledge_base_fills_metadata_from_crawl(tmp_path: Path):
    from src.chunking.knowledge_base import load_knowledge_base
    (tmp_path / "land_property.yaml").write_text(KB_YAML, encoding="utf-8")
    crawl = {"https://dolma.gov.np": {"source_name": "DoLMA — Official Portal", "fetched_at": "2026-11-23T10:00:00",
                                      "document_type": "portal", "saved_path": "data/raw/land_property/DoLMA__x.html"}}
    [c] = load_knowledge_base(COUNT, max_tokens=450, kb_dir=tmp_path, crawl_records=crawl)
    assert c["chunk_id"] == "land_property::officials"
    # display name: the source's name in config/domains.yaml wins over the crawl record's
    assert c["source_name"] == "Department of Land Management and Archive — Official Portal"
    assert c["fetched_at"].startswith("2026-11-23") and c["content_type"] == "contact"  # crawl record, by URL
    assert c["authority"]                                   # from config/domains.yaml
    assert c["search_text"].startswith(c["source_name"] + " › पदाधिकारी")


def test_knowledge_base_rejects_missing_fields_and_duplicates(tmp_path: Path):
    from src.chunking.knowledge_base import load_knowledge_base
    (tmp_path / "a.yaml").write_text("domain_id: x\nchunks:\n  - id: a\n    url: u\n    text: t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_knowledge_base(COUNT, 450, kb_dir=tmp_path, crawl_records={})
    (tmp_path / "a.yaml").write_text(KB_YAML + KB_YAML.split("chunks:\n")[1], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_knowledge_base(COUNT, 450, kb_dir=tmp_path, crawl_records={})


# ── retrieval ──────────────────────────────────────────────────────────────────
def test_bm25_ranks_matching_chunk_first_and_filters_domain():
    from src.retrieval.indexer import BM25Retriever
    chunks = [
        {"chunk_id": "a", "domain_id": "land_property", "text": "जग्गा नामसारी गर्ने प्रक्रिया हुन्छ।"},
        {"chunk_id": "b", "domain_id": "health_social_welfare", "text": "स्वास्थ्य बीमा कसरी लिने हुन्छ।"},
        {"chunk_id": "c", "domain_id": "employment_labour", "text": "श्रम स्वीकृति कसरी लिने।"},
        {"chunk_id": "d", "domain_id": "identity_civil_registration", "text": "नागरिकता प्राप्त गर्न के गर्ने?"},
    ]
    r = BM25Retriever()
    r.build_index(chunks)
    assert r.search("जग्गा नामसारी कसरी गर्ने?")[0].chunk_id == "a"
    assert all(x.domain_id == "health_social_welfare" for x in r.search("कसरी लिने", domain="health_social_welfare"))


def test_bm25_rejects_chunks_without_text():
    """Regression: documents (cleaned_text) instead of chunks (text) used to crash with ZeroDivisionError."""
    from src.retrieval.indexer import BM25Retriever
    with pytest.raises(ValueError, match="process"):
        BM25Retriever().build_index([{"chunk_id": "x", "cleaned_text": "नेपाली"}])


# ── processing ─────────────────────────────────────────────────────────────────
def test_latest_raw_file_per_source(tmp_path: Path):
    from src.processing.pipeline import _latest_raw_files
    d = tmp_path / "land_property"
    d.mkdir()
    for name in ["DoLMA__20260101T000000Z.html", "DoLMA__20261119T000000Z.html", "Other__20260101T000000Z.html"]:
        (d / name).write_text("<html>x</html>", encoding="utf-8")
    names = sorted(p.name for p in _latest_raw_files(tmp_path))
    assert names == ["DoLMA__20261119T000000Z.html", "Other__20260101T000000Z.html"]


# ── API ────────────────────────────────────────────────────────────────────────
def test_chat_returns_503_before_engine_loaded():
    from fastapi.testclient import TestClient
    from src.main import app
    client = TestClient(app)  # not used as a context manager -> lifespan (model loading) doesn't run
    assert client.post("/api/chat", json={"message": "नमस्ते"}).status_code == 503


def test_chat_validates_empty_message():
    from fastapi.testclient import TestClient
    from src.main import app
    assert TestClient(app).post("/api/chat", json={"message": ""}).status_code == 422

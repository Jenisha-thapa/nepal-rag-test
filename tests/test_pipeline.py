"""Tests for the RAG pipeline steps (query analysis, fused retrieval, prompt) and the conversation store/API.
No embedding model or LLM is loaded: retrieval uses BM25 and the LLM is a fake."""

from pathlib import Path

import pytest

from src.generation.chat import ChatSession, CitationGroundedGenerator, QueryAnalysis, QueryAnalyzer
from src.storage import ConversationStore

DOMAINS = [{"id": "land_property", "name_en": "Land", "description": "land registration"},
           {"id": "health_social_welfare", "name_en": "Health", "description": "allowances"}]


class FakeLLM:
    label = "fake/llm"

    def __init__(self, reply: str) -> None:
        self.reply, self.prompts = reply, []

    def generate(self, messages, temperature=0.0, max_tokens=0) -> str:
        self.prompts.append(messages[-1]["content"])
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


# ── step 1: query analysis ────────────────────────────────────────────────────
def test_analyzer_parses_llm_json_and_keeps_known_domain_only():
    llm = FakeLLM('Sure!\n{"domain": "land_property", "standalone_question": "How do I register land?",'
                  ' "search_queries": ["जग्गा रजिष्ट्रेशन कसरी गर्ने?", "land registration process"]}')
    a = QueryAnalyzer([llm], DOMAINS, keyword_domain=lambda q: "").analyze(
        "and for land?", [{"role": "user", "content": "How do I get a citizenship?"}])
    assert (a.method, a.domain_id, a.standalone) == ("llm", "land_property", "How do I register land?")
    assert a.queries == ["and for land?", "How do I register land?", "जग्गा रजिष्ट्रेशन कसरी गर्ने?",
                         "land registration process"]
    assert "How do I get a citizenship?" in llm.prompts[0]  # follow-ups see the conversation

    bad_domain = FakeLLM('{"domain": "taxes", "standalone_question": "x", "search_queries": []}')
    assert QueryAnalyzer([bad_domain], DOMAINS, lambda q: "").analyze("x").domain_id == ""


def test_analyzer_falls_back_to_keywords_when_llm_fails():
    for llm in (FakeLLM("not json at all"), FakeLLM(RuntimeError("503"))):
        a = QueryAnalyzer([llm], DOMAINS, keyword_domain=lambda q: "land_property").analyze("जग्गा?")
        assert (a.method, a.domain_id, a.queries) == ("keywords", "land_property", ["जग्गा?"])


# ── step 2: fused retrieval ───────────────────────────────────────────────────
def _bm25_session(chunks):
    from src.retrieval.indexer import BM25Retriever
    session = ChatSession.__new__(ChatSession)  # skip loading models
    session.chunks = chunks
    session.retriever = BM25Retriever()
    session.retriever.build_index(chunks)
    return session


def test_retrieve_puts_domain_first_but_keeps_other_domains():
    chunks = [
        {"chunk_id": "h", "domain_id": "health_social_welfare", "text": "बीमा बीमा बीमा रकम तिर्ने"},
        {"chunk_id": "e", "domain_id": "employment_labour", "text": "वैदेशिक रोजगार बीमा रकम"},
        {"chunk_id": "l", "domain_id": "land_property", "text": "जग्गा नामसारी"},
    ]
    s = _bm25_session(chunks)
    analysis = QueryAnalysis(question="बीमा रकम", standalone="बीमा रकम", domain_id="employment_labour")
    domain, results = s.retrieve("बीमा रकम", analysis=analysis)
    ids = [r.chunk_id for r in results]
    assert domain == "employment_labour" and ids[0] == "e" and "h" in ids  # cross-domain match still reaches the LLM


def test_retrieve_uses_paraphrases():
    chunks = [{"chunk_id": "reg", "domain_id": "land_property", "text": "जग्गा रजिष्ट्रेशन मालपोत कार्यालयमा हुन्छ"},
              {"chunk_id": "x", "domain_id": "land_property", "text": "अरू विषय"},
              {"chunk_id": "y", "domain_id": "land_property", "text": "अर्को कुरा"}]  # BM25 idf needs >2 docs
    s = _bm25_session(chunks)
    plain = QueryAnalysis(question="register land", standalone="register land")
    assert "reg" not in [r.chunk_id for r in s.retrieve("register land", analysis=plain)[1]]  # English vs Nepali text
    para = QueryAnalysis(question="register land", standalone="register land", search_queries=["जग्गा रजिष्ट्रेशन"])
    assert s.retrieve("register land", analysis=para)[1][0].chunk_id == "reg"


# ── step 3: prompt ────────────────────────────────────────────────────────────
def test_prompt_has_sections_and_language_directive():
    gen = CitationGroundedGenerator.__new__(CitationGroundedGenerator)
    p = gen._build_user_prompt("and the fee?", "[स्रोत 1] ...", standalone="What is the passport fee?",
                               history=[{"role": "user", "content": "passport"}, {"role": "assistant", "content": "ok"}])
    order = [p.index(h) for h in ("## CONVERSATION SO FAR", "## QUESTION", "## THE QUESTION UNDERSTOOD AS",
                                  "## OFFICIAL SOURCE PASSAGES", "## TASK")]
    assert order == sorted(order) and "WHOLE answer in English" in p
    assert "write the whole answer in Nepali" in gen._build_user_prompt("शुल्क कति?", "ctx")
    assert "CONVERSATION" not in gen._build_user_prompt("शुल्क कति?", "ctx")


# ── conversation store + API ──────────────────────────────────────────────────
def test_store_scopes_conversations_to_user(tmp_path: Path):
    store = ConversationStore(tmp_path / "c.db")
    conv = store.create("u_alice", "जग्गा नामसारी")
    store.add_message(conv["id"], "user", "q1")
    store.add_message(conv["id"], "assistant", "a1", {"sources": [{"url": "https://x"}]})
    assert [(c["message_count"], c["domain_id"]) for c in store.list("u_alice")] == [(2, "")]
    answer_id = store.add_message(conv["id"], "assistant", "a2", {"domain_id": "land_property"})
    assert store.list("u_alice")[0]["domain_id"] == "land_property"
    assert not store.set_feedback("u_bob", answer_id, True) and store.set_feedback("u_alice", answer_id, True)
    assert store.get("u_alice", conv["id"])["messages"][-1]["meta"]["helpful"] is True
    assert store.list("u_bob") == [] and store.get("u_bob", conv["id"]) is None
    assert store.get("u_alice", conv["id"])["messages"][1]["meta"]["sources"][0]["url"] == "https://x"
    assert store.history(conv["id"], limit=1) == [{"role": "assistant", "content": "a2"}]
    assert not store.rename("u_bob", conv["id"], "x") and store.rename("u_alice", conv["id"], "Land")
    assert not store.delete("u_bob", conv["id"]) and store.delete("u_alice", conv["id"])
    assert store.get("u_alice", conv["id"]) is None


@pytest.fixture
def client(tmp_path: Path):
    from fastapi.testclient import TestClient
    from src.main import app, state
    state["store"] = ConversationStore(tmp_path / "api.db")
    yield TestClient(app)  # no lifespan: the RAG engine is not loaded
    state["store"] = None


def test_conversation_api(client):
    from src.main import state
    conv = state["store"].create("u_test1", "first")
    assert [c["id"] for c in client.get("/api/conversations?user_id=u_test1").json()] == [conv["id"]]
    assert client.patch(f"/api/conversations/{conv['id']}?user_id=u_test1", json={"title": "Renamed"}).json()["title"] == "Renamed"
    assert client.get(f"/api/conversations/{conv['id']}?user_id=u_other").status_code == 404
    assert client.get("/api/conversations?user_id=bad id!").status_code == 422
    assert client.delete(f"/api/conversations/{conv['id']}?user_id=u_test1").json() == {"deleted": 1}
    state["store"].create("u_test1", "a"), state["store"].create("u_test1", "b")
    assert client.delete("/api/conversations?user_id=u_test1").json() == {"deleted": 2}


def test_chat_rejects_bad_ids(client):
    assert client.post("/api/chat", json={"message": "hi", "user_id": "x y"}).status_code == 422
    assert client.post("/api/chat", json={"message": "hi", "conversation_id": "../etc"}).status_code == 422

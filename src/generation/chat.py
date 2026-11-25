"""
The RAG pipeline (ChatSession.ask):

  1. Understand  QueryAnalyzer — one short LLM call: classify the domain, rewrite a follow-up
                 as a standalone question, paraphrase it (formal Nepali + English).
                 Without an LLM: keyword domain classifier, the question as typed.
  2. Retrieve    every query searched inside the domain (metadata filter) and across the whole
                 knowledge base, fused with Reciprocal Rank Fusion -> top-k passages.
  3. Answer      CitationGroundedGenerator — evidence-only answer with [स्रोत n] citations.
                 LLM chain: primary -> fallback (e.g. Ollama -> Gemini) -> extractive answer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    """A generated response with citations."""
    answer: str
    citations: list[dict[str, Any]]
    domain_id: str
    confidence: float
    refused: bool = False
    refusal_reason: str = ""
    llm_used: str = ""  # e.g. "gemini/gemini-3.8-flash" or "extractive-fallback"
    analysis: dict[str, Any] = field(default_factory=dict)  # step 1: domain, standalone question, paraphrases


def ollama_ready(base_url: str, model: str, timeout: float = 2.0) -> tuple[bool, str]:
    """True if the Ollama server is up AND the model is pulled."""
    import requests
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        return False, f"Ollama not reachable at {base_url} ({e.__class__.__name__})"
    names = [m.get("name", "") for m in resp.json().get("models", [])]
    if model in names or f"{model}:latest" in names:
        return True, ""
    return False, f"Ollama running but model '{model}' not pulled (available: {names or 'none'})"


class LLMClient:
    """Unified LLM client for OpenAI, OpenRouter, Anthropic, Gemini, and Ollama."""

    def __init__(self, provider: str, model: str, api_key: str = "", base_url: str = "") -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.1, max_tokens: int = 1024) -> str:
        if self.provider == "openai":
            return self._call_openai(messages, temperature, max_tokens)
        elif self.provider == "openrouter":
            return self._call_openrouter(messages, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages, temperature, max_tokens)
        elif self.provider == "gemini":
            return self._call_gemini(messages, temperature, max_tokens)
        elif self.provider == "ollama":
            return self._call_ollama(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    def _call_gemini(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.api_key)
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user",
                          parts=[types.Part(text=m["content"])])
            for m in messages if m["role"] != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        # Retry transient errors (503 overloaded / 429 rate limit) with backoff.
        import time
        delays = [2, 5, 10]
        for attempt in range(len(delays) + 1):
            try:
                response = client.models.generate_content(model=self.model, contents=contents, config=config)
                break
            except Exception as e:
                transient = any(s in str(e) for s in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                if not transient or attempt == len(delays):
                    raise
                logger.warning("Gemini busy (%s) — retrying in %ss", str(e)[:60], delays[attempt])
                time.sleep(delays[attempt])
        if not (response.text or "").strip():
            raise RuntimeError("Gemini returned an empty response")
        return response.text

    def _call_openai(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_openrouter(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        system = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)
        response = client.messages.create(
            model=self.model, system=system, messages=user_messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return response.content[0].text if response.content else ""

    def _call_ollama(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        import ollama
        client = ollama.Client(host=self.base_url) if self.base_url else ollama
        response = client.chat(
            model=self.model, messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"]


class CitationGroundedGenerator:
    """Generates citation-grounded answers from retrieved context using an LLM."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3",
        system_language: str = "nepali",
    ) -> None:
        settings = get_settings()
        self.provider = provider
        self.model = model
        self.system_language = system_language
        self.temperature = settings["generation"]["temperature"]
        self.max_tokens = settings["generation"]["max_tokens"]
        self.refusal_message = settings["generation"]["refusal"]["message"]
        self.fallback_provider = os.getenv("LLM_FALLBACK_PROVIDER") or settings["generation"].get("fallback_provider", "")
        self.fallback_model = os.getenv("LLM_FALLBACK_MODEL") or settings["generation"].get("fallback_model", "")

        # Ordered chain: primary provider first, then the fallback (e.g. Ollama -> Gemini).
        primary = self._init_llm(os.getenv("LLM_PROVIDER", self.provider), os.getenv("LLM_MODEL", self.model))
        self.llms: list[LLMClient] = [primary] if primary else []
        if self.fallback_provider and (not primary or primary.provider != self.fallback_provider):
            fallback = self._init_llm(self.fallback_provider, self.fallback_model)
            if fallback:
                self.llms.append(fallback)
        self.llm = self.llms[0] if self.llms else None
        logger.info("LLM chain: %s", " -> ".join(c.label for c in self.llms) + " -> extractive" if self.llms else "extractive only")

    def _init_llm(self, provider: str, model: str) -> LLMClient | None:
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key") or os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                logger.warning("GEMINI_API_KEY not set — skipping Gemini")
                return None
            return LLMClient("gemini", os.getenv("GEMINI_MODEL") or model, api_key=api_key)

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key or api_key == "":
                logger.warning("OPENAI_API_KEY not set — using fallback generation")
                return None
            return LLMClient("openai", model, api_key=api_key)

        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key or api_key == "":
                logger.warning("OPENROUTER_API_KEY not set — using fallback generation")
                return None
            return LLMClient("openrouter", model, api_key=api_key)

        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key or api_key == "":
                logger.warning("ANTHROPIC_API_KEY not set — using fallback generation")
                return None
            return LLMClient("anthropic", model, api_key=api_key)

        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL") or model
            ready, reason = ollama_ready(base_url, model)
            if not ready:
                logger.warning("%s — skipping Ollama", reason)
                return None
            return LLMClient("ollama", model, base_url=base_url)

        return None

    REFUSAL_EN = "Sorry, the official sources I have do not contain enough information to answer this. Please contact the relevant office."

    def _build_system_prompt(self) -> str:
        # The LLM is the "reader/writer" of the RAG pipeline: it never answers from its own
        # knowledge, only from the numbered official passages the retriever puts in the prompt.
        return f"""You are "Nepal Sewa AI" (नेपाल सेवा AI), an assistant for Nepal government public services.
You answer ONLY from the numbered official source passages given in the user message ([स्रोत 1], [स्रोत 2], ...).

GROUNDING
1. Use only facts stated in the passages. Never add fees, deadlines, documents, names, phone numbers, laws or steps from your own knowledge. The conversation so far only tells you what a follow-up question refers to; never take facts from earlier answers.
2. The passages come from a search engine; some may be about a different topic. Use EVERY passage that is relevant to the question (combine facts from several passages: process, documents, fees, office, contacts) and ignore the unrelated ones.
3. Be helpful: if the passages answer only part of the question, answer that part fully and then say in one short sentence which part the official sources don't cover and which office to ask (if a passage names one). Questions can be worded in any way and do not need to match an FAQ; answer whenever a passage is relevant.
4. Only if NO passage is related to the question at all, reply with exactly this one sentence and nothing else:
   - Nepali question: "{self.refusal_message}"
   - English question: "{self.REFUSAL_EN}"
5. If passages disagree, give both and mention their dates.
6. For "latest / recent / हालैको" questions, state the date written in the passage; never call an old item recent. If no date is given, say so.

LANGUAGE
7. Answer entirely in the language of the question (Nepali question → Nepali answer, English question → English answer). Keep official names, numbers, phone numbers, emails and URLs exactly as in the passages.

FORMAT
8. Start directly with the answer in 1–3 short sentences. No headings and no labels such as "सङ्क्षिप्त उत्तर" or "Short answer".
9. Then, only if the passages contain them, add short "- " bullet points for: required documents, steps, fees/amounts, time limits, where to go, contacts. Omit any item the passages don't mention; never write "उपलब्ध छैन" or "not available".
10. Put the passage number right after each fact it supports, e.g. "मासिक रु. ४००० [स्रोत 1]". Use **bold** only for the single key figure or answer.
11. End with one final line giving the official page(s) you used (at most 2, copied from the passages' URL lines):
   - Nepali: "थप जानकारी: <URL>"
   - English: "More information: <URL>"
   Do not write a separate list of sources."""

    def _build_user_prompt(self, query: str, context: str, standalone: str = "",
                           history: list[dict[str, str]] | None = None) -> str:
        """Sectioned prompt: conversation (context only) → question → how it was understood →
        sources → task. Facts may come only from the SOURCES section."""
        # Most passages are Nepali, which pulls the model into Nepali; state the language explicitly.
        if re.search(r"[ऀ-ॿ]", query):
            language = "The question is in Nepali: write the whole answer in Nepali."
        else:
            language = ("The question is in English: write the WHOLE answer in English, translating the facts "
                        "from the Nepali passages (keep names, numbers and URLs exact), and end with "
                        "\"More information: <URL>\".")
        sections = []
        if history:
            turns = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:400]}"
                              for m in history[-4:])
            sections.append("## CONVERSATION SO FAR (only to understand the question — not a source of facts)\n"
                            + turns)
        sections.append(f"## QUESTION\n{query}")
        if standalone and standalone.strip() != query.strip():
            sections.append(f"## THE QUESTION UNDERSTOOD AS\n{standalone}")
        sections.append(f"## OFFICIAL SOURCE PASSAGES\n{context}")
        sections.append(f"## TASK\nAnswer the QUESTION using only the OFFICIAL SOURCE PASSAGES, following all the "
                        f"rules. {language}")
        return "\n\n".join(sections)

    def _clean_text(self, text: str) -> str:
        """Remove punctuation for comparison."""
        return re.sub(r'[।?!.,;:\[\]{}()@#$]','', text.lower())

    def _keyword_score(self, query: str, text: str) -> int:
        """Score how relevant a text is to the query."""
        q = self._clean_text(query)
        t = self._clean_text(text)
        score = 0
        for word in q.split():
            if word in t:
                score += 1
        if q in t:
            score += 10
        return score

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences (handles Nepali and English)."""
        sentences = re.split(r'[।!?\.]\s*', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _score_sentence(self, sentence: str, query: str) -> float:
        """Score a sentence's relevance to the query."""
        q = self._clean_text(query)
        s = self._clean_text(sentence)
        query_words = set(q.split())
        sentence_words = set(s.split())
        if not query_words:
            return 0.0
        intersection = query_words & sentence_words
        union = query_words | sentence_words
        jaccard = len(intersection) / len(union) if union else 0.0
        overlap = len(intersection) / len(query_words) if query_words else 0.0
        phrase_bonus = 0.5 if q in s else 0.0
        return jaccard * 0.3 + overlap * 0.5 + phrase_bonus

    def _extract_key_info(self, text: str) -> dict[str, list[str]]:
        """Extract key information like fees, documents, time periods."""
        info = {'fees': [], 'documents': [], 'time': [], 'process': []}
        fee_patterns = [r'रु[\s\.]?\d+', r'Rs[\s\.]?\d+', r'शुल्क[:\s]*[^\n]+', r'fee[:\s]*[^\n]+', r'\d+\s*रुपैयाँ']
        for pattern in fee_patterns:
            info['fees'].extend(re.findall(pattern, text, re.IGNORECASE))
        doc_patterns = [r'कागजात[:\s]*[^\n]+', r'दस्तावेज[:\s]*[^\n]+', r'document[:\s]*[^\n]+', r'प्रमाणपत्र[:\s]*[^\n]+']
        for pattern in doc_patterns:
            info['documents'].extend(re.findall(pattern, text, re.IGNORECASE))
        time_patterns = [r'\d+\s*दिन', r'\d+\s*महिना', r'\d+\s*वर्ष', r'\d+\s*days?', r'\d+\s*months?', r'समय[:\s]*[^\n]+']
        for pattern in time_patterns:
            info['time'].extend(re.findall(pattern, text, re.IGNORECASE))
        return info

    def _parse_qa_pairs(self, text: str) -> list[dict[str, str]]:
        """Parse FAQ-style text into Q&A pairs."""
        qa_pairs = []
        patterns = [
            r'प्रश्न[:\s]*(.+?)\s*उत्तर[:\s]*(.+?)(?=प्रश्न[:\s]|$)',
            r'Question[:\s]*(.+?)\s*Answer[:\s]*(.+?)(?=Question[:\s]|$)',
            r'[०-९\d]+[.\s]*प्रश्न[:\s]*(.+?)\s*उत्तर[:\s]*(.+?)(?=[०-९\d]+[.\s]*प्रश्न[:\s]|$)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for q, a in matches:
                q_clean = q.strip()
                a_clean = a.strip()
                if len(q_clean) > 5 and len(a_clean) > 10:
                    qa_pairs.append({'question': q_clean, 'answer': a_clean})
        return qa_pairs

    def _score_qa_relevance(self, query: str, qa_question: str) -> float:
        """Score how relevant a QA pair's question is to the user's query."""
        q = self._clean_text(query)
        qa_q = self._clean_text(qa_question)
        if not q or not qa_q:
            return 0.0
        query_words = set(q.split())
        qa_words = set(qa_q.split())
        if not query_words:
            return 0.0
        intersection = query_words & qa_words
        union = query_words | qa_words
        jaccard = len(intersection) / len(union) if union else 0.0
        overlap = len(intersection) / len(query_words) if query_words else 0.0
        phrase_bonus = 0.5 if (q in qa_q or qa_q in q) else 0.0
        key_terms = {'नागरिकता', 'nagarikta', 'citizenship', 'राष्ट्रिय', 'national', 'परिचयपत्र', 'id', 'card', 'बनाउन', 'कसरी', 'कति', 'समय', 'शुल्क', 'कागजात', 'दस्तावेज', 'प्रक्रिया', 'जग्गा', 'नामसारी', 'land', 'ssf', 'सामाजिक', 'श्रम', 'labour', 'स्वास्थ्य', 'health', 'बीमा', 'insurance'}
        key_matches = sum(1 for term in key_terms if term in q and term in qa_q)
        key_bonus = key_matches * 0.1
        return jaccard * 0.2 + overlap * 0.4 + phrase_bonus + key_bonus

    def _fallback_generate(self, query: str, retrieval_results: list[Any]) -> str:
        """Generate answer from context without LLM — intelligent Q&A extraction."""
        if not retrieval_results:
            return self.refusal_message

        all_qa_pairs = []
        for result in retrieval_results:
            qa_pairs = self._parse_qa_pairs(result.text)
            for qa in qa_pairs:
                relevance = self._score_qa_relevance(query, qa['question'])
                all_qa_pairs.append({**qa, 'relevance': relevance, 'source': result.source_name, 'url': result.source_url})

        all_qa_pairs.sort(key=lambda x: x['relevance'], reverse=True)
        top_qa = [qa for qa in all_qa_pairs if qa['relevance'] > 0.15][:3]

        if not top_qa:
            return self._extract_key_sentences(query, retrieval_results)

        parts = []
        for qa in top_qa:
            answer_text = re.sub(r'\s+', ' ', qa['answer']).strip()
            if len(answer_text) > 500:
                answer_text = answer_text[:500] + "..."
            parts.append(answer_text)

        answer = "\n\n".join(parts)
        sources_used = list(set(qa['source'] for qa in top_qa))
        if sources_used:
            answer += f"\n\nस्रोत (Sources): {', '.join(sources_used)}"

        return answer

    def _extract_key_sentences(self, query: str, retrieval_results: list[Any]) -> str:
        """Extract key sentences from raw text when no Q&A pairs found."""
        all_sentences = []
        for result in retrieval_results:
            sentences = self._split_into_sentences(result.text)
            for sent in sentences:
                score = self._score_sentence(sent, query)
                all_sentences.append({'text': sent, 'score': score, 'source': result.source_name})

        all_sentences.sort(key=lambda x: x['score'], reverse=True)
        top_sentences = [s for s in all_sentences if s['score'] > 0.1][:3]

        if not top_sentences:
            return self.refusal_message

        parts = [s['text'] for s in top_sentences]
        answer = " ".join(parts)
        sources_used = list(set(s['source'] for s in top_sentences))
        if sources_used:
            answer += f"\n\nस्रोत (Sources): {', '.join(sources_used)}"

        return answer

    def generate(
        self,
        query: str,
        retrieval_results: list[Any],
        domain_id: str = "",
        standalone: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> GeneratedResponse:
        """Generate a citation-grounded answer."""
        if not retrieval_results:
            return GeneratedResponse(
                answer=self.refusal_message,
                citations=[],
                domain_id=domain_id,
                confidence=0.0,
                refused=True,
                refusal_reason="No results",
            )

        citations = []
        for i, result in enumerate(retrieval_results, 1):
            meta = result.metadata
            citations.append({
                "index": i,
                "name": result.source_name,
                "url": result.source_url,
                "domain_id": result.domain_id,
                "page_title": result.title,
                "section": result.section,
                "content_type": meta.get("content_type", ""),
                "date_bs": meta.get("date_bs", ""),
                "score": round(float(result.score), 4),
                "text": result.text,
            })

        # Each passage carries where it comes from (page › section, date) so the LLM can
        # cite precisely and say how current the information is.
        context_parts = []
        for i, result in enumerate(retrieval_results, 1):
            where = " › ".join(p for p in (result.source_name, result.section) if p)
            date = result.metadata.get("date_bs")
            context_parts.append(
                f"[स्रोत {i}] {where}\nURL: {result.source_url}" + (f"\nमिति: {date}" if date else "") + f"\n\n{result.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(query, context, standalone, history)},
        ]
        for llm in self.llms:
            try:
                answer = llm.generate(messages, temperature=self.temperature, max_tokens=self.max_tokens).strip()
                confidence = sum(r.score for r in retrieval_results) / len(retrieval_results)
                # the model was told to answer with exactly the refusal message when evidence is missing
                refused = answer.startswith(self.refusal_message[:30]) or answer.startswith(self.REFUSAL_EN[:30])
                return GeneratedResponse(
                    answer=answer,
                    citations=citations,
                    domain_id=domain_id,
                    confidence=confidence,
                    refused=refused,
                    refusal_reason="Model found no supporting evidence" if refused else "",
                    llm_used=llm.label,
                )
            except Exception as e:
                logger.error("LLM %s failed: %s — trying next option", llm.label, e)

        answer = self._fallback_generate(standalone or query, retrieval_results)
        confidence = sum(r.score for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0.0

        return GeneratedResponse(
            answer=answer,
            citations=citations,
            domain_id=domain_id,
            confidence=confidence,
            refused=False,
            llm_used="extractive-fallback",
        )


@dataclass
class QueryAnalysis:
    """Step 1 of the pipeline: what the user is asking and how to search for it."""
    question: str                      # the user's message as typed
    standalone: str                    # rewritten to make sense without the conversation
    domain_id: str = ""                # classified service domain ("" = not sure / other)
    search_queries: list[str] = field(default_factory=list)  # paraphrases (Nepali official terms, English)
    method: str = "keywords"           # "llm" or "keywords" (no LLM available / LLM failed)

    @property
    def queries(self) -> list[str]:
        """Every distinct query to search with: the question itself first, then the paraphrases."""
        return list(dict.fromkeys(q for q in [self.question, self.standalone, *self.search_queries] if q))

    def to_dict(self) -> dict[str, Any]:
        return {"domain_id": self.domain_id, "standalone": self.standalone,
                "search_queries": self.search_queries, "method": self.method}


class QueryAnalyzer:
    """Classify the question into a service domain, make follow-ups standalone and paraphrase it.

    One short LLM call returning JSON. The knowledge base is mostly written in formal Nepali,
    so a paraphrase with official terms ("register land" -> "जग्गा रजिष्ट्रेशन") finds passages
    the user's own wording misses. Without an LLM the keyword classifier is used instead.
    """

    def __init__(self, llms: list[LLMClient], domains: list[dict[str, Any]], keyword_domain) -> None:
        self.llms = llms
        self.domains = domains
        self.keyword_domain = keyword_domain
        self.domain_ids = {d["id"] for d in domains}

    def _prompt(self, question: str, history: list[dict[str, str]]) -> str:
        domain_lines = "\n".join(
            f"- {d['id']}: {d.get('name_en', '')} / {d.get('name_np', '')}. {' '.join(d.get('description', '').split())}"
            for d in self.domains
        )
        convo = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
                          for m in history[-4:]) or "(none)"
        return f"""You prepare searches for a help desk about Nepal government services.
The knowledge base is written mostly in formal Nepali (official terms), with short English summaries.

Service domains:
{domain_lines}
- none: the question is about none of these

Conversation so far:
{convo}

New user message: {question}

Reply with ONLY this JSON object:
{{"domain": "<one domain id from the list, or none>",
 "standalone_question": "<the new message as a complete question that makes sense without the conversation, in the user's language>",
 "search_queries": ["<the question in formal Nepali using official government terms>", "<the question in plain English>"]}}

Rules: do not answer the question. Do not add facts (amounts, dates, names, numbers) the user did not write. Keep names and numbers exactly."""

    def analyze(self, question: str, history: list[dict[str, str]] | None = None) -> QueryAnalysis:
        history = history or []
        fallback = QueryAnalysis(question=question, standalone=question, domain_id=self.keyword_domain(question))
        if not self.llms:
            return fallback
        messages = [{"role": "user", "content": self._prompt(question, history)}]
        for llm in self.llms:
            try:
                raw = llm.generate(messages, temperature=0.0, max_tokens=400)
                data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
                domain = str(data.get("domain", "")).strip()
                standalone = str(data.get("standalone_question") or question).strip()[:400]
                queries = [str(q).strip()[:300] for q in data.get("search_queries", []) if str(q).strip()][:3]
                return QueryAnalysis(question=question, standalone=standalone,
                                     domain_id=domain if domain in self.domain_ids else "",
                                     search_queries=queries, method="llm")
            except Exception as e:  # unparsable JSON, network error, ... -> next LLM, then keywords
                logger.warning("Query analysis with %s failed (%s) — trying next option", llm.label, str(e)[:80])
        return fallback


class ChatSession:
    """The RAG pipeline: understand the question -> retrieve passages -> write a grounded answer."""

    def __init__(self, provider: str = "ollama", model: str = "llama3") -> None:
        settings = get_settings()
        self.generator = CitationGroundedGenerator(
            provider=provider, model=model, system_language=settings["generation"]["system_language"],
        )
        from src.retrieval.indexer import load_chunks, make_retriever
        from src.utils import REPO_ROOT, load_yaml
        self.chunks = load_chunks()
        self.retriever = make_retriever()
        if self.chunks:
            self.retriever.build_index(self.chunks)  # dense index is reused when unchanged
        domains = load_yaml(REPO_ROOT / "config" / "domains.yaml").get("domains", [])
        self.analyzer = QueryAnalyzer(self.generator.llms, domains, self._keyword_domain)
        self.history: list[dict[str, str]] = []  # terminal chat only; the web app keeps history per conversation

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    # Checked in order — specific signals first. "सामाजिक सुरक्षा" alone is ambiguous
    # (SSF fund vs. social-security allowance), so it is only a last-resort welfare hint.
    _DOMAIN_KEYWORDS: list[tuple[str, list[str]]] = [
        ("identity_civil_registration", ["नागरिकता", "nagarikta", "citizenship", "national id", "राष्ट्रिय परिचय",
                                         "परिचयपत्र", "जन्म दर्ता", "मृत्यु दर्ता", "विवाह दर्ता"]),
        ("land_property", ["जग्गा", "जमिन", "नामसारी", "भूमि", "लालपुर्जा", "land", "property", "jagga", "lalpurja"]),
        ("health_social_welfare", ["भत्ता", "बृद्धा", "वृद्ध", "जेष्ठ", "ज्येष्ठ", "एकल महिला", "अपाङ्ग", "allowance",
                                   "senior citizen", "स्वास्थ्य", "health", "medical", "insurance board"]),
        ("employment_labour", ["कोष", "ssf", "वैदेशिक", "रोजगार", "श्रम", "foreign employment", "labour", "dofe"]),
        ("health_social_welfare", ["सामाजिक सुरक्षा", "बीमा", "insurance"]),
    ]

    def _keyword_domain(self, query: str) -> str:
        """Keyword domain guess — returned only when the keywords point to exactly ONE domain.
        "भत्ताको परिचयपत्र" hits both welfare (भत्ता) and identity (परिचयपत्र): ambiguous -> ""."""
        q = query.lower()
        *specific, (last_resort_domain, last_resort_kws) = self._DOMAIN_KEYWORDS
        hits = {domain_id for domain_id, keywords in specific if any(kw in q for kw in keywords)}
        if not hits and any(kw in q for kw in last_resort_kws):
            hits = {last_resort_domain}
        return hits.pop() if len(hits) == 1 else ""

    def reset(self) -> None:
        self.history = []

    def analyze(self, query: str, history: list[dict[str, str]] | None = None, use_llm: bool = True) -> QueryAnalysis:
        """Step 1 — classify + rewrite + paraphrase (LLM), or the keyword classifier only."""
        if use_llm:
            return self.analyzer.analyze(query, history)
        return QueryAnalysis(question=query, standalone=query, domain_id=self._keyword_domain(query))

    def retrieve(self, query: str, domain: str | None = None,
                 analysis: QueryAnalysis | None = None) -> tuple[str, list]:
        """Step 2 — search every query of the analysis inside the domain AND across the whole
        knowledge base, then fuse the ranked lists (Reciprocal Rank Fusion).

        Searching the domain (metadata filter: BM25 mask + Chroma `where`) puts the right
        service's passages first; the whole-KB lists (half weight) keep a wrong or missing
        classification from hiding the answer. Returns (domain_id, top-k passages).
        Without `analysis` only the keyword classifier is used (fast, no LLM — the benchmark)."""
        top_k = get_settings()["retrieval"]["top_k"]
        if not self.chunks:
            return "", []
        analysis = analysis or self.analyze(query, use_llm=False)
        domain_id = domain or analysis.domain_id  # the service the user picked in the UI wins

        ranked_lists: list[tuple[float, list]] = []
        for q in analysis.queries:
            if domain_id:
                ranked_lists.append((1.0, self.retriever.search(q, top_k=top_k, domain=domain_id)))
            ranked_lists.append((0.5 if domain_id else 1.0, self.retriever.search(q, top_k=top_k)))

        fused: dict[str, float] = {}
        best: dict[str, Any] = {}
        for weight, results in ranked_lists:
            for rank, r in enumerate(results, 1):
                fused[r.chunk_id] = fused.get(r.chunk_id, 0.0) + weight / (60 + rank)
                best.setdefault(r.chunk_id, r)
        top = sorted(fused, key=fused.get, reverse=True)[:top_k]
        results = [best[cid] for cid in top]
        for cid, r in zip(top, results):
            r.score = round(fused[cid] * 60, 4)  # fused score, ~1.0 = ranked first everywhere
        return (domain_id or (results[0].domain_id if results else "")), results

    def ask(self, query: str, domain: str | None = None,
            history: list[dict[str, str]] | None = None) -> GeneratedResponse:
        """understand -> retrieve -> answer. `history`: earlier turns of this conversation."""
        terminal = history is None  # terminal chat keeps its own history; the web app passes it in
        history = self.history if terminal else history
        analysis = self.analyze(query, history)
        domain_id, retrieval_results = self.retrieve(query, domain, analysis)
        response = self.generator.generate(query, retrieval_results, domain_id,
                                           standalone=analysis.standalone, history=history)
        response.analysis = analysis.to_dict()

        if terminal:
            self.history += [{"role": "user", "content": query}, {"role": "assistant", "content": response.answer}]
        return response

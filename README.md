# Nepal Sewa AI · नेपाल सेवा AI

A **Retrieval-Augmented Generation (RAG)** assistant that answers questions about Nepal
government public services in **Nepali or English**. Every answer is built **only from official
government pages**, cites the passages it used and links to the official page. Any question can be
asked: every question searches the whole knowledge base, and the answer uses every relevant passage.
If the sources cover only part of a question it answers that part and says what is missing; it never
fills gaps from the LLM's own memory.

**Stack:** FastAPI · ChromaDB · BGE-M3 embeddings · BM25 · Gemini / Ollama. Managed entirely with **[uv](https://docs.astral.sh/uv/)**.

| Domain | Example question | Official sources |
|---|---|---|
| Identity & Civil Registration | राष्ट्रिय परिचयपत्र बनाउन के के कागजात चाहिन्छ? | donidcr.gov.np |
| Land & Property | How do I register land in Nepal? | dolma.gov.np, molcfaga.gov.np, dos.gov.np |
| Employment & Labour | सामाजिक सुरक्षा कोषमा पेन्सन पाउन कति वर्ष योगदान गर्नुपर्छ? | dofe.gov.np, ssf.gov.np |
| Health & Social Welfare | ७० वर्ष पुगेका जेष्ठ नागरिकले मासिक कति भत्ता पाउँछन्? → **रु. ४०००** | donidcr.gov.np, hib.gov.np |

---

## Quick start: clone and run (any machine or a fresh VM)

Requires only **[uv](https://docs.astral.sh/uv/getting-started/installation/)**. uv installs
Python 3.14 (pinned in `.python-version`) and all dependencies (locked in `uv.lock`) itself.

```bash
# 1. install uv (once per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh          # Linux / macOS
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# 2. clone and run
git clone <repo-url>
cd <repo-folder>
uv run fastapi run          # all interfaces, port 8000 → http://<vm-ip>:8000
# or
uv run fastapi dev          # development, auto-reload → http://127.0.0.1:8000
```

That's it: no install step and no chunking. On the first run:

1. **uv** installs Python 3.14 and the locked packages into `.venv` (about 1–2 min). On Linux,
   PyTorch comes from PyTorch's CPU-only index, so no CUDA packages are downloaded.
2. **The app loads the knowledge base**: the 530 chunks in `data/processed/all_chunks.json` (committed).
3. **The BGE-M3 embedding model is downloaded once** to the Hugging Face cache (about 2.3 GB).
4. **The vector store is built once from the chunks** into `data/index/chroma` (not committed:
   each machine builds its own). This takes about 8–10 min on a CPU-only machine. Later starts load
   it in about 30 s and never re-embed.

The page opens immediately. While the engine loads, a progress bar under the header shows the
stage: reading chunks, loading the model, and "embedding 160/530 (30%)" on the first run. The
server log shows the same progress. To build the store before starting the server instead, run
`uv run nepal-rag build`.

On a cloud VM, open port 8000 in its firewall / security group (or put a reverse proxy in front).

**Optional, for generated answers:** add a free Gemini key.

```bash
cp .env.example .env        # Windows: copy .env.example .env
# edit .env → GEMINI_API_KEY=your-key   (https://aistudio.google.com/apikey)
```

Without a key the app still works. It answers with the best-matching official passage,
and the answer's "कसरी खोजियो?" panel shows `extractive-fallback`.

**Rebuilding:** only needed when you change the knowledge base.

- Edited `data/chunks/new_chunks.json`, the embedding model or chunk settings? **Stop the server first** (Windows locks the index files), then run `uv run nepal-rag build`. It regenerates `all_chunks.json` and re-embeds (about 8–10 min on CPU for 530 chunks, with progress in the log). Running `build` again with nothing changed embeds nothing ("Vector index is already up to date").
- Until you rebuild, the server keeps using the existing index and logs a warning that it is older than the chunks; it never re-embeds on its own.
- Deleted `data/index/`? You don't need to do anything: the app builds the vector store once at startup (progress bar in the page), as on a fresh clone.
- Deleted `all_chunks.json`? The app regenerates it from `data/chunks/new_chunks.json` at startup.



| Command | What it does |
|---|---|
| `uv run fastapi dev` | Dev server with auto-reload on `127.0.0.1:8000` |
| `uv run fastapi run` | Production-style server on `0.0.0.0:8000` |
| `uv run nepal-rag fetch` | Crawl every source in `config/domains.yaml` into `data/raw/` |
| `uv run nepal-rag fetch --missing` | Fetch only sources that have no successful download yet |
| `uv run nepal-rag build` | Knowledge base → `all_chunks.json` → embed into ChromaDB (only if the chunks changed) |
| `uv run nepal-rag eval [--rewrite]` | Retrieval benchmark: hit@k and MRR (`--rewrite`: with the LLM query-understanding step) |
| `uv run nepal-rag ask "प्रश्न"` / `chat` | Ask in the terminal |
| `uv run pytest` | Run the tests |
| `uv add <pkg>` / `uv remove <pkg>` | Manage dependencies. Never use pip. |

`fastapi dev` and `fastapi run` find the app through `pyproject.toml`
(`[tool.fastapi] entrypoint = "src.main:app"`), so no file path is needed. Every package folder
under `src/` has an `__init__.py`. **Don't add one at the repo root:** with a path argument the
CLI would build the import name `nepal-rag-test.src.main`, which is invalid because of the hyphen.

---

## How it works

This is **RAG**: the LLM does not search and does not answer from its own memory. Our code
retrieves official passages, and the LLM reads them and writes a short, cited answer.

```
 OFFLINE (build once)                                       uv run nepal-rag fetch / build
 ┌────────────────┐     ┌──────────────────────────┐     ┌──────────────────────────────┐
 │ 1. CRAWL       │────▶│ 2. KNOWLEDGE BASE        │────▶│ 3. INDEX                     │
 │ 119 official   │     │ 530 checked chunks       │     │ BGE-M3 vectors → ChromaDB    │
 │ pages, PDFs,   │     │ data/chunks/             │     │ + BM25 keyword index         │
 │ Acts & Rules   │     │   new_chunks.json        │     │ metadata stored per vector   │
 │ (robots.txt,   │     │ one topic per chunk      │     │ data/index/chroma            │
 │  crawl-delay)  │     └──────────────────────────┘     └──────────────────────────────┘

 ONLINE (every question)                                    POST /api/chat
 question (+ last turns of the conversation)
   ─▶ ① UNDERSTAND  one short LLM call → JSON: domain (classified among the 4, or none),
                    standalone question (follow-ups resolved), paraphrases in formal Nepali + English
                    (no LLM → keyword classifier, question as typed)
   ─▶ ② RETRIEVE    each query searched inside the domain (metadata filter) AND in the whole KB
                    (hybrid: RRF 0.9 BGE-M3 + 0.1 BM25), all lists fused with RRF → top 8
   ─▶ ③ AUGMENT     sectioned prompt: conversation · question · understood as · [स्रोत 1..8] · task
   ─▶ ④ GENERATE    the LLM combines every relevant passage: direct answer, bullets, [स्रोत n]
                    after each fact, "थप जानकारी: <official URL>" — partial answers allowed,
                    refuses only when no passage is related
   ─▶ answer + sources + passages + "how it was searched", saved in the conversation (SQLite)
```

**The app does not browse the web while answering.** It answers from the crawl snapshot; each
source's fetch date is shown. To refresh, re-crawl (`fetch`), update the chunks, and `build`.

### 1. Crawling (`src/ingestion/fetch_sources.py`)
The sources are listed in `config/domains.yaml`: FAQs, citizen charters, service and procedure
pages, and citizen-facing notices chosen from each portal's menus. The fetcher is polite:
- **robots.txt:** obeys Disallow rules and `Crawl-delay` (donidcr, dolma, dofe and hib ask for 10 s).
- **TLS:** certificates are verified.
- **Retries:** a 503/429 gets one retry after a back-off (honours `Retry-After`).
- **Order:** sites are fetched round-robin so one slow site doesn't stall the others.
- **Records:** each download is versioned by timestamp, with a `.meta.json` record (URL, fetch time, sha256).
- **Acts and Rules:** 29 laws (Acts and their Rules) published by the Nepal Law Commission. Its PDF server
  (`giwmscdnone.gov.np`, shared by most government portals) refuses every scripted download from our network,
  so the crawler fetches the **Internet Archive's byte-for-byte copies** of the same official PDFs instead
  (`url: https://web.archive.org/web/<timestamp>id_/<official url>`, with the original address kept as
  `official_url` in `domains.yaml`). Most copies were captured in 2026; for each document the newest capture that
  is a real PDF (not a bot-check page) is used. Some older Rules PDFs use the legacy Preeti font and were converted
  to Unicode when the chunks were written.
- **Manual downloads:** `uv run nepal-rag import [folder]` registers PDFs you downloaded in a browser. Each file is
  matched to a configured source by checksum (HEAD request: size + ETag), so file names don't matter.

### 2. Knowledge base (`data/chunks/new_chunks.json`, loaded by `src/chunking/knowledge_base.py`)
The crawled pages are **not** chunked blindly. The source data has real problems:
- navigation menus repeated across pages
- FAQ questions hidden in accordion buttons
- tables flattened into one line
- PDFs whose Devanagari text extracts garbled (for example "३० दिनभित्र" comes out as "३० वदनवभत्र")
- notices whose content sits in attachments

So the knowledge base is **extracted and written from the crawled pages, then checked**: one topic per
chunk (at most 450 tokens), in the source's wording, with a 1–2 sentence English summary so English
questions match too. It was built by one extraction agent per domain that crawled each portal further
(citizen charters, FAQs, notices, acts, procedure PDFs) with the project's own polite fetcher and
wrote only what the cited page says; the numbers were spot-checked afterwards.

All domains live in one file, organised per domain, each with its sources and known gaps:

```json
{
  "total_chunks": 530,
  "domains": [
    {
      "domain_id": "identity_civil_registration",
      "authority": "Department of National ID and Civil Registration (DoNIDCR)",
      "sources": [{"name": "...", "url": "...", "type": "faq"}],
      "gaps": ["topics the official sources do not cover"],
      "chunks": [
        {
          "id": "nid-lost-duplicate-fee",
          "url": "https://donidcr.gov.np/pages/abboyed-asked-questions--rational-identity-10",
          "section": "राष्ट्रिय परिचयपत्र — हराएमा प्रतिलिपि र शुल्क",
          "type": "fees",
          "date_bs": "३ भदौ, २०८३",
          "text": "राष्ट्रिय परिचयपत्र हराएमा … प्रतिलिपि शुल्क: रु. ५०० राजस्व …"
        }
      ]
    }
  ]
}
```

`type` is one of faq, procedure, documents, fees, eligibility, contact, notice, about, services,
timeline, grievance or act; `date_bs` is optional. The older per-domain files `data/chunks/<domain>.yaml`
(the first 84-chunk version) are used only if `new_chunks.json` is absent.

The loader adds the rest of the metadata automatically:
- **from the crawl record of that URL:** fetch time, document type
- **from `domains.yaml`:** the authority and the source's display name
- **computed:** language, token count (measured with the embedding model's tokenizer), content hash, and the `search_text` that gets embedded ("source › section + text")

The loader also catches mistakes: missing fields, duplicate IDs, URLs that were never fetched,
and chunks too long for the embedder.

**Current knowledge base: 530 chunks.**

| Domain | Web pages & PDFs | Acts & Rules | Total |
|---|---|---|---|
| Health & Social Welfare | 77 | 108 (Health Insurance Act and Rules, Social Security Act and Rules, Public Health Service, Safe Motherhood, Senior Citizens, Disability Act and Rules, Children) | 185 |
| Employment & Labour | 72 | 73 (Foreign Employment, Labour Act and Rules, Right to Employment, Child Labour, Contribution-based Social Security Act and Rules) | 145 |
| Identity & Civil Registration | 44 | 63 (National ID and Vital Registration Act, Citizenship Act and Rules, Passport Act) | 107 |
| Land & Property | 24 | 69 (Land Revenue Act and Rules, Lands Act and Rules, Land Survey Act and Rules, Land Use, Land Acquisition) | 93 |

Law chunks restate one provision each in plain language and cite it ("मालपोत ऐन, २०३४ को दफा ७ अनुसार …"),
so the answer can say which section a right, deadline or penalty comes from. The first version had 84 chunks.

To edit: change or add entries in `data/chunks/new_chunks.json`, then stop the server and run
`uv run nepal-rag build`.
Rule: write only what the cited page says.

*Comparison mode:* set `chunking.knowledge_base: auto` in `config/settings.yaml` to chunk the
crawled pages automatically instead (DOM-based extractor `src/processing/html_extract.py` +
`src/chunking/structure_aware.py`).

### 3. Retrieval (`src/retrieval/indexer.py`)
- **Dense:** `BAAI/bge-m3` multilingual embeddings in a persistent ChromaDB collection. The full chunk metadata is stored with each vector and is used for domain filtering (`where={"domain_id": …}`).
- **Keyword:** BM25 over Devanagari and Latin tokens.
- **Fusion:** Reciprocal Rank Fusion with weights 0.9 dense and 0.1 BM25 (chosen by benchmark, see below).
- **Step ① — query understanding** (`QueryAnalyzer` in `src/generation/chat.py`). One short LLM call (temperature 0, JSON output) does three things:
  - **classifies** the question into one of the 4 domains (or none), using each domain's description from `domains.yaml`;
  - **rewrites a follow-up** ("and how much does it cost?") into a standalone question, using the conversation's last turns;
  - **paraphrases** it in formal Nepali with official terms and in plain English. The knowledge base is mostly formal Nepali, so "register land" also searches "जग्गा रजिष्ट्रेशन".

  It is told not to answer and not to add facts. If no LLM is available or the JSON can't be parsed, a keyword classifier is used instead ("जग्गा" → land, only when the keywords point to exactly one domain) with the question as typed.
- **Step ② — filtered + global retrieval.** Every query (question, standalone question, paraphrases) is searched twice: **inside the classified domain** (metadata filter: BM25 mask + Chroma `where`) and **across the whole knowledge base** at half weight. All ranked lists are fused with Reciprocal Rank Fusion. The domain's passages come first, but a wrong or missing classification can't hide the answer, and cross-domain passages (e.g. SSF for a foreign-employment question) still reach the LLM. The service picked in the UI overrides the classifier.
- **Every question can be asked.** Questions don't have to match an FAQ: the 3 example questions per domain in the UI are only shortcuts.
- **Top 8 passages** go to the LLM. Chunks are small, so a full answer often combines several (process, documents, fees, office).
- **Built once:** the vector store is embedded once (`nepal-rag build`) and then only loaded. A fingerprint of model, chunk texts and metadata is stored *after* the last chunk is added, so `build` skips embedding when nothing changed and an interrupted build is redone instead of reused half-empty. The server and `ask`/`chat`/`eval` never re-embed: if the chunks changed they log a warning and use the existing index.

### 4. Generation (`src/generation/chat.py`)
The user message is built in sections (`_build_user_prompt`), so the model can tell context from evidence:

```
## CONVERSATION SO FAR (only to understand the question — not a source of facts)   ← follow-ups only
## QUESTION                     the user's words
## THE QUESTION UNDERSTOOD AS   the standalone question from step ①, when different
## OFFICIAL SOURCE PASSAGES     [स्रोत 1] source › section, URL, date, text … [स्रोत 8]
## TASK                         answer from the passages only + the answer language
```

The system prompt (`_build_system_prompt`) makes the LLM a grounded reader:
- Use only facts in the passages: never add fees, deadlines, documents or numbers from the LLM's own knowledge, and never reuse facts from earlier answers in the conversation.
- Combine every relevant passage, and ignore passages that are off-topic.
- If the passages answer only part of the question, answer that part and say which part the official sources don't cover and which office to ask.
- Only if no passage is related, reply with one fixed refusal sentence (Nepali or English); the UI shows that answer on a yellow background.
- Answer in the question's language. The code detects it and tells the model explicitly, because the mostly-Nepali passages otherwise pull English answers into Nepali.
- Format: start with a direct 1–3 sentence answer, then bullets for documents, steps, fees, deadlines and contacts, but only those the sources mention.
- Put `[स्रोत n]` after each fact, and end with `थप जानकारी: <official URL>`.
- Give dates for "latest" questions, and never call an old notice recent.

**LLM chain:** `Ollama (local) → Gemini (gemini-3.5-flash-lite) → extractive fallback`.
- Ollama is skipped automatically if it isn't running or the model isn't pulled.
- Gemini retries 503/429 errors.
- With no LLM, the best-matching passage is returned verbatim.

---

## Evaluation

`uv run nepal-rag eval` runs **53 questions** (`data/eval/questions.yaml`) covering all four
domains, in Nepali and English, phrased the way citizens ask. A question is a hit at rank *k*
if one of the top-*k* passages contains the expected fact. Expected facts are listed in both
the knowledge-base wording and the source's own wording, so both chunking modes are scored
fairly. This measures **retrieval only**, so it is fast and repeatable.

**Current system** (530 chunks, BGE-M3, RRF 0.1/0.9, top 8, domain-filtered + whole-KB retrieval fused with RRF), 53 questions:

| Query understanding (step ①) | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| Keyword domain classifier, question as typed (`eval`) | 0.85 | 0.94 | **1.00** | 0.91 |
| **LLM: domain + standalone question + Nepali/English paraphrases (`eval --rewrite`)** | **0.89** | **0.98** | **1.00** | **0.93** |

Every expected fact is within the top 5 of the 8 passages given to the LLM. The paraphrases help
most with English questions over the Nepali passages. With the 217-chunk knowledge base and plain
keyword routing it was hit@1 0.92 / hit@5 0.98 on 48 questions; the 530-chunk version has 2.5× more
(and more similar, law-based) chunks to choose from and 5 harder questions on the Acts.

**Knowledge base** (first crawl, 41 questions, BGE-M3, RRF 0.1/0.9):

| Knowledge base | Chunks | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| Automatic chunking of the crawled pages | 267 | 0.61 | 0.83 | 0.90 | 0.73 |
| Hand-written chunks (first version) | 84 | 0.95 | 0.98 | 1.00 | 0.97 |

**Fusion** (84-chunk knowledge base, 41 questions):

| Fusion | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| Weighted score sum 0.4 BM25 / 0.6 dense (previous) | 0.80 | 0.93 | 0.98 | 0.87 |
| RRF 0.3 / 0.7 | 0.88 | 0.98 | 1.00 | 0.92 |
| **RRF 0.1 / 0.9 (current)** | **0.95** | **0.98** | **1.00** | **0.97** |
| Dense only | 0.95 | 0.98 | 1.00 | 0.97 |
| BM25 only | 0.73 | 0.85 | 0.93 | 0.81 |

Earlier experiments on the first 6-page crawl showed the embedding model mattered most there:
MiniLM (128-token limit, which truncated 20 of 61 chunks) reached MRR 0.79, multilingual-e5-base
0.69, and BGE-M3 0.83 on a 21-question set.

**Caveats:**
- The same person wrote the chunks and the questions, so treat the absolute numbers as optimistic.
- The *comparisons* are fair, since they use the same questions, crawl and model.
- The quality of the generated answers (step ③) is not covered by this benchmark.

---

## Web UI and API

FastAPI's `lifespan` handler serves `web/index.html` at `/` (`HTMLResponse`) and loads the RAG
engine once, in a background thread, so the page opens immediately and shows a progress bar until
the engine is ready (`/api/health` → `progress`). The page is a single plain HTML, CSS and
JavaScript file with no build step, and it calls the API with `fetch()`. The interface text is in
Nepali with English alongside.

**Users, conversations and chats** (like ChatGPT): a **user** has many **conversations**; a
conversation is a list of **chats** (question → answer). There is no login: the page creates a
random user ID (`u_…`), keeps it in the browser and shows it in the top bar ("बदल्नुहोस्" switches
to another ID). Every conversation has its own ID (`c_…`), shown above the chats and kept in the
address bar (`/#c_…`), so a reload or a shared link reopens it.

UI features (a plain help-desk layout: top bar, conversation list on the left, chats on the right):
- **Fixed page:** the header, sidebar and question box stay in place; only the chats and, when it is long, the conversation list scroll.
- **Conversation list:** grouped by date (today, yesterday, this week, earlier), each with its service tag (जग्गा, रोजगार/SSF, …) and number of chats; search, Rename and Delete per conversation, and "Delete all". Click the title above the chats to rename it; the open conversation can also be deleted or **downloaded** as a text file with its sources.
- **Stats** in the sidebar: passages and sources in the knowledge base, your conversations and chats.
- **Feedback:** "उपयोगी भयो? हो / होइन" under every answer, saved with the answer (click again to undo).
- **Related questions** from the same service appear under the latest answer as one-click buttons.
- **Scope:** the welcome page says clearly that only the 4 services are covered, with a one-line summary and example questions for each.
- **Follow-up questions** work: the last turns of the conversation are used to understand them (step ①).
- **Evidence:** each answer lists its sources with retrieval dates. "Show passages" lists the 8 retrieved passages, and clicking a `[n]` citation opens and highlights that passage. "Show how it was searched" shows the classified service, the standalone question, the paraphrases searched and which LLM answered.
- **Service filter:** search one service, or all services (classified automatically).
- **Answer layout:** 15.5 px text with a slightly larger first sentence, spaced bullets, citation markers as small boxed numbers, the official link as a footnote line; header shows service, number of sources and time taken. Copy answer and Try again on errors.
- **Keyboard:** Enter sends, Shift+Enter adds a new line, `/` focuses the question box, Esc closes the menu on phones.
- Responsive: on phones the conversation list slides in from the "Conversations" button.

Conversations are stored in SQLite (`src/storage.py`, file `data/conversations.db`, created on first use and not committed). Every read and change checks that the conversation belongs to the user ID.

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/chat` | `{"message", "user_id", "conversation_id": null \| "c_…", "domain": null \| "<domain_id>"}` → answer, domain, LLM used, latency, refused, sources, passages, `analysis` (step ①), `conversation_id`, `message_id`. Omitting `conversation_id` starts a new conversation. |
| GET | `/api/conversations?user_id=…` | The user's conversations (title, dates, message count, service of the last answer), newest first |
| GET | `/api/conversations/{id}?user_id=…` | One conversation with all messages (answers keep their sources and passages) |
| PATCH | `/api/conversations/{id}?user_id=…` | `{"title": "…"}` rename |
| DELETE | `/api/conversations/{id}?user_id=…` | Delete one conversation |
| DELETE | `/api/conversations?user_id=…` | Delete all of the user's conversations |
| PUT | `/api/messages/{id}/feedback?user_id=…` | `{"helpful": true \| false \| null}` rate an answer |
| GET | `/api/health` | Status (`loading` / `ready` / `error`), startup `progress` (stage, done, total), chunk count, LLM chain |
| GET | `/api/domains` | Domains and example questions |
| GET | `/api/sources` | Crawled source pages with fetch dates |
| GET | `/docs` | Interactive OpenAPI docs |

---

## Configuration

| File | Contents |
|---|---|
| `.env` (from `.env.example`, never committed) | `GEMINI_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`, `LLM_FALLBACK_*`, `OLLAMA_*` |
| `config/settings.yaml` | Knowledge-base mode, chunk size limits, embedding model, fusion weights, top-k, LLM chain |
| `config/domains.yaml` | Domains, source URLs to crawl and their display names, quick-nav example questions for the UI |
| `data/chunks/new_chunks.json` | The knowledge base (all domains, with sources and gaps) |

After changing the chunks, the embedding model or chunk settings, run `uv run nepal-rag build`.

---

## Project structure

```
├── pyproject.toml / uv.lock      # dependencies (uv), `nepal-rag` CLI, FastAPI entrypoint
├── .python-version               # Python 3.14 (uv installs it automatically)
├── .env.example                  # copy to .env and add your key
├── config/
│   ├── settings.yaml
│   └── domains.yaml              # domains + crawl sources + UI examples
├── web/index.html                # the UI (served at /)
├── src/
│   ├── main.py                   # FastAPI app: lifespan + chat and conversation routes
│   ├── storage.py                # conversations + messages (SQLite)
│   ├── cli.py                    # fetch / build / process / index / eval / ask / chat
│   ├── evaluate.py               # retrieval benchmark
│   ├── utils.py                  # settings, paths, .env, tokenizer-based token counting
│   ├── ingestion/fetch_sources.py        # polite crawler (robots.txt, crawl-delay, TLS, retries)
│   ├── chunking/knowledge_base.py        # loads data/chunks/new_chunks.json + fills metadata
│   ├── processing/html_extract.py        # DOM → typed blocks (auto mode)
│   ├── processing/pipeline.py            # build_chunks(): manual or auto knowledge base
│   ├── chunking/structure_aware.py       # structure-aware chunking (auto mode)
│   ├── retrieval/indexer.py              # BM25, ChromaDB dense, RRF hybrid, metadata filters
│   └── generation/chat.py                # pipeline: query understanding, fused retrieval, grounded prompt, LLM chain
├── data/
│   ├── raw/<domain>/             # crawl snapshot: pages/PDFs + .meta.json
│   ├── chunks/new_chunks.json    # knowledge base (530 chunks, all domains)
│   ├── chunks/<domain>.yaml      # first version (84 chunks), used only without new_chunks.json
│   ├── eval/questions.yaml       # benchmark (53 questions)
│   ├── processed/all_chunks.json # chunks the retriever loads (generated by build)
│   ├── index/chroma/             # vector store (generated by build)
│   └── conversations.db          # web chat history (created on first use, not committed)
└── tests/
```

**What is committed (about 70 MB):** code, config, the web page, tests, `data/raw` (crawl
snapshot), `data/chunks` (knowledge base), `data/eval`, and
`data/processed/all_chunks.json`, so a fresh clone runs without re-crawling or re-chunking.

**Not committed, built per machine:** `data/index/` (the vector store, built once from the chunks on
the first start) and `data/conversations.db` (chat history).

**Never committed:** `.env` (API keys), `.venv`, caches, logs and intermediates
(`data/processed/<domain>/`, `data/chunks_add/`). See `.gitignore`.

**First push:**
```bash
git init
git add .
git status            # check that .env and .venv are NOT listed
git commit -m "Nepal Sewa AI: RAG over official Nepal government service pages"
git branch -M main
git remote add origin <repo-url>
git push -u origin main
```

---

## Limitations and next steps

- **Laws say "as prescribed"; amounts live elsewhere.** The Acts and Rules give rights, procedures, deadlines, offices, appeals and penalties, but many current amounts (registration fees, land tax rates, allowance amounts, minimum wage, passport fees) are set by annual Finance Acts and government notices, which are not in the knowledge base. Law chunks note when a provision may have been amended since the version cited.
- **Some documents could not be read or found.** The NID and Vital Registration Rules 2077 PDF uses a broken font (unreadable even as page images), and no archived copy of the Foreign Employment Rules 2064 exists. Most notice and charter PDFs sit on the file server that refuses scripted downloads and are not archived, so only their titles and dates are included, including the **Land Administration Directive (भूमि प्रशासन निर्देशिका)**. Each domain's `gaps` list in `new_chunks.json` records what is missing.
- **Sources can disagree.** For example, the amended Social Security Act (2081) sets the senior-citizen allowance age at 68 (60 for Dalits and single women), while an older DoNIDCR rates page says 70. The prompt tells the model to give both with their dates.
- The knowledge base is a snapshot. Re-crawl periodically and update the chunks whose sources changed; the content hash in each `.meta.json` shows what changed.
- Follow-up questions use only the last 4 messages of the conversation, and only to understand the question; facts always come from the retrieved passages.

This is an assistive research tool, not legal advice. Always confirm with the cited office.

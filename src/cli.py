"""
Data-pipeline + terminal CLI.

    uv run nepal-rag build [--fetch]   # (re-download) -> knowledge base chunks -> embed into ChromaDB (only if changed)
    uv run nepal-rag fetch [--missing] # crawl sources in config/domains.yaml -> data/raw
    uv run nepal-rag import [folder]   # PDFs downloaded by hand in a browser -> data/raw (+ crawl records)
    uv run nepal-rag process           # data/chunks/*.yaml (or auto-chunked pages) -> all_chunks.json
    uv run nepal-rag eval [--rewrite]  # retrieval benchmark (--rewrite: with LLM query understanding)
    uv run nepal-rag index [--force]   # embed chunks -> data/index/chroma
    uv run nepal-rag ask "प्रश्न"       # one question in the terminal
    uv run nepal-rag chat              # interactive terminal chat

The web app is started separately with `uv run fastapi dev` (see src/main.py).
"""

from __future__ import annotations

import argparse

from src.utils import get_settings, setup_logging


def cmd_fetch(args: argparse.Namespace) -> None:
    from src.ingestion.fetch_sources import fetch_all
    fetch_all(domain=args.domain, delay_seconds=args.delay, only_missing=args.missing)


def cmd_import(args: argparse.Namespace) -> None:
    from pathlib import Path

    from src.ingestion.fetch_sources import import_downloads
    import_downloads(Path(args.inbox))


def cmd_process(args: argparse.Namespace) -> None:
    from src.processing.pipeline import build_chunks
    build_chunks(verbose=True)


def cmd_index(args: argparse.Namespace) -> None:
    from src.retrieval.indexer import build_index
    build_index(force_rebuild=args.force)


def cmd_build(args: argparse.Namespace) -> None:
    if args.fetch:
        from src.ingestion.fetch_sources import fetch_all
        fetch_all()
    from src.processing.pipeline import build_chunks
    from src.retrieval.indexer import build_index
    build_chunks(verbose=True)
    build_index()  # embeds only if the chunks changed since the last build
    print("\nBuild complete. Start the web app with:  uv run fastapi dev")


def cmd_eval(args: argparse.Namespace) -> None:
    from src.evaluate import run_eval
    run_eval(rewrite=args.rewrite)


def _session(args: argparse.Namespace):
    from src.generation.chat import ChatSession
    settings = get_settings()
    return ChatSession(
        provider=args.provider or settings["generation"]["provider"],
        model=args.model or settings["generation"]["model"],
    )


def _print_response(response) -> None:
    a = response.analysis
    if a:
        print(f"\n[{a.get('method')}] domain: {a.get('domain_id') or '-'} · understood as: {a.get('standalone')}")
        for q in a.get("search_queries", []):
            print(f"    search: {q}")
    print(f"\n{response.answer}\n\n(LLM: {response.llm_used or 'none'})")
    urls = list(dict.fromkeys((c["name"], c["url"]) for c in response.citations))
    if urls:
        print("\nSources:")
        for i, (name, url) in enumerate(urls, 1):
            print(f"  [{i}] {name} — {url}")


def cmd_ask(args: argparse.Namespace) -> None:
    _print_response(_session(args).ask(args.question))


def cmd_chat(args: argparse.Namespace) -> None:
    chat = _session(args)
    print("=" * 60 + "\nNepal Sewa AI — type 'quit' to exit\n" + "=" * 60)
    while True:
        try:
            query = input("\nप्रश्न / Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if query:
            _print_response(chat.ask(query))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nepal-rag", description="Nepal Sewa AI — data pipeline and terminal chat")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="process + chunk + index (add --fetch to re-download first)")
    p.add_argument("--fetch", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("fetch", help="download configured sources -> data/raw")
    p.add_argument("--domain", default=None)
    p.add_argument("--delay", type=float, default=1.0, help="minimum seconds between requests to a site")
    p.add_argument("--missing", action="store_true", help="only fetch sources without a successful download yet")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("import", help="register PDFs downloaded by hand (browser) as crawl records")
    p.add_argument("inbox", nargs="?", default="data/raw/_inbox", help="folder with the downloaded PDFs")
    p.set_defaults(func=cmd_import)

    sub.add_parser("process", help="knowledge base -> data/processed/all_chunks.json").set_defaults(func=cmd_process)

    p = sub.add_parser("index", help="embed chunks -> data/index/chroma (only if changed)")
    p.add_argument("--force", action="store_true", help="re-embed even if the index is up to date")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("eval", help="retrieval benchmark (hit@k, MRR) on data/eval/questions.yaml")
    p.add_argument("--rewrite", action="store_true", help="use the LLM query-understanding step (domain + paraphrases)")
    p.set_defaults(func=cmd_eval)

    for name, func, helptext in (("ask", cmd_ask, "ask one question"), ("chat", cmd_chat, "interactive chat")):
        p = sub.add_parser(name, help=helptext)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--provider", default=None)
        p.add_argument("--model", default=None)
        p.set_defaults(func=func)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(level=args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()

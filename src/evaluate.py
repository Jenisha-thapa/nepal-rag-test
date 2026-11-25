"""
Retrieval benchmark: hit@1 / hit@3 / hit@5 and MRR over data/eval/questions.yaml.

A question is a hit at rank r if any `expect` string appears in the r-th retrieved
passage. This measures retrieval only (no answer writing), so runs are repeatable —
use it to compare chunking strategies, embedding models and query understanding:

    uv run nepal-rag eval             # keyword domain classifier, the question as typed (no LLM)
    uv run nepal-rag eval --rewrite   # + step 1 of the pipeline: LLM domain classification and
                                      #   paraphrases (one LLM call per question)
"""

from __future__ import annotations

import sys

from src.utils import REPO_ROOT, load_yaml, normalize_nepali_text

QUESTIONS_PATH = REPO_ROOT / "data" / "eval" / "questions.yaml"


def run_eval(verbose: bool = True, rewrite: bool = False) -> dict[str, float]:
    from src.generation.chat import ChatSession

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles: ✓ / Devanagari

    questions = load_yaml(QUESTIONS_PATH)
    session = ChatSession()
    ranks: list[int | None] = []

    for item in questions:
        expect = [normalize_nepali_text(e) for e in item["expect"]]
        analysis = session.analyze(item["q"], use_llm=rewrite)
        _, results = session.retrieve(item["q"], analysis=analysis)
        rank = next(
            (i for i, r in enumerate(results, 1) if any(e in normalize_nepali_text(r.text) for e in expect)),
            None,
        )
        ranks.append(rank)
        if verbose:
            extra = f"  [{analysis.method}: {analysis.domain_id or '-'}]" if rewrite else ""
            print(f"  {'✓' if rank else '✗'} rank={rank or '-':<2}  {item['q']}{extra}")

    n = len(ranks)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in (1, 3, 5)}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / n
    print(f"\n  {n} questions · " + " · ".join(f"{k} {v:.2f}" for k, v in metrics.items())
          + f" · chunks indexed {session.num_chunks}" + (" · with LLM query rewriting" if rewrite else ""))
    return metrics

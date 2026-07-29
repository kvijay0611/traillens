"""
Eval harness for TrialLens.

Runs every question in eval/golden_dataset.json through the live agent graph
and scores four dimensions:

  1. keyword_recall   - fraction of expected_keywords present in the answer
                         (proxy for topical relevance / correctness)
  2. citation_check    - if requires_citation, did the answer include at
                         least one NCT ID?
  3. refusal_check     - if expects_refusal (out-of-domain or no-data
                         questions), did the model correctly avoid
                         fabricating an answer?
  4. latency_ms / cost - operational metrics per query

This is intentionally dependency-light (no external LLM-judge API) so it
runs anywhere with just an ANTHROPIC_API_KEY. Swap in RAGAS (faithfulness,
context precision/recall) for a more rigorous LLM-judged eval -- see
README "Extending the eval harness".

Usage:
    python -m eval.run_eval
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents import run_query
from backend.db import init_db

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

NCT_ID_PATTERN = re.compile(r"NCT\d{8}")
REFUSAL_MARKERS = ["no relevant data", "no matching data", "does not appear", "no trials found",
                    "not related to clinical trials", "outside the scope", "no data was found",
                    "couldn't find", "could not find", "not available in"]


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def has_citation(answer: str) -> bool:
    return bool(NCT_ID_PATTERN.search(answer))


def looks_like_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(marker in lower for marker in REFUSAL_MARKERS)


def run_eval():
    init_db()
    with open(GOLDEN_PATH) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        print(f"Running: {case['id']} - {case['query']}")
        state = run_query(case["query"])
        answer = state.get("answer", "")

        recall = keyword_recall(answer, case.get("expected_keywords", []))
        citation_ok = (not case.get("requires_citation")) or has_citation(answer)
        refusal_ok = (not case.get("expects_refusal")) or looks_like_refusal(answer)

        results.append({
            "id": case["id"],
            "query": case["query"],
            "type": case["type"],
            "keyword_recall": round(recall, 2),
            "citation_check_passed": citation_ok,
            "refusal_check_passed": refusal_ok,
            "latency_ms": state.get("latency_ms"),
            "answer_preview": answer[:150],
        })

    total = len(results)
    avg_recall = sum(r["keyword_recall"] for r in results) / total
    citation_pass_rate = sum(r["citation_check_passed"] for r in results) / total
    refusal_pass_rate = sum(r["refusal_check_passed"] for r in results) / total
    avg_latency = sum(r["latency_ms"] for r in results) / total

    report = {
        "summary": {
            "total_cases": total,
            "avg_keyword_recall": round(avg_recall, 3),
            "citation_check_pass_rate": round(citation_pass_rate, 3),
            "refusal_check_pass_rate": round(refusal_pass_rate, 3),
            "avg_latency_ms": round(avg_latency, 1),
        },
        "cases": results,
    }

    print("\n=== EVAL REPORT ===")
    print(json.dumps(report["summary"], indent=2))

    out_path = os.path.join(os.path.dirname(__file__), "eval_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    run_eval()

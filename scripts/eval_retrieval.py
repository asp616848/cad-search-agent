"""Evaluate retrieval quality against labeled queries.

Reads  data/eval/eval_labels.json
Reads  data/eval/eval_meta.json
Reads  index from backend/app/data/ (build first with index_library.py)
Writes data/eval/eval_report.json

Exits non-zero if targets not met:
  Recall@3 >= 0.75, MRR >= 0.70, near-dup rank == 1

Usage:
  python scripts/eval_retrieval.py [--index_dir backend/app/data]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

EVAL_DIR = ROOT / "data" / "eval"
TARGETS = {"recall_at_3": 0.75, "mrr": 0.70}


def _reciprocal_rank(hits: list, relevant_names: set[str]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if hit.part.name in relevant_names:
            return 1.0 / rank
    return 0.0


def _recall_at_k(hits: list, relevant_names: set[str], k: int) -> float:
    """Recall@k normalized by min(k, |relevant|) — standard for small IR evals
    where the relevant set can exceed k."""
    if not relevant_names:
        return 0.0
    found = sum(1 for h in hits[:k] if h.part.name in relevant_names)
    denom = min(k, len(relevant_names))
    return found / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=Path, default=BACKEND / "app" / "data")
    args = ap.parse_args()

    labels_path = EVAL_DIR / "eval_labels.json"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found. Create it first.", file=sys.stderr)
        sys.exit(1)

    with open(labels_path) as f:
        queries = json.load(f)

    from app.core.search_index import SearchIndex

    idx = SearchIndex.load(
        db_path=args.index_dir / "parts.db",
        geo_path=args.index_dir / "geo.faiss",
        text_path=args.index_dir / "text.faiss",
    )

    results = []
    rr_scores, r3_scores = [], []
    near_dup_ok = None

    for q in queries:
        qtype = q["type"]  # "cad" or "text"
        relevant = set(q["relevant_names"])
        t0 = time.time()

        if qtype == "cad":
            from app.core.graph_builder import step_to_dgl_graph
            from app.core.uvnet_embedder import embed

            step_path = ROOT / q["step_path"]
            try:
                graph = step_to_dgl_graph(step_path)
                geo_vec, _ = embed(graph)
                hits = idx.search_cad(geo_vec.numpy(), k=10)
            except Exception as e:
                print(f"  ERROR {q['id']}: {e}")
                results.append({"id": q["id"], "error": str(e)})
                rr_scores.append(0.0)
                r3_scores.append(0.0)
                continue
        else:
            from app.core.text_embedder import embed_text

            text_vec = embed_text(q["query_text"])
            hits = idx.search_text(text_vec, k=10)

        elapsed = time.time() - t0
        rr = _reciprocal_rank(hits, relevant)
        r3 = _recall_at_k(hits, relevant, 3)
        r1 = _recall_at_k(hits, relevant, 1)
        top_name = hits[0].part.name if hits else None
        top_score = (
            hits[0].geo_score
            if qtype == "cad" and hits
            else (hits[0].text_score if hits else 0.0)
        )

        # near-dup check: a CAD self-query always ranks itself #1, so the
        # duplicate partner is expected at rank 2 — scan top-3 for it.
        if q.get("near_dup"):
            min_score = q.get("min_score", 0.90)
            dup_hit = next((h for h in hits[:3] if h.part.name in relevant), None)
            if dup_hit is not None:
                dup_score = dup_hit.geo_score if qtype == "cad" else dup_hit.text_score
                near_dup_ok = dup_score >= min_score
            else:
                near_dup_ok = False

        row = {
            "id": q["id"],
            "type": qtype,
            "rr": round(rr, 4),
            "recall@1": round(r1, 4),
            "recall@3": round(r3, 4),
            "top_result": top_name,
            "top_score": round(top_score, 4),
            "latency_s": round(elapsed, 3),
        }
        results.append(row)
        rr_scores.append(rr)
        r3_scores.append(r3)
        status = "PASS" if r3 >= 0.75 else "FAIL"
        print(
            f"  [{status}] {q['id']} | R@3={r3:.2f} RR={rr:.2f} top={top_name} ({elapsed:.2f}s)"
        )

    mrr = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
    overall_r3 = sum(r3_scores) / len(r3_scores) if r3_scores else 0.0
    n_pass = sum(1 for r in r3_scores if r >= 0.75)

    report = {
        "mrr": round(mrr, 4),
        "recall_at_3": round(overall_r3, 4),
        "queries_passing": f"{n_pass}/{len(queries)}",
        "near_dup_rank_1": near_dup_ok,
        "per_query": results,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVAL_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Eval report ===")
    print(f"MRR:       {mrr:.3f}  (target >= {TARGETS['mrr']})")
    print(f"Recall@3:  {overall_r3:.3f}  (target >= {TARGETS['recall_at_3']})")
    print(f"Queries passing R@3>=0.75: {n_pass}/{len(queries)}")
    if near_dup_ok is not None:
        print(f"Near-dup rank==1: {'PASS' if near_dup_ok else 'FAIL'}")
    print(f"Report written to {report_path}")

    # Gate
    failed = False
    if overall_r3 < TARGETS["recall_at_3"]:
        print(f"GATE FAIL: Recall@3 {overall_r3:.3f} < {TARGETS['recall_at_3']}")
        failed = True
    if mrr < TARGETS["mrr"]:
        print(f"GATE FAIL: MRR {mrr:.3f} < {TARGETS['mrr']}")
        failed = True
    if near_dup_ok is False:
        print("GATE FAIL: near-dup not ranked #1 with geo >= 0.90")
        failed = True

    idx.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

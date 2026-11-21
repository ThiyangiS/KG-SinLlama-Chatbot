"""Compare the active retrieval rankings against the original 32 notebook labels.

Run from the Sigiriya-Hybrid-GraphRAG directory with
python -m scripts.evaluate_retrieval
"""

import argparse
import ast
import csv
import hashlib
import json
import os
from collections import Counter
from statistics import mean

from src.config import E5_MODEL, HEADING_INDEX, PROJECT_DIR, RRF_K, TEXT_INDEX, TOP_K, VECTOR_LIMIT
from src.embeddings import E5Embeddings
from src.graph import connect
from src.retrieval import SigiriyaRetriever


DATASET = PROJECT_DIR / "notebooks" / "Sigiriya_Local_GraphRAG_32_Question_Evaluation.ipynb"
OUTPUT = PROJECT_DIR / "results" / "retrieval"
APPROACHES = ("original", "multi_query", "hybrid")
METRICS = ("recall_at_3", "recall_at_5", "reciprocal_rank", "complete_at_3", "complete_at_5")
UNANNOTATED = {
    "answer_correctness": "not_evaluated",
    "graph_path_relevance": "not_evaluated",
    "selected_evidence_coverage": "not_evaluated",
}


def load_cases(path=DATASET):
    """Read the notebook's literal TESTS assignment without executing any cells."""
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    datasets = []
    for cell in notebook["cells"]:
        code = "".join(cell["source"])
        if cell["cell_type"] != "code" or not any(
            line.startswith("TESTS =") for line in code.splitlines()
        ):
            continue
        for node in ast.parse(code).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets
            ):
                datasets.append(ast.literal_eval(node.value))
    if len(datasets) != 1:
        raise ValueError("Expected one literal TESTS assignment in the evaluation notebook.")
    cases, = datasets
    if Counter(case["type"] for case in cases) != {"single": 22, "multi": 7, "unsupported": 3}:
        raise ValueError("Expected 22 single-topic, 7 multi-topic and 3 unsupported cases.")
    if [case["id"] for case in cases] != [f"Q{i:02d}" for i in range(1, 33)]:
        raise ValueError("Expected question IDs Q01 through Q32 in order.")
    for case in cases:
        expected = case["expected_ids"]
        count = len(set(expected))
        case_type = case["type"]
        valid = {"single": count == 1, "multi": count >= 2, "unsupported": count == 0}
        if not case["question"].strip() or not valid[case_type] or count != len(expected):
            raise ValueError(f"Invalid question or relevance labels for {case['id']}.")
    return cases


def score_ranking(ranking, expected):
    """Score a ranked chunk list. Empty gold sets are unscored."""
    required = set(expected)
    if not required:
        return dict.fromkeys(METRICS)
    hits_at_3 = required.intersection(ranking[:3])
    hits_at_5 = required.intersection(ranking[:5])
    first_hit = next((rank for rank, chunk in enumerate(ranking, 1) if chunk in required), None)
    return {
        "recall_at_3": len(hits_at_3) / len(required),
        "recall_at_5": len(hits_at_5) / len(required),
        "reciprocal_rank": 1 / first_hit if first_hit else 0.0,
        "complete_at_3": hits_at_3 == required,
        "complete_at_5": hits_at_5 == required,
    }


def evaluate(retriever, cases):
    """Retrieve once per question and score all three rankings after retrieval."""
    labeled_chunks = {chunk for case in cases for chunk in case["expected_ids"]}
    missing = sorted(labeled_chunks - retriever.sources.keys())
    if missing:
        raise ValueError(f"The connected database lacks labeled chunks: {', '.join(missing)}")

    rows = []
    for case in cases:
        result = retriever.retrieve(case["question"])
        for approach in APPROACHES:
            ranking = getattr(result, approach)
            scores = score_ranking(ranking, case["expected_ids"])
            rows.append({
                "question_id": case["id"],
                "type": case["type"],
                "question": case["question"],
                "answerability_label": case["answerability"],
                "approach": approach,
                "expected_ids": case["expected_ids"],
                "ranking": ranking,
                "retrieval_scored": bool(case["expected_ids"]),
                **scores,
                **UNANNOTATED,
            })
    return rows


def summarize(rows):
    """Macro-average over labeled questions for each approach and topic group."""
    summaries = []
    for approach in APPROACHES:
        for group in ("all_labeled", "single", "multi"):
            scored = [
                row for row in rows
                if row["approach"] == approach and row["retrieval_scored"]
                and (group == "all_labeled" or row["type"] == group)
            ]
            names = ("recall_at_3", "recall_at_5", "mrr", "complete_at_3", "complete_at_5")
            summaries.append({
                "approach": approach,
                "group": group,
                "questions_scored": len(scored),
                **{
                    name: mean(row[metric] for row in scored) if scored else None
                    for name, metric in zip(names, METRICS)
                },
                **UNANNOTATED,
            })
    return summaries


def _sha256_json(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def export(rows, summaries, cases, sources, output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (
        ("sigiriya_32_current_per_question.csv", rows),
        ("sigiriya_32_current_summary.csv", summaries),
    ):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, records[0].keys())
            writer.writeheader()
            writer.writerows({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                for key, value in row.items()
            } for row in records)

    metadata = {
        "dataset": DATASET.relative_to(PROJECT_DIR).as_posix(),
        "dataset_sha256": _sha256_json(cases),
        "question_counts": dict(Counter(case["type"] for case in cases)),
        "gold_provenance": "Historical source-section labels from the notebook's TESTS assignment.",
        "embedding_model": E5_MODEL,
        "vector_indexes": [TEXT_INDEX, HEADING_INDEX],
        "vector_limit": VECTOR_LIMIT,
        "rrf_k": RRF_K,
        "protected_hybrid_window": TOP_K,
        "source_chunks_sha256": _sha256_json({
            chunk_id: [row["heading"], row["evidence"]] for chunk_id, row in sources.items()
        }),
        "production_file_sha256": {
            name: hashlib.sha256((PROJECT_DIR / "src" / name).read_bytes()).hexdigest()
            for name in ("config.py", "embeddings.py", "graph.py", "retrieval.py")
        },
        "unscored_policy": "Retrieve for no-gold questions and exclude them from all metric denominators.",
        **UNANNOTATED,
    }
    (output / "sigiriya_32_current_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Check dataset without connecting.")
    args = parser.parse_args()
    cases = load_cases()
    if args.validate_only:
        print(f"Dataset verified: {len(cases)} questions (29 labeled, 3 without relevance labels).")
        return
    if not os.getenv("NEO4J_PASSWORD"):
        parser.error("Set NEO4J_PASSWORD for your existing local Neo4j database.")
    with connect() as session:
        retriever = SigiriyaRetriever(session, E5Embeddings())
        rows = evaluate(retriever, cases)
        export(rows, summarize(rows), cases, retriever.sources)
    print(f"Wrote {len(rows)} ranking rows and summary tables to {OUTPUT}")


if __name__ == "__main__":
    main()

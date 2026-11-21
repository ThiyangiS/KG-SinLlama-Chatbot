"""Ablate the saved live rankings and replay graph diagnostics from the manifest."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scripts.evaluate_retrieval import (
    APPROACHES, METRICS, OUTPUT, _sha256_json, load_cases, score_ranking,
)
from src.config import PROJECT_DIR, RRF_K, TOP_K
from src.retrieval import (
    DESTINATION_ANCHOR, decompose, graph_paths, overlap, rank_chunks, resolve_anchors,
)


BASELINE = OUTPUT / "sigiriya_32_current"
REPORTS = OUTPUT / "diagnostics"
CANDIDATE_WINDOW = 8
GRAPH_SUPPORT_MIN = 0.25
GRAPH_BONUS = 0.006


def _quote(source, needle):
    """Rebuild the source excerpt stored by the Stage 2 manifest loader."""
    start = source.index(needle)
    separators = "\n。.!?"
    left = max(source.rfind(sep, 0, start) for sep in separators) + 1
    ends = [source.find(sep, start + len(needle)) for sep in separators]
    right = min((end + 1 for end in ends if end >= 0), default=len(source))
    return source[left:right].strip()


class ManifestGraph:
    """Read-only adapter for the two graph operations used during replay."""

    def __init__(self, manifest, chunks):
        self.entities = {entity["id"]: entity for entity in manifest["entities"]}
        self.chunks = chunks
        self.links = defaultdict(set)
        self.outgoing = defaultdict(list)
        for fact in manifest["facts"]:
            subject, obj, chunk = (fact[key] for key in ("subject", "object", "chunk_id"))
            relation, needle = fact["relation"], fact["needle"]
            self.links[chunk].update((subject, obj))
            identity = "|".join((subject, relation, obj, chunk, needle))
            fact_id = "sg:fact:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            self.outgoing[subject].append((obj, {
                "relation": relation, "fact_id": fact_id, "chunk_id": chunk,
                "quote": _quote(chunks[chunk][1], needle),
            }))

    def run(self, query, **params):
        if "LINKS_TO_ENTITY" in query:
            rows = []
            for pos, chunk in enumerate(params["ids"]):
                heading, evidence = self.chunks[chunk]
                rows.extend({
                    **self.entities[entity], "pos": pos, "heading": heading, "evidence": evidence,
                } for entity in sorted(self.links[chunk]))
            return rows[:100]
        if "relationships(p)" in query:
            rows = []
            for seed in params["seeds"]:
                for node, edge in self.outgoing[seed]:
                    rows.append(self._path([seed, node], [edge]))
                    rows.extend(
                        self._path([seed, node, end], [edge, second])
                        for end, second in self.outgoing[node]
                        if second["fact_id"] != edge["fact_id"]
                    )
            return rows[:300]
        raise ValueError("Unexpected query in manifest replay.")

    def _path(self, ids, edges):
        return {
            "seed": ids[0], "entity_ids": ids, "edges": edges,
            "names": [self.entities[entity]["name"] for entity in ids],
        }


def supported_graph_candidates(question, anchors, paths, chunks):
    """Rank cited edges with a topical quote and a non-destination anchor."""
    specific = set(anchors) - {DESTINATION_ANCHOR}
    scores = {}
    for path in paths:
        if not specific.intersection(path["entity_ids"]):
            continue
        for edge in path["edges"]:
            cid, quote = edge.get("chunk_id"), edge.get("quote")
            if not quote or not edge.get("fact_id") or cid not in chunks:
                continue
            if quote not in chunks[cid][1]:
                continue
            support = overlap(question, quote)
            if support >= GRAPH_SUPPORT_MIN:
                scores[cid] = max(support, scores.get(cid, 0))
    return scores


def promote_one(original, multi, no_graph, graph_support):
    """Keep the first two slots; allow one corroborated candidate into slot three."""
    if len(original) < TOP_K:
        return no_graph
    incumbent = original[TOP_K - 1]
    scores = defaultdict(float)
    for ranking, weight in ((original, 1.0), (multi, 0.7)):
        for rank, cid in enumerate(ranking, 1):
            scores[cid] += weight / (RRF_K + rank)
    for cid, support in graph_support.items():
        scores[cid] += GRAPH_BONUS * support
    # Multi-query needs independent vector support; graph candidates must
    # still occur in at least one ranker's shortlist.
    shortlist = set(original[:CANDIDATE_WINDOW])
    candidates = (set(multi[:TOP_K]) & shortlist) | (
        set(graph_support) & (shortlist | set(multi[:CANDIDATE_WINDOW]))
    )
    candidates.difference_update(original[:TOP_K - 1])
    candidates.add(incumbent)
    winner = min(candidates, key=lambda cid: (-scores[cid], cid != incumbent, cid))
    head = [*original[:TOP_K - 1], winner]
    return head + [cid for cid in dict.fromkeys([*no_graph, *original, *multi]) if cid not in head]


def _read_inputs():
    cases = load_cases()
    meta = json.loads(BASELINE.with_name(BASELINE.name + "_run.json").read_text(encoding="utf-8"))
    if meta["dataset_sha256"] != _sha256_json(cases):
        raise ValueError("The historical questions or labels differ from the saved evaluation.")
    for name, digest in meta["production_file_sha256"].items():
        current = hashlib.sha256((PROJECT_DIR / "src" / name).read_bytes()).hexdigest()
        if current != digest:
            raise ValueError(f"Production {name} differs from the saved run; replay would mix versions.")
    source_path, = (PROJECT_DIR / "data" / "raw").glob("SIGIRIYA*.txt")
    sections = re.findall(
        r"(?ms)^##\s+(\d+)\.\s+([^\n]+)\n(.*?)(?=^##\s+\d+\.|\Z)",
        source_path.read_text(encoding="utf-8-sig"),
    )
    chunks = {
        f"sigiriya_{int(num):03d}": [f"{int(num)}. {title.strip()}", body.strip()]
        for num, title, body in sections
    }
    if _sha256_json(chunks) != meta["source_chunks_sha256"]:
        raise ValueError("The checked-in source text differs from the live evaluation corpus.")
    with BASELINE.with_name(BASELINE.name + "_per_question.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        saved = {}
        for row in csv.DictReader(handle):
            key = row["question_id"], row["approach"]
            if key in saved:
                raise ValueError(f"Duplicate saved ranking: {key}.")
            saved[key] = row
    expected_keys = {(case["id"], method) for case in cases for method in APPROACHES}
    if set(saved) != expected_keys:
        raise ValueError("Saved ranking rows do not cover all 32 questions and three approaches.")
    for case in cases:
        for method in APPROACHES:
            row = saved[case["id"], method]
            if row["question"] != case["question"] or json.loads(row["expected_ids"]) != case["expected_ids"]:
                raise ValueError(f"Saved question or labels differ for {case['id']}.")
    manifest_path = PROJECT_DIR / "original_group_work" / "Sigiriya_KG_Expansion_Manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    return cases, saved, chunks, ManifestGraph(manifest, chunks)


def evaluate_saved(cases, saved, chunks, session):
    entities = list(session.entities.values())
    diagnostics, rows = [], []
    for case in cases:
        qid, question, gold = case["id"], case["question"], case["expected_ids"]
        rankings = {method: json.loads(saved[qid, method]["ranking"]) for method in APPROACHES}
        original, multi, current = (rankings[method] for method in APPROACHES)
        anchors = resolve_anchors(session, question, entities, original)
        paths = graph_paths(session, question, entities, anchors)
        _, replay = rank_chunks(original, multi, paths)
        _, no_graph = rank_chunks(original, multi, [])
        support = supported_graph_candidates(question, anchors, paths, chunks)
        rankings.update(
            hybrid_without_graph=no_graph,
            graph_only=sorted(support, key=lambda cid: (-support[cid], cid)),
            promote_multi=promote_one(original, multi, no_graph, {}),
            promote_graph=promote_one(original, multi, no_graph, support),
        )
        diagnostics.append({
            "question_id": qid, "question": question, "type": case["type"],
            "gold_sections": {cid: chunks[cid][0] for cid in gold},
            "subquestions": decompose(question),
            "replayed_anchors": anchors,
            "replayed_paths": paths,
            "heuristic_graph_support": support,
            "hybrid_replay_matches_saved": replay == current,
            "replayed_hybrid_ranking": replay,
            "rankings": rankings,
        })
        baseline_scores = score_ranking(original, gold)
        for method, ranking in rankings.items():
            scores = score_ranking(ranking, gold)
            rows.append({
                "question_id": qid, "type": case["type"], "method": method,
                "gold_ranks": ";".join(
                    f"{cid}:{ranking.index(cid) + 1 if cid in ranking else '-'}" for cid in gold
                ),
                **scores,
                **{f"delta_{metric}": scores[metric] - baseline_scores[metric] if gold else None
                   for metric in METRICS},
            })
    return diagnostics, rows


def summaries(rows):
    summary = []
    for method in dict.fromkeys(row["method"] for row in rows):
        for group in ("all", "single", "multi"):
            scored = [row for row in rows if row["method"] == method and row["recall_at_3"] is not None
                      and (group == "all" or row["type"] == group)]
            metrics = {metric: mean(row[metric] for row in scored) for metric in METRICS}
            metrics["mrr"] = metrics.pop("reciprocal_rank")
            summary.append({"method": method, "group": group, "questions": len(scored), **metrics})
    return summary


def paired_deltas(rows):
    """Expose question-by-question changes for the two controlled graph comparisons."""
    indexed = {(row["question_id"], row["method"]): row for row in rows}
    comparisons = []
    for treatment, control in (
        ("hybrid", "hybrid_without_graph"),
        ("promote_graph", "promote_multi"),
        ("promote_graph", "hybrid"),
    ):
        for row in rows:
            if row["method"] != treatment or row["recall_at_3"] is None:
                continue
            baseline = indexed[row["question_id"], control]
            comparisons.append({
                "question_id": row["question_id"], "type": row["type"],
                "treatment": treatment, "control": control,
                "control_gold_ranks": baseline["gold_ranks"],
                "treatment_gold_ranks": row["gold_ranks"],
                **{f"delta_{metric}": row[metric] - baseline[metric] for metric in METRICS},
            })
    return comparisons


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORTS)
    args = parser.parse_args()
    diagnostics, rows = evaluate_saved(*_read_inputs())
    if not all(row["hybrid_replay_matches_saved"] for row in diagnostics):
        parser.error("Manifest replay disagrees with the saved live hybrid ranks; resolve the graph snapshot before comparing.")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "ablation_per_question.csv", rows)
    _write_csv(args.output / "ablation_summary.csv", summaries(rows))
    _write_csv(args.output / "paired_deltas.csv", paired_deltas(rows))
    (args.output / "graph_replay.json").write_text(
        json.dumps({
            "provenance": "Saved live rankings; graph paths reconstructed from the checked-in manifest.",
            "graph_path_relevance": "heuristic_only_no_independent_annotations",
            "experiment": {"shortlist": CANDIDATE_WINDOW, "min_quote_overlap": GRAPH_SUPPORT_MIN,
                           "graph_bonus": GRAPH_BONUS, "protected_slots": TOP_K - 1},
            "questions": diagnostics,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    matched = sum(row["hybrid_replay_matches_saved"] for row in diagnostics)
    print(f"Hybrid manifest replay matched {matched}/{len(diagnostics)} saved rankings.")
    print(f"Wrote diagnostics and ablation tables to {args.output}")


if __name__ == "__main__":
    main()

import csv
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import Mock

from scripts.evaluate_retrieval import evaluate, export, load_cases, score_ranking, summarize


class RetrievalEvaluationTests(unittest.TestCase):
    def test_multi_topic_counts_every_required_chunk(self):
        scores = score_ranking(["a", "a", "noise", "b", "c"], ["a", "b", "c"])
        self.assertAlmostEqual(scores["recall_at_3"], 1 / 3)
        self.assertEqual(scores["recall_at_5"], 1.0)
        self.assertEqual(scores["reciprocal_rank"], 1.0)
        self.assertFalse(scores["complete_at_3"])
        self.assertTrue(scores["complete_at_5"])

    def test_mrr_uses_full_ranking_and_handles_misses(self):
        ranking = ["noise"] * 5 + ["gold"]
        scores = score_ranking(ranking, ["gold"])
        self.assertEqual(scores["recall_at_5"], 0.0)
        self.assertAlmostEqual(scores["reciprocal_rank"], 1 / 6)
        self.assertEqual(score_ranking([], ["gold"])["reciprocal_rank"], 0.0)

    def test_empty_gold_is_unscored(self):
        self.assertTrue(all(value is None for value in score_ranking(["a"], []).values()))

    def test_dataset_load_and_full_export_with_fake_retriever(self):
        cases = load_cases()
        self.assertEqual(len(cases), 32)
        sources = {
            chunk: {"heading": chunk, "evidence": "fixture"}
            for case in cases for chunk in case["expected_ids"]
        }
        retriever = Mock(sources=sources)
        retriever.retrieve.return_value = SimpleNamespace(
            original=["sigiriya_001"], multi_query=[], hybrid=["sigiriya_002", "sigiriya_001"],
        )
        rows = evaluate(retriever, cases)
        summaries = summarize(rows)
        self.assertEqual(len(rows), 96)
        self.assertEqual(
            [call.args for call in retriever.retrieve.call_args_list],
            [(case["question"],) for case in cases],
        )
        self.assertEqual(retriever.retrieve.call_count, 32)
        self.assertEqual(summaries[0]["questions_scored"], 29)
        self.assertAlmostEqual(summaries[0]["mrr"], 1 / 29)
        self.assertEqual(summaries[2]["questions_scored"], 7)
        self.assertEqual(summaries[6]["approach"], "hybrid")
        self.assertAlmostEqual(summaries[6]["mrr"], 1.5 / 29)
        self.assertEqual(rows[-1]["graph_path_relevance"], "not_evaluated")
        output = Path(__file__).resolve().parent / f"evaluation-test-{uuid4().hex}"
        output.mkdir()
        self.addCleanup(output.rmdir)
        for name in ("per_question.csv", "summary.csv", "run.json"):
            self.addCleanup((output / f"sigiriya_32_current_{name}").unlink, missing_ok=True)
        export(rows, summaries, cases, sources, output)
        with (output / "sigiriya_32_current_per_question.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            exported = list(csv.DictReader(handle))
        self.assertEqual(exported[0]["question"], cases[0]["question"])
        self.assertEqual(json.loads(exported[0]["ranking"]), ["sigiriya_001"])
        self.assertEqual(exported[-1]["recall_at_3"], "")
        self.assertTrue((output / "sigiriya_32_current_summary.csv").is_file())
        self.assertTrue((output / "sigiriya_32_current_run.json").is_file())

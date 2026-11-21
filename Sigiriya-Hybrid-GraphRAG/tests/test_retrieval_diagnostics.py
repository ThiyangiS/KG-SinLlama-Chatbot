import unittest

from scripts.diagnose_retrieval import paired_deltas, promote_one, supported_graph_candidates
from scripts.evaluate_retrieval import score_ranking
from src.retrieval import rank_chunks


class ExperimentalRankerTests(unittest.TestCase):
    def test_only_grounded_topical_paths_qualify(self):
        quote = "fountains operate through water pressure"
        chunks = {"supported": ["Fountains", quote], "noise": ["Safari", "elephant tours"]}
        edges = [
            {"chunk_id": "supported", "fact_id": "f1", "quote": quote},
            {"chunk_id": "supported", "fact_id": "forged", "quote": "fountains water changed"},
            {"chunk_id": "noise", "fact_id": "f2", "quote": "elephant tours"},
        ]
        path = {"entity_ids": ["sg:dest", "fountain"], "edges": edges}
        self.assertEqual(
            supported_graph_candidates("fountains water pressure", ["fountain"], [path], chunks),
            {"supported": 1.0},
        )
        self.assertEqual(supported_graph_candidates("fountains", ["other"], [path], chunks), {})
        self.assertEqual(supported_graph_candidates("fountains", ["sg:dest"], [path], chunks), {})
        path["edges"] = [edges[1], edges[2], {"chunk_id": "supported", "quote": quote}]
        self.assertEqual(supported_graph_candidates("fountains", ["fountain"], [path], chunks), {})

    def test_control_and_graph_treatment_preserve_the_first_two_slots(self):
        original = ["a", "b", "c", "d", "e", "f", "g", "h"]
        multi = original.copy()
        _, control = rank_chunks(original, multi, [])
        self.assertEqual(promote_one(original, multi, control, {}), control)
        ranked = promote_one(original, multi, control, {"h": 0.8, "off_shortlist": 1.0})
        self.assertEqual(ranked[:3], ["a", "b", "h"])
        self.assertCountEqual(ranked, original)

    def test_multi_query_can_earn_slot_without_graph(self):
        original = ["a", "b", "c", "d"]
        multi = ["d", "a", "b", "c"]
        _, control = rank_chunks(original, multi, [])
        self.assertEqual(promote_one(original, multi, control, {})[:3], ["a", "b", "d"])

    def test_paired_deltas_isolate_graph_from_multi_query(self):
        rankings = {
            "hybrid": ["a", "b", "x", "gold"], "hybrid_without_graph": ["a", "b", "x", "gold"],
            "promote_multi": ["a", "b", "x", "gold"], "promote_graph": ["a", "b", "gold", "x"],
        }
        rows = [{
            "question_id": "Q", "type": "multi", "method": method,
            "gold_ranks": str(ranking.index("gold") + 1), **score_ranking(ranking, ["gold"]),
        } for method, ranking in rankings.items()]
        historical, experiment, _ = paired_deltas(rows)
        self.assertEqual(historical["delta_recall_at_3"], 0.0)
        self.assertEqual(experiment["delta_recall_at_3"], 1.0)
        self.assertEqual(experiment["delta_recall_at_5"], 0.0)

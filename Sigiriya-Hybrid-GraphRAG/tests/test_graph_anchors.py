"""Offline graph anchor checks using the existing Sigiriya graph manifest."""

import json
import re
import unittest
from pathlib import Path

from src.retrieval import find_anchors, graph_paths, resolve_anchors


class ManifestSession:
    def __init__(self, facts, entities, chunks):
        self.by_entity = {row["id"]: row for row in entities}
        self.by_chunk = chunks
        self.links = {}
        self.edges = {}
        self.seeds = []
        for index, fact in enumerate(facts):
            subject, target, chunk = fact["subject"], fact["object"], fact["chunk_id"]
            self.links.setdefault(chunk, set()).update((subject, target))
            edge = {
                "relation": fact["relation"], "fact_id": f"fixture-{index}",
                "chunk_id": chunk, "quote": fact["needle"],
            }
            self.edges.setdefault(subject, []).append((target, edge))

    def run(self, query, **params):
        if "LINKS_TO_ENTITY" in query:
            if "c.text AS evidence" not in query:
                raise AssertionError("The linked-entity query must return source text.")
            return [
                {**self.by_entity[entity], **self.by_chunk[chunk], "pos": position}
                for position, chunk in enumerate(params["ids"])
                for entity in sorted(self.links.get(chunk, []))
            ]
        if "relationships(p)" in query:
            if "*1..2" not in query:
                raise AssertionError("Expected one-hop and two-hop traversal.")
            self.seeds = params["seeds"]
            rows = []
            for seed in self.seeds:
                for target, edge in self.edges.get(seed, []):
                    rows.append(self.path([seed, target], [edge]))
                    for final, next_edge in self.edges.get(target, []):
                        rows.append(self.path([seed, target, final], [edge, next_edge]))
            return rows
        raise AssertionError(query)

    def path(self, ids, edges):
        return {
            "seed": ids[0], "entity_ids": ids, "edges": edges,
            "names": [self.by_entity[entity]["name"] for entity in ids],
        }


class GraphAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project = Path(__file__).resolve().parents[1]
        manifest_path = project / "original_group_work" / "Sigiriya_KG_Expansion_Manifest.json"
        cls.manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        corpus_path = next((project / "data" / "raw").glob("SIGIRIYA*.txt"))
        sections = re.findall(
            r"(?ms)^## (\d+)\. ([^\n]+)\n(.*?)(?=^## \d+\.|\Z)",
            corpus_path.read_text(encoding="utf-8-sig"),
        )
        cls.chunks = {
            f"sigiriya_{int(number):03d}": {"heading": title, "evidence": body.strip()}
            for number, title, body in sections
        }
        cls.entities = cls.manifest["entities"]

    def setUp(self):
        self.session = ManifestSession(self.manifest["facts"], self.entities, self.chunks)

    def resolve(self, question, *ranking):
        return resolve_anchors(self.session, question, self.entities, list(ranking))

    def test_explicit_fresco_filters_unrelated_vector_chunks(self):
        question = "සීගිරි බිතුසිතුවම් ගැන කියන්න."
        anchors = self.resolve(
            question, "sigiriya_004", "sigiriya_012", "sigiriya_021", "sigiriya_015",
        )
        self.assertEqual(anchors, ["sg:fresco"])
        paths = graph_paths(self.session, question, self.entities, anchors)
        self.assertEqual(
            {tuple(row["entity_ids"]) for row in paths},
            {("sg:dest", "sg:fresco"), ("sg:fresco", "sg:women"), ("sg:dest", "sg:fresco", "sg:women")},
        )

    def test_multi_entity_water_gardens_preserve_both_neighborhoods(self):
        question = "ජල උද්‍යාන සහ දිය උල්පත් ගැන කියන්න."
        anchors = self.resolve(
            question, "sigiriya_006", "sigiriya_007", "sigiriya_015", "sigiriya_012",
        )
        self.assertCountEqual(anchors, ["sg:watergarden", "sg:fountain"])
        paths = graph_paths(self.session, question, self.entities, anchors)
        routes = {tuple(row["entity_ids"]) for row in paths}
        self.assertIn(("sg:watergarden", "sg:channels"), routes)
        self.assertIn(("sg:dest", "sg:fountain", "sg:pressure"), routes)
        self.assertFalse(any(
            {"sg:safari", "sg:hiking", "sg:wasps"}.intersection(route) for route in routes
        ))

    def test_alternate_alias_and_inflection_are_explicit(self):
        self.assertEqual(find_anchors("දිව්‍ය කාන්තාවන් ගැන කියන්න.", self.entities), ["sg:women"])
        self.assertEqual(find_anchors("සිතුවම්වල රූප මොනවාද?", self.entities), ["sg:fresco"])
        self.assertEqual(find_anchors("Tell me about the Mirror Wall.", self.entities), ["sg:mirror"])
        self.assertEqual(find_anchors("Tell me about guidelines.", self.entities), [])

    def test_paraphrase_recovers_entity_from_relevant_chunk(self):
        question = "පර්වතයේ බටහිර මුහුණතේ තිබූ චිත්‍ර ගැන කියන්න."
        self.assertEqual(find_anchors(question, self.entities), [])
        self.assertEqual(
            self.resolve(question, "sigiriya_021", "sigiriya_004", "sigiriya_015"),
            ["sg:fresco"],
        )

    def test_recovery_adds_a_paraphrased_topic_beside_an_explicit_one(self):
        question = "කැටපත් පවුර සහ පර්වතයේ බටහිර මුහුණතේ තිබූ චිත්‍ර ගැන කියන්න."
        self.assertEqual(
            self.resolve(question, "sigiriya_005", "sigiriya_004", "sigiriya_021"),
            ["sg:mirror", "sg:fresco"],
        )

    def test_two_hop_bridge_retains_both_facts_and_source_references(self):
        question = "දිව්‍ය කාන්තාවන් ගැන කියන්න."
        anchors = self.resolve(question, "sigiriya_004", "sigiriya_021")
        self.assertEqual(anchors, ["sg:women"])
        path, = graph_paths(self.session, question, self.entities, anchors)
        self.assertEqual(path["entity_ids"], ["sg:dest", "sg:fresco", "sg:women"])
        self.assertEqual([edge["relation"] for edge in path["edges"]], ["HAS_ATTRACTION", "DEPICTS"])
        for edge in path["edges"]:
            self.assertTrue(edge["fact_id"])
            self.assertEqual(edge["chunk_id"], "sigiriya_004")
            self.assertIn(edge["quote"], self.chunks[edge["chunk_id"]]["evidence"])

    def test_no_entity_does_not_promote_rank_only_candidates(self):
        question = "අඟහරු ග්‍රහලෝකය ගැන කියන්න."
        anchors = self.resolve(question, "sigiriya_012", "sigiriya_021", "sigiriya_015")
        self.assertEqual(anchors, [])
        self.assertEqual(graph_paths(self.session, question, self.entities, anchors), [])
        self.assertEqual(self.session.seeds, [])

    def test_destination_overview_keeps_the_destination_fallback(self):
        question = "සීගිරිය ගැන කියන්න."
        anchors = self.resolve(question, "sigiriya_021", "sigiriya_015")
        self.assertEqual(anchors, ["sg:dest"])
        self.assertTrue(graph_paths(self.session, question, self.entities, anchors))
        self.assertEqual(self.session.seeds, ["sg:dest"])

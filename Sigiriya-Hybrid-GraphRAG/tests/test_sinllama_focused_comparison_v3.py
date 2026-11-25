import unittest
from copy import deepcopy
from unittest.mock import patch

from scripts.sinllama_diagnostic_v1 import EVIDENCE, load_packet
from scripts.sinllama_focused_comparison_v3 import (
    FocusedComparisonGenerator, audit, run_comparison, run_selected,
)
from scripts.sinllama_legacy_rag_comparison_v1 import LEGACY_V4, ComparisonGenerator
from scripts.sinllama_legacy_rag_comparison_v2 import topic_rows
from src.generation import _quote_pieces
from src.prompting import build_prompt


class FocusedComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _ = load_packet(EVIDENCE)
        cls.topics = topic_rows(cls.rows["Q03"])

    def test_prompt_exact_and_clean_evidence_changes_only_rendered_metadata(self):
        topic = self.topics[0]
        source = deepcopy(topic["sources"][0])
        source["source_quote"] += "\n[retained] 10. Quoted content."
        prompt = build_prompt(topic["question"], [source], variant="focused_sinhala_qa", clean_evidence=True)
        expected = (
            "පහත තොරතුරු පමණක් භාවිත කරමින් ප්‍රශ්නයට පිළිතුරු දෙන්න.\n"
            "අදාළ වැදගත් කරුණු සියල්ල ඇතුළත් කරන්න.\n"
            "තොරතුරුවල නොමැති කරුණු එකතු නොකරන්න.\n"
            "අවිනිශ්චිත කරුණු නිශ්චිත ලෙස නොකියන්න.\n"
            "ප්‍රශ්නය නැවත නොලියන්න.\n"
            "පිළිතුර අවසන් වූ විට <END_ANSWER> යොදන්න.\n\n"
            f"තොරතුරු:\n{topic['heading']}\n{source['source_quote']}\n\n"
            f"ප්‍රශ්නය: {topic['question']}\nපිළිතුර:"
        )
        self.assertEqual(prompt, expected)
        with_metadata = build_prompt(topic["question"], [source], variant="focused_sinhala_qa")
        self.assertEqual(with_metadata.replace(f"[{source['chunk_id']}] 4. ", "", 1), prompt)

    def test_generation_settings_and_boundaries_match_legacy(self):
        topic = self.topics[0]
        reference = ComparisonGenerator(config=LEGACY_V4)
        generator = FocusedComparisonGenerator(config=LEGACY_V4)
        seen = []

        def propose(question, sources):
            seen.append((generator.config, generator._prompt_markers(), generator._build_prompt(question, sources)))
            generator.last_generation = {"raw_model_output": ""}
            return ""

        with patch.object(generator, "propose", side_effect=propose), patch.object(generator, "_tokenizer"):
            for arm in ("A", "B", "C"):
                generator.generate_unit(arm, topic)
        for config, markers, _ in seen:
            self.assertEqual(config, LEGACY_V4)
            self.assertEqual(markers, reference._prompt_markers())
        self.assertEqual(seen[0][2], reference._build_prompt(topic["question"], topic["sources"]))
        self.assertIn("[sigiriya_004]", seen[1][2])
        self.assertNotIn("[sigiriya_004]", seen[2][2])

    def test_v2_label_regression_keeps_incomplete_fresco_answer_rejected(self):
        topic = self.topics[0]
        claim = _quote_pieces(topic["sources"][0]["source_quote"])[-1]
        generation = {"raw_model_output": claim + "ප්‍රශ්නය:", "candidate": claim}
        self.assertIn("ප්‍රශ්නය:", audit(topic, generation, v2_cleanup=True)["cleaned_model_answer"])
        checked = audit(topic, generation, label_cleanup=True)
        self.assertEqual(checked["cleaned_model_answer"], claim)
        self.assertEqual(checked["prompt_labels_leaked"], ["ප්‍රශ්නය:"])
        self.assertEqual(checked["prompt_labels_remaining"], [])
        self.assertFalse(checked["question_echo_detected"])
        self.assertEqual((checked["matched_exact_source_claims"], checked["required_source_claims"]), (1, 5))
        self.assertEqual(checked["strict"]["reason"], "missing_selected_evidence")
        self.assertEqual(checked["final_answer"], topic["extractive_fallback"])

    def test_cleanup_keeps_claims_echoes_and_unknown_ids_for_verification(self):
        topic = self.topics[1]
        quote = topic["sources"][0]["source_quote"]
        for text, expected in (
            (quote + "\nතොරතුරු:", quote),
            (quote + "\nතොරතුරු: Unsupported claim.", quote + "\nතොරතුරු: Unsupported claim."),
            (quote + " [unknown]\nතොරතුරු:", quote + " [unknown]"),
            (quote + "\nප්‍රශ්නය: " + topic["question"], quote + "\nප්‍රශ්නය: " + topic["question"]),
        ):
            checked = audit(topic, {"raw_model_output": text, "candidate": text}, label_cleanup=True)
            self.assertEqual(checked["cleaned_model_answer"], expected)
            self.assertEqual(checked["strict"]["accepted"], expected == quote)
            self.assertEqual(checked["question_echo_detected"], topic["question"] in text)

    def test_topics_and_regressions_use_selected_sources_and_independent_fallback(self):
        def generate(arm, row):
            quote = row["sources"][0]["source_quote"]
            if row.get("heading") == self.topics[1]["heading"]:
                quote = "Unsupported claim.තොරතුරු:"
            return {"raw_model_output": quote, "candidate": quote}

        records = list(run_comparison(self.rows, generate))
        self.assertEqual(len(records), 8)
        for index in range(0, 6, 2):
            first, second = records[index:index + 2]
            self.assertEqual(first["selected_evidence"], self.topics[0]["sources"])
            self.assertEqual(second["selected_evidence"], self.topics[1]["sources"])
            self.assertTrue(first["strict"]["accepted"])
            self.assertEqual(second["final_answer"], self.topics[1]["extractive_fallback"])
        self.assertTrue(all(record["strict"]["accepted"] for record in records[6:]))

    def test_selected_inference_reuses_the_comparison_decisions(self):
        def generate(arm, row):
            raw = row["sources"][0]["source_quote"]
            return {"raw_model_output": raw, "candidate": raw}

        payload = {"kind": "sigiriya_evidence", "schema_version": 1,
                   "questions": self.rows.values()}
        comparisons = list(run_comparison(self.rows, generate))
        for arm in ("A", "B", "C"):
            selection = {"multi_topic": arm, "factual": "R-Q01", "mechanism": "R-Q05"}
            records = list(run_selected(payload, selection, generate))
            expected = [row for row in comparisons if row["arm"] in selection.values()]
            for record, comparison in zip(records, expected, strict=True):
                self.assertEqual(record["raw_model_output"], comparison["raw_model_output"])
                self.assertEqual(record["selected_evidence"], comparison["selected_evidence"])
                self.assertEqual(record["strict"], comparison["strict"])

    def test_selected_inference_skips_generation_for_unsupported_questions(self):
        row = {**self.rows["Q01"], "id": "Q02", "sources": [],
               "status": "INSUFFICIENT_DOCUMENT_EVIDENCE"}
        payload = {"kind": "sigiriya_evidence", "schema_version": 1, "questions": [row]}
        with patch.object(FocusedComparisonGenerator, "generate_unit") as generate:
            record, = run_selected(payload, {"Q02": "R-Q01"}, generate)
        generate.assert_not_called()
        self.assertTrue(record["strict"]["fallback_used"])
        self.assertEqual(record["final_answer"], row["extractive_fallback"])

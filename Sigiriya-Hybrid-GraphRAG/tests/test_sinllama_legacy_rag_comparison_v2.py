import unittest
from copy import deepcopy

from scripts.sinllama_diagnostic_v1 import EVIDENCE, load_packet
from scripts.sinllama_legacy_rag_comparison_v1 import SETTINGS
from scripts.sinllama_legacy_rag_comparison_v2 import PROFILES, assess, compare, topic_rows
from src.generation import clean_experimental_completion


def completion(text):
    return {"raw_model_output": text, "candidate": text}


class ComparisonV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _ = load_packet(EVIDENCE)

    def test_marker_variants_pass_both_existing_verifiers(self):
        row = self.rows["Q01"]
        quote = row["sources"][0]["source_quote"]
        for marker in ("<END_ANSWER>", "<endanswer>", "<end answer>", "< EnD \t_ AnSwEr >", "<end\nanswer>"):
            with self.subTest(marker=marker):
                raw = quote + marker
                result = assess(row, completion(raw), cleanup=True)
                self.assertEqual(result["raw_model_output"], raw)
                self.assertEqual(result["cleaned_model_answer"], quote)
                self.assertTrue(result["strict"]["accepted"])
                self.assertTrue(result["experimental"]["accepted"])

    def test_source_id_cleanup_is_selected_exact_and_outside_the_sentence(self):
        row = self.rows["Q05"]
        sources = row["sources"]
        quote = sources[0]["source_quote"]
        for text in (f"{quote} [sigiriya_007] <end answer>",
                     f"{quote}\n[sigiriya_007]\n<endanswer>",
                     f"{quote}<endanswer> [sigiriya_007]",
                     f"[sigiriya_007]\n{quote}<endanswer>"):
            with self.subTest(text=text):
                result = assess(row, completion(text), cleanup=True)
                self.assertEqual(result["cleaned_model_answer"], quote)
                self.assertTrue(result["strict"]["accepted"])
        for text in (f"{quote} [sigiriya_004]", f"{quote} [SIGIRIYA_007]",
                     f"{quote} [unknown]", f"{quote[:-1]} [sigiriya_007]."):
            with self.subTest(text=text):
                result = assess(row, completion(text + "<endanswer>"), cleanup=True)
                self.assertEqual(result["cleaned_model_answer"], text)
                self.assertFalse(result["strict"]["accepted"])
                self.assertFalse(result["experimental"]["accepted"])
        quoted_id = f"{quote} [sigiriya_007]"
        self.assertEqual(clean_experimental_completion(
            quoted_id, [{**sources[0], "source_quote": quoted_id}]
        ), quoted_id)
        raw = f"{quote} <END_ANSWER> [unknown]"
        result = assess(row, {"raw_model_output": raw, "candidate": quote}, cleanup=True)
        self.assertIn("[unknown]", result["cleaned_model_answer"])
        self.assertFalse(result["experimental"]["accepted"])

    def test_claims_and_unrecognized_text_are_preserved(self):
        row = self.rows["Q05"]
        quote = row["sources"][0]["source_quote"]
        altered = quote.replace("වූ බව සැලකේ.", "විය.")
        result = assess(row, completion(altered + " <END_ANSWER>"), cleanup=True)
        self.assertEqual(result["cleaned_model_answer"], altered)
        self.assertFalse(result["strict"]["accepted"])
        self.assertFalse(result["experimental"]["accepted"])
        for text in (quote + " <end-answer>", quote + " <end answer> Extra claim."):
            self.assertEqual(clean_experimental_completion(text, row["sources"]), text)

    def test_question_echo_detected_even_when_existing_boundary_hides_it(self):
        row = self.rows["Q01"]
        quote = row["sources"][0]["source_quote"]
        raw = f"{quote}\nQuestion: {row['question']}"
        result = assess(row, {"raw_model_output": raw, "candidate": quote}, cleanup=True)
        self.assertTrue(result["question_echo_detected"])
        self.assertEqual(result["strict"]["reason"], "question_echo")
        self.assertEqual(result["final_answer"], row["extractive_fallback"])

    def test_single_topic_uses_same_b_completion_and_unchanged_baselines(self):
        self.assertEqual(PROFILES["A"], SETTINGS["A"])
        self.assertEqual(PROFILES["B"], SETTINGS["B"])
        row = self.rows["Q05"]
        calls = []

        def generate(arm, selected):
            calls.append((arm, selected))
            return completion(row["sources"][0]["source_quote"] + "<endanswer>")

        a, b, c = list(compare({"Q05": row}, generate))
        self.assertEqual(calls, [("A", row), ("B", row)])
        self.assertEqual(b["raw_model_output"], c["raw_model_output"])
        self.assertFalse(b["strict"]["accepted"])
        self.assertTrue(c["strict"]["accepted"])

    def test_topics_use_separate_evidence_and_verify_and_fallback_independently(self):
        row = self.rows["Q03"]
        first, second = row["sources"]
        calls = []

        def generate(arm, selected):
            calls.append((arm, selected["question"], selected["sources"]))
            # Fresco text is grounded for the first topic only.
            return completion(first["source_quote"] + " <end answer>")

        *_, result = compare({"Q03": row}, generate)
        self.assertEqual(result["arm"], "D")
        self.assertEqual(calls[-2:], [
            ("B", row["subquestions"][0], [first]), ("B", row["subquestions"][1], [second]),
        ])
        frescoes, wall = result["subquestions"]
        for mode in ("strict", "experimental"):
            self.assertEqual(frescoes[mode]["verified_answer"], first["source_quote"])
            self.assertIsNone(frescoes[mode]["fallback_answer"])
            self.assertIsNone(wall[mode]["verified_answer"])
            self.assertEqual(wall[mode]["fallback_answer"], second["source_quote"])
            self.assertEqual(result["combined_final_answers"][mode], row["extractive_fallback"])

    def test_misaligned_packet_does_not_cross_feed_topics(self):
        row = deepcopy(self.rows["Q03"])
        row["sources"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered topic mapping"):
            topic_rows(row)


if __name__ == "__main__":
    unittest.main()

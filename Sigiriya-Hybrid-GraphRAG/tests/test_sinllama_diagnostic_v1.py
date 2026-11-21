import unittest

from scripts.sinllama_diagnostic_v1 import (
    EVIDENCE, PROFILES, diagnostic_prompt, load_packet, offline_results, run_generation,
)
from src.prompting import build_evidence_context, build_prompt


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _ = load_packet(EVIDENCE)

    def test_prompt_controls(self):
        for row in self.rows.values():
            question, sources = row["question"], row["sources"]
            self.assertEqual(diagnostic_prompt("question_only", question, sources), question)
            for profile in PROFILES[1:]:
                prompt = diagnostic_prompt(profile, question, sources)
                self.assertIn(build_evidence_context(sources, max_chars_each=None), prompt)
                self.assertEqual(prompt, build_prompt(question, sources, variant=profile))

    def test_verification_matrix(self):
        results = {row["case"]: row for row in offline_results(self.rows)}
        self.assertEqual(
            {case for case, row in results.items() if row["strict"]["accepted"]}, {"exact_supported"},
        )
        self.assertEqual(
            {case for case, row in results.items() if not row["experimental"]["fallback_used"]},
            {"exact_supported", "allowlisted_paraphrase", "mixed_suffix"},
        )
        self.assertEqual(results["incomplete"]["strict"]["reason"], "missing_selected_evidence")
        checks = results["certainty_changed"]["experimental"]["first_mismatch_checks"]
        self.assertFalse(checks["uncertainty_preserved"])
        checks = results["short_paraphrase"]["experimental"]["first_mismatch_checks"]
        self.assertTrue(all(checks.values()))
        mixed = results["mixed_suffix"]
        self.assertEqual(mixed["experimental"]["origin"], "verified_recovered_prefix")
        self.assertEqual(mixed["experimental"]["verified_answer"],
                         self.rows["Q05"]["sources"][0]["source_quote"])
        self.assertTrue(mixed["experimental"]["discarded_continuation"].strip())
        self.assertTrue(mixed["experimental"]["manual_review"])

    def test_generation_audit_keeps_echo_and_cleaned_candidate(self):
        class Echo:
            def propose(self, question, sources):
                self.last_generation = {
                    "raw_model_output": question, "stop_reason": "prompt_marker_or_question_echo",
                }
                return ""

        records = list(run_generation(self.rows, Echo()))
        self.assertEqual(len(records), 9)
        for row in records:
            self.assertEqual(row["raw_model_output"], self.rows[row["question_id"]]["question"])
            self.assertEqual(row["cleaned_output"], "")
            self.assertEqual(row["generation_findings"], ["question_echo", "empty_cleaned_output"])
            self.assertIsNone(row["experimental"]["verified_answer"])
            self.assertEqual(row["experimental"]["final_answer"], row["extractive_fallback"])

    def test_inference_errors_are_distinct_from_verifier_rejections(self):
        class Broken:
            def propose(self, question, sources):
                raise RuntimeError("inference failed")

        for row in run_generation(self.rows, Broken()):
            self.assertIn("RuntimeError", row["generation_error"])
            self.assertIsNone(row["strict"])
            self.assertIsNone(row["experimental"])
            self.assertEqual(row["final_answer"], row["extractive_fallback"])


if __name__ == "__main__":
    unittest.main()

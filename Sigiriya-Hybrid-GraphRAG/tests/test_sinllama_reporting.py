import unittest

from scripts.sinllama_focused_comparison_v3 import audit
from scripts.sinllama_reporting import classify, print_report


def result(raw, *, quote="The wall was polished. Poems were written on it.",
           question="Tell me about the wall.", label_cleanup=False):
    row = {
        "question": question, "extractive_fallback": quote or "No grounded answer.",
        "sources": [{"chunk_id": "wall_01", "source_quote": quote}] if quote else [],
    }
    return audit(row, {"raw_model_output": raw, "candidate": (raw or "").strip()},
                 label_cleanup=label_cleanup)


class ReportingTests(unittest.TestCase):
    def test_verified_and_cleaned_answers_are_labeled_separately(self):
        plain = result("The wall was polished.", quote="The wall was polished.")
        cleaned = result("The wall was polished.<end answer>", quote="The wall was polished.",
                         label_cleanup=True)
        for record, expected in ((plain, "SINLLAMA_GENERATED"), (cleaned, "SINLLAMA_GENERATED_CLEANED")):
            output = []
            print_report([{**record, "question_id": "Q01"}], emit=output.append)
            self.assertEqual(classify(record)["classification"], "VERIFIED_SUCCESS")
            self.assertIn(expected, output)
            self.assertIn("- Verification: ACCEPTED", output)
            sections = ["QUESTION NUMBER AND QUESTION", "RAW SINLLAMA ANSWER", "SINLLAMA ANSWER STATUS",
                        "FINAL GROUNDED ANSWER", "ANSWER SOURCE", "SOURCE EVIDENCE", "FALLBACK"]
            self.assertEqual([line for line in output if line in sections], sections)

    def test_unrelated_requires_confirmed_review(self):
        record = result("The moon orbits Earth.")
        self.assertEqual(classify(record)["classification"], "NEEDS_MANUAL_REVIEW")
        self.assertEqual(classify(record)["relevant"], "UNCERTAIN")
        status = classify({**record, "reviewed_relevance": "NO"})
        self.assertEqual((status["classification"], status["relevant"]), ("UNRELATED_GENERATION", "NO"))

    def test_supported_but_incomplete_preserves_supported_and_missing_claims(self):
        record = result("The wall was polished.")
        status = classify(record)
        self.assertEqual(status["classification"], "PARTIAL_ANSWER")
        self.assertEqual((status["source_supported"], status["complete"]), ("YES", "PARTIAL"))
        self.assertEqual(status["supported_claims"], ["The wall was polished."])
        self.assertEqual(status["missing_claims"], ["Poems were written on it."])
        self.assertTrue(record["strict"]["fallback_used"])

    def test_supported_content_with_unsupported_claim(self):
        status = classify(result("The wall was polished. The wall is made of gold."))
        self.assertEqual((status["classification"], status["source_supported"]), ("UNSUPPORTED_CLAIM", "PARTIAL"))
        self.assertEqual(status["unmatched_claims"], ["The wall is made of gold."])

    def test_uncertainty_change_requires_matching_claim(self):
        record = result("මෙම පද්ධතිය ක්‍රියාත්මක විය.", quote="මෙම පද්ධතිය ක්‍රියාත්මක වූ බව සැලකේ.")
        self.assertEqual(classify(record)["classification"], "UNCERTAINTY_CHANGED")
        other = result("පවුර විය.", quote="මෙම පද්ධතිය ක්‍රියාත්මක වූ බව සැලකේ.")
        self.assertEqual(classify(other)["classification"], "NEEDS_MANUAL_REVIEW")

    def test_question_echo_and_format_error(self):
        self.assertEqual(classify(result("Tell me about the wall."))["classification"], "QUESTION_ECHO")
        self.assertEqual(classify(result("The wall was polished.<endanswer>"))["classification"], "FORMAT_ERROR")

    def test_missing_evidence_and_missing_generation(self):
        record = result(None, quote="")
        self.assertEqual(classify(record)["classification"], "NO_RELEVANT_EVIDENCE")
        self.assertEqual(classify(record)["generated"], "NO")
        output = []
        print_report([{**record, "question_id": "Q01"}], emit=output.append)
        self.assertIn("No supporting source was found.", output)
        self.assertIn("EXTRACTIVE_FALLBACK", output)

    def test_grouping_and_mixed_topic_provenance(self):
        first = result("The wall was polished.", quote="The wall was polished.")
        second = result("Eagles migrate.", question="Tell me about the poems.",
                        quote="Poems were written on the wall.")
        records = [
            {**first, "question_id": "Q03", "arm": "A", "topic_heading": "Wall"},
            {**first, "question_id": "Q01", "arm": "A"},
            {**second, "question_id": "Q03", "arm": "A", "topic_heading": "Poems"},
            {**first, "question_id": "Q03", "arm": "B", "topic_heading": "Wall"},
        ]
        output = []
        print_report(records, questions={"Q03": "Tell me about the wall and its poems."}, emit=output.append)
        self.assertEqual(output.count("QUESTION NUMBER AND QUESTION"), 2)
        self.assertLess(output.index("Q01  Tell me about the wall."),
                        output.index("Q03  Tell me about the wall and its poems."))
        self.assertLess(output.index("SUBQUESTION 2  Tell me about the poems."), output.index("ARM B"))
        self.assertIn("MIXED_VERIFIED_AND_FALLBACK", output)
        self.assertIn("Poems: Poems were written on the wall.", output)
        self.assertIn("[wall_01] Poems were written on the wall.", output)
        self.assertIn("Accepted generations: 1", output)
        self.assertIn("Extractive fallbacks: 1", output)

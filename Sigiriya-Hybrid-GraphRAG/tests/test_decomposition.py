"""Regression checks for Sinhala topic and intent preservation."""

import unittest
from unittest.mock import patch

from src.retrieval import _multi_query, clean_topic, decompose


class DecompositionTests(unittest.TestCase):
    def test_single_topic_keeps_the_original(self):
        question = "සීගිරි බිතුසිතුවම් ගැන කියන්න."
        self.assertEqual(decompose(question), [question])

    def test_general_explanation_copies_the_request_once(self):
        self.assertEqual(clean_topic("  කැටපත් පවුර විස්තර කරන්න.  "), "කැටපත් පවුර")
        self.assertEqual(
            decompose("බිතුසිතුවම් සහ කැටපත් පවුර විස්තර කරන්න."),
            ["බිතුසිතුවම් විස්තර කරන්න.", "කැටපත් පවුර විස්තර කරන්න."],
        )

    def test_both_topics_keep_the_operating_mechanism(self):
        for marker, action in (
            (" දෙකම", "ක්‍රියාත්මක වන ආකාරය පැහැදිලි කරන්න."),
            ("", "ක්‍රියාත්මක වන්නේ කෙසේද?"),
            ("", "ක්‍රියාත්මක වන්නේ කොහොමද?"),
            ("", "භාවිත කරන්නේ කෙසේද?"),
        ):
            with self.subTest(action=action):
                self.assertEqual(
                    decompose(f"දිය උල්පත් සහ ජල පද්ධති{marker} {action}"),
                    [f"දිය උල්පත් {action}", f"ජල පද්ධති {action}"],
                )

    def test_overview_and_embedded_mechanism_keep_their_respective_scopes(self):
        self.assertEqual(
            decompose("ජල උද්‍යාන සහ දිය උල්පත් ක්‍රියාත්මක වන ආකාරය විස්තර කරන්න."),
            [
                "ජල උද්‍යාන ගැන විස්තර කියන්න.",
                "දිය උල්පත් ක්‍රියාත්මක වන ආකාරය විස්තර කරන්න.",
            ],
        )

    def test_locations_keep_the_location_predicate(self):
        self.assertEqual(
            decompose("කැටපත් පවුර සහ නාග පෙන ගුහාව පිහිටා ඇත්තේ කොහේද?"),
            ["කැටපත් පවුර පිහිටා ඇත්තේ කොහේද?", "නාග පෙන ගුහාව පිහිටා ඇත්තේ කොහේද?"],
        )

    def test_reason_keeps_the_predicate_and_why(self):
        self.assertEqual(
            decompose("බිතුසිතුවම් සහ කැටපත් පවුර වැදගත් වන්නේ ඇයි?"),
            ["බිතුසිතුවම් වැදගත් වන්නේ ඇයි?", "කැටපත් පවුර වැදගත් වන්නේ ඇයි?"],
        )

    def test_comparison_keeps_both_sides_in_one_question(self):
        question = "ජල උද්‍යාන සහ ගල් උද්‍යානය අතර වෙනස්කම් මොනවාද?"
        self.assertEqual(decompose(question), [question])

    def test_coupled_concepts_stay_together(self):
        for question in (
            "ගුරුත්වාකර්ෂණය සහ ජල පීඩනය භාවිත කරන්නේ කෙසේද?",
            "ආහාර සහ පාන ගැන විස්තර කරන්න.",
        ):
            with self.subTest(question=question):
                self.assertEqual(decompose(question), [question])

    def test_ambiguous_or_unsupported_questions_keep_the_original(self):
        for question in (
            "බිතුසිතුවම් සහ කැටපත් පවුරට සම්බන්ධ වන්නේ කුමක්ද?",
            "බිතුසිතුවම් සහ පවුර සහ උද්‍යාන ගැන කියන්න.",
            "ගල් සහ මැටි යොදාගෙන බිත්ති තනන ආකාරය පැහැදිලි කරන්න.",
            "  පවුර සහ උද්‍යාන  වෙනුවෙන් කුමක් කළ යුතුද?  ",
            "ගමන් මාර්ගයත් එහි වේලාවත් මොනවාද?",
        ):
            with self.subTest(question=question):
                self.assertEqual(decompose(question), [question])

    def test_multi_query_keeps_raw_topic_rankings_for_evidence(self):
        question = "බිතුසිතුවම් සහ කැටපත් පවුර ගැන කියන්න."
        with patch("src.retrieval.fused_search", return_value=["other", "art", "wall"]):
            topics, merged, raw = _multi_query(
                None, None, question, {"art": "බිතුසිතුවම්", "wall": "කැටපත් පවුර"}
            )
        self.assertEqual(topics, ["බිතුසිතුවම් ගැන කියන්න.", "කැටපත් පවුර ගැන කියන්න."])
        self.assertEqual(merged, ["art", "wall", "other"])
        self.assertEqual(raw, [["other", "art", "wall"], ["other", "art", "wall"]])

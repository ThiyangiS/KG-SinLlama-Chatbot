"""Offline evidence regressions using verbatim sections of the Sigiriya source."""

import re
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.chatbot import SigiriyaChatbot
from src.evidence import select_answer_quote, split_sentences
from src.prompting import NO_LIVE_SI, NO_SOURCE_SI
from src.retrieval import RetrievalResult, decompose


FRESCOES = "සීගිරි බිතුසිතුවම්"
WATER_GARDENS = "ජල උද්‍යාන"
FOUNTAINS = "දිය උල්පත් උද්‍යානය"
MIRROR_WALL = "කැටපත් පවුර"


class EvidenceSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
        document = next(raw_dir.glob("SIGIRIYA*.txt")).read_text(encoding="utf-8-sig")
        sections = re.findall(r"(?ms)^## \d+\. ([^\n]+)\n(.*?)(?=^## \d+\.|\Z)", document)
        cls.sources = {
            title: {"chunk_id": title, "heading": title, "evidence": body.strip()}
            for title, body in sections
        }

    def setUp(self):
        self.retriever = Mock(sources=self.sources)
        self.retriever.retrieve.side_effect = self.retrieval_fixture
        self.chatbot = SigiriyaChatbot(self.retriever)

    def retrieval_fixture(self, question):
        facets = decompose(question)
        ranks = list(self.sources)
        return RetrievalResult(ranks, ranks, ranks, facets, [ranks] * len(facets), [], [], [])

    def assert_grounded(self, result, expected_topics):
        self.assertEqual([row["heading"] for row in result["sources"]], expected_topics)
        excerpts = {}
        for row in result["sources"]:
            sentences = [
                sentence
                for part in row["source_quote"].split(" […] ")
                for sentence in split_sentences(part)
            ]
            source = self.sources[row["chunk_id"]]["evidence"]
            self.assertTrue(sentences)
            self.assertEqual(len(sentences), len(set(sentences)))
            for sentence in sentences:
                self.assertIn(sentence, split_sentences(source))
            self.assertIn(row["source_quote"], result["answer"])
            excerpts[row["heading"]] = sentences
        return excerpts

    def test_fresco_overview_adds_distinct_facts(self):
        result = self.chatbot.answer(f"{FRESCOES} ගැන කියන්න.")
        self.assertEqual(result["status"], "SOURCE_GROUNDED_EXTRACTIVE")
        sentences = self.assert_grounded(result, [FRESCOES])[FRESCOES]
        self.assertTrue(3 <= len(sentences) <= 5)
        for detail in ("කාන්තා රූප", "බටහිර මුහුණත", "මීටර් 100"):
            self.assertTrue(any(detail in sentence for sentence in sentences), detail)

    def test_female_figures_get_one_direct_sentence(self):
        result = self.chatbot.answer(
            f"{FRESCOES}වල කාන්තා රූප ලෙස නිරූපණය කර ඇත්තේ කවුරුන්ද?"
        )
        self.assertEqual(result["status"], "SOURCE_GROUNDED_EXTRACTIVE")
        sentence, = self.assert_grounded(result, [FRESCOES])[FRESCOES]
        self.assertIn("කාන්තා රූප", sentence)
        self.assertIn("අප්සරාවන් හෝ දිව්‍ය කාන්තාවන්", sentence)

    def test_artist_abstains_without_citing_a_depicted_figure(self):
        for question in (
            f"{FRESCOES} ඇඳපු චිත්‍ර ශිල්පියා කවුද?",
            f"{FRESCOES}වල කාන්තා රූප නිරූපණය කළ චිත්‍ර ශිල්පියා කවුරුන්ද?",
        ):
            with self.subTest(question=question):
                result = self.chatbot.answer(question)
                self.assertEqual(result["status"], "INSUFFICIENT_DOCUMENT_EVIDENCE")
                self.assertEqual(result["answer"], NO_SOURCE_SI)
                self.assertEqual(result["sources"], [])

    def test_fountain_mechanism_variants_quote_the_operating_method(self):
        for wording in (
            "ක්‍රියාත්මක වන ආකාරය විස්තර කරන්න.",
            "ක්‍රියාත්මක වන්නේ කෙසේද?",
            "ක්‍රියාත්මක වන්නේ කොහොමද?",
            "සඳහා ජල පීඩනය භාවිත කරන්නේ කෙසේද?",
        ):
            with self.subTest(wording=wording):
                result = self.chatbot.answer(f"දිය උල්පත් {wording}")
                self.assertEqual(result["status"], "SOURCE_GROUNDED_EXTRACTIVE")
                sentences = self.assert_grounded(result, [FOUNTAINS])[FOUNTAINS]
                self.assertLessEqual(len(sentences), 2)
                self.assertTrue(any("ගුරුත්වාකර්ෂණය සහ ජල පීඩනය" in s for s in sentences))
                self.assertTrue(all("සමමිතිකව" not in s for s in sentences))

    def test_two_garden_facets_keep_separate_evidence(self):
        result = self.chatbot.answer(
            "ජල උද්‍යාන සහ දිය උල්පත් ක්‍රියාත්මක වන ආකාරය විස්තර කරන්න."
        )
        self.assertEqual(result["status"], "SOURCE_GROUNDED_EXTRACTIVE")
        self.assert_grounded(result, [WATER_GARDENS, FOUNTAINS])
        self.assertIn("ජල තටාක", result["sources"][0]["source_quote"])
        self.assertIn("භූගත ජල මාර්ග", result["sources"][0]["source_quote"])
        self.assertIn("ගුරුත්වාකර්ෂණය සහ ජල පීඩනය", result["sources"][1]["source_quote"])

    def test_two_broad_topics_keep_separate_evidence(self):
        result = self.chatbot.answer(f"{FRESCOES} සහ {MIRROR_WALL} ගැන කියන්න.")
        self.assertEqual(result["status"], "SOURCE_GROUNDED_EXTRACTIVE")
        self.assert_grounded(result, [FRESCOES, MIRROR_WALL])
        self.assertIn("කාන්තා රූප", result["sources"][0]["source_quote"])
        self.assertIn("කවි සහ විවිධ සටහන්", result["sources"][1]["source_quote"])

    def test_location_uses_one_direct_location_sentence(self):
        result = self.chatbot.answer(f"{FRESCOES} දැකිය හැක්කේ කොහේද?")
        self.assertEqual(result["status"], "SOURCE_GROUNDED_EXTRACTIVE")
        sentence, = self.assert_grounded(result, [FRESCOES])[FRESCOES]
        self.assertIn("පිහිටි පර්වතයේ කොටසක", sentence)

    def test_reason_and_missing_process_in_a_generic_section(self):
        reason = "පද්ධතිය උණුසුම නිසා ස්වයංක්‍රීයව නතර වේ."
        source = {"heading": "පද්ධතිය", "evidence": f"පද්ධතිය විශාල ගොඩනැගිල්ලක ඇත. {reason}"}
        self.assertEqual(select_answer_quote("පද්ධතිය නතර වන්නේ ඇයි?", source), (reason, True))
        self.assertEqual(select_answer_quote("පද්ධතිය භාවිත කරන්නේ කෙසේද?", source), ("", False))

    def test_partial_answer_keeps_only_the_supported_facet(self):
        result = self.chatbot.answer(f"{FRESCOES} සහ ඇන්ටාක්ටිකාවේ පෙන්ගුවින් ගැන කියන්න.")
        self.assertEqual(result["status"], "PARTIAL_SOURCE_EVIDENCE")
        self.assert_grounded(result, [FRESCOES])
        self.assertIn(NO_SOURCE_SI, result["answer"])

    def test_live_price_declines_without_retrieval_or_citations(self):
        result = self.chatbot.answer("අද සීගිරියේ ප්‍රවේශපත්‍ර මිල කීයද?")
        self.assertEqual(result["status"], "LIVE_DATA_UNSUPPORTED")
        self.assertEqual(result["answer"], NO_LIVE_SI)
        self.assertEqual(result["sources"], [])
        self.retriever.retrieve.assert_not_called()

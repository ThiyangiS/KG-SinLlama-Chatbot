"""Regression fixtures for the reported v2 failure shapes using exported evidence."""

import json
import unittest

from scripts.sinllama_handoff import DEMO_QUESTIONS, FOCUS_QUESTIONS, export_evidence, replay_completions
from src.chatbot import SigiriyaChatbot
from src.config import PROJECT_DIR
from src.generation import strict_quote_verification
from src.prompting import build_prompt
from src.verification import has_repeated_claims, verify_experimental


class ExperimentalVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prefer the actual evidence packet. A clean checkout can reconstruct
        # the nine questions from the document and manifest fixture.
        path = PROJECT_DIR / "results/generation/sigiriya_sinllama_evidence.json"
        if path.is_file():
            cls.payload = json.loads(path.read_text(encoding="utf-8-sig"))
        else:
            from test_sinllama_handoff import FixtureRetriever

            cls.payload = export_evidence(SigiriyaChatbot(FixtureRetriever()), DEMO_QUESTIONS)

    def test_q01_concatenated_repetition_recovers_complete_claim(self):
        sources = self.payload["questions"][0]["sources"]
        claim = sources[0]["source_quote"]
        candidate = claim + claim + claim.replace("මෙම", "එම", 1)
        self.assertTrue(has_repeated_claims(candidate))
        self.assertFalse(strict_quote_verification(candidate, sources, require_complete=True)[0])
        checked = verify_experimental(candidate, sources)
        self.assertEqual(checked["answer"], claim)
        self.assertEqual(checked["origin"], "verified_recovered_prefix")
        self.assertNotEqual(checked["discarded_continuation"], "")

    def test_q03_one_topic_never_counts_as_complete(self):
        sources = self.payload["questions"][2]["sources"]
        first_topic = sources[0]["source_quote"]
        checked = verify_experimental(first_topic + first_topic, sources)
        self.assertIsNone(checked["answer"])
        self.assertGreater(checked["required_claims"], checked["matched_claims"])
        self.assertEqual(checked["origin"], "extractive_fallback")
        both = " ".join(source["source_quote"] for source in sources)
        self.assertEqual(verify_experimental(both, sources)["answer"], both)

    def test_q05_only_recovers_prefix_after_verifying_both_claims(self):
        sources = self.payload["questions"][4]["sources"]
        paraphrase = sources[0]["source_quote"].replace("භාවිත කරමින්", "යොදා ගනිමින්")
        unsupported = "1. පළමු දිය උල්පතට ජලය ලීටර් 500ක් අවශ්‍ය වේ."
        self.assertFalse(strict_quote_verification(paraphrase, sources, require_complete=True)[0])
        full = verify_experimental(paraphrase, sources)
        self.assertEqual(full["origin"], "full_generated_answer")
        self.assertEqual(full["answer"], paraphrase)
        recovered = verify_experimental(paraphrase + unsupported, sources)
        self.assertEqual(recovered["answer"], paraphrase)
        self.assertEqual(recovered["origin"], "verified_recovered_prefix")
        self.assertEqual(recovered["matched_claims"], 2)
        self.assertIn("500", recovered["discarded_continuation"])
        first_claim = paraphrase.partition(".")[0] + "."
        self.assertIsNone(verify_experimental(first_claim + unsupported, sources)["answer"])
        self.assertIsNone(verify_experimental(paraphrase.replace("බව සැලකේ", "බව තහවුරුය"), sources)["answer"])

    def test_replay_does_not_replace_or_rewrite_saved_responses(self):
        records = []
        for number in FOCUS_QUESTIONS:
            row = self.payload["questions"][number - 1]
            quote = " ".join(source["source_quote"] for source in row["sources"])
            records.append({**row, "model_generated_answer": quote, "raw_model_output": quote})
        snapshot = json.dumps(records)
        result = replay_completions(self.payload, {"results": records})
        self.assertEqual([row["question_number"] for row in result["results"]], list(FOCUS_QUESTIONS))
        self.assertEqual(json.dumps(records), snapshot)
        self.assertTrue(all(row["strict_baseline"]["accepted"] for row in result["results"]))


class ClaimSafetyTests(unittest.TestCase):
    def test_numbers_names_polarity_and_uncertainty_cannot_change(self):
        quote = "Mapagala සීගිරියට කිලෝමීටර් 1.5ක් පමණ දකුණින් පිහිටා ඇතැයි සැලකේ."
        sources = [{"chunk_id": "fixture", "source_quote": quote}]
        self.assertEqual(verify_experimental(quote, sources)["matched_claims"], 1)
        for old, new, check in (
            ("1.5", "15", "numbers_preserved"),
            ("Mapagala", "Dambulla", None),
            ("පිහිටා", "නොපිහිටා", "negation_preserved"),
            ("පමණ", "", "uncertainty_preserved"),
        ):
            with self.subTest(old=old):
                checked = verify_experimental(quote.replace(old, new, 1), sources)
                self.assertIsNone(checked["answer"])
                self.assertTrue(checked["manual_review"])
                if check:
                    self.assertFalse(checked["first_mismatch_checks"][check])

    def test_few_shot_example_is_separate_and_never_verification_evidence(self):
        source = {"chunk_id": "real", "heading": "Topic", "source_quote": "Selected source."}
        prompt = build_prompt("Current question?", [source], variant="few_shot")
        example, current = prompt.split("[End example")
        self.assertNotIn("Selected source.", example)
        self.assertNotIn("රතු බෝලයක්", current)
        self.assertIsNone(verify_experimental("පෙට්ටියේ රතු බෝලයක් ඇත.", [source])["answer"])

    def test_minor_variation_in_concatenated_loop(self):
        phrase = "මෙම දිය උල්පත පද්ධතිය ක්‍රියාත්මක වූ බව සැලකේ"
        loop = phrase + phrase.replace("මෙම", "එම", 1) + phrase
        self.assertTrue(has_repeated_claims(loop))

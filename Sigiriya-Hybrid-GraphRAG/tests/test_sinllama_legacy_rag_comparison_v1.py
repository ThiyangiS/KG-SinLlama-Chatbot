import hashlib
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from scripts.sinllama_diagnostic_v1 import EVIDENCE, load_packet
from scripts.sinllama_legacy_rag_comparison_v1 import (
    BOUNDARIES, SETTINGS, ComparisonGenerator, clean_legacy_answer, review_rows, run_comparison,
)
from src.generation import PROMPT_MARKERS, SinLlamaGenerator
from src.legacy_rag_prompt import END_MARKER, render_legacy_rag_prompt
from src.prompting import PROMPT_FORMATS, build_evidence_context, build_prompt


class Tokens(list):
    @property
    def shape(self):
        return (1, len(self[0]))

    def to(self, device):
        return self


class Tokenizer:
    eos_token_id = 0

    def __call__(self, text, *, return_tensors):
        return {"input_ids": Tokens([list(map(ord, text))])}

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _ = load_packet(EVIDENCE)

    def test_prompt_matches_notebook_and_inserts_full_saved_context(self):
        template = render_legacy_rag_prompt("{question}", "{context}")
        self.assertEqual(
            hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "f957822563b47a6ef222b107e26951bfee893dbc917ff46fc71773d753c55d4c",
        )
        for row in self.rows.values():
            question, sources = row["question"], row["sources"]
            context = build_evidence_context(sources, max_chars_each=None)
            prompt = build_prompt(question, sources, variant="legacy_rag_prompt")
            self.assertEqual(prompt.count("සීගිරිය පිහිටා ඇත්තේ කොහේද?"), 1)
            self.assertEqual(prompt.count("දිය ඇල්ලේ උස කොපමණද?"), 1)
            self.assertEqual(prompt.count(END_MARKER), 3)
            self.assertTrue(prompt.endswith(f"{context}\n\nප්‍රශ්නය: {question}\nපිළිතුර:"))
            for profile in PROMPT_FORMATS:
                self.assertIn(context, build_prompt(question, sources, variant=profile))

    def test_configurations_and_production_stopping(self):
        for key in ("max_new_tokens", "max_input_tokens", "simple_max_new_tokens", "do_sample",
                    "repetition_penalty", "no_repeat_ngram_size"):
            self.assertEqual(SETTINGS["A"][key], SETTINGS["B"][key])
        self.assertEqual(SETTINGS["A"]["repetition_penalty"], 1.0)
        self.assertEqual(SETTINGS["C"]["max_new_tokens"], 220)
        self.assertEqual(SETTINGS["C"]["max_input_tokens"], 3000)
        self.assertEqual(SETTINGS["C"]["repetition_penalty"], 1.1)
        self.assertFalse(SETTINGS["C"]["do_sample"])
        self.assertEqual(SETTINGS["C"]["no_repeat_ngram_size"], 0)
        self.assertEqual(SinLlamaGenerator()._prompt_markers(), PROMPT_MARKERS)
        generator = ComparisonGenerator()
        generator._tokenizer = SimpleNamespace(pad_token_id=0)
        with patch.object(generator, "propose") as propose:
            def complete(question, sources):
                self.assertIn(END_MARKER, generator._prompt_markers())
                generator.last_generation = {
                    "raw_model_output": "Answer text<END_ANSWER>",
                    "stop_reason": "prompt_marker_or_question_echo",
                }
                return "Answer text"

            propose.side_effect = complete
            self.assertEqual(generator.generate_experiment("B", "Question?", []), "Answer text")
            self.assertEqual(generator.last_generation["stop_reason"], "end_answer_marker")

    def test_legacy_generation_ignores_prompt_markers_and_keeps_full_completion(self):
        generator = ComparisonGenerator()
        generator._tokenizer = Tokenizer()
        config = SimpleNamespace(max_length=20)

        def generate(input_ids, stopping_criteria, **kwargs):
            stop = stopping_criteria[0]
            self.assertIn(END_MARKER, generator._tokenizer.decode(input_ids[0]))
            self.assertFalse(stop(input_ids, None))
            for marker in BOUNDARIES:
                self.assertTrue(stop([input_ids[0] + list(map(ord, "Answer " + marker))], None))
            self.assertIsNone(kwargs["generation_config"].max_length)
            self.assertEqual(kwargs["max_new_tokens"], 220)
            self.assertEqual(kwargs["repetition_penalty"], 1.1)
            self.assertEqual(kwargs["eos_token_id"], kwargs["pad_token_id"])
            self.assertTrue(kwargs["use_cache"])
            return [input_ids[0] + list(map(ord, " Answer <END_ANSWER>"))]

        generator._model = SimpleNamespace(
            generation_config=config, generate=generate,
            get_input_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(device="cpu")),
        )
        fake_modules = {
            "torch": SimpleNamespace(inference_mode=nullcontext),
            "transformers": SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list),
        }
        with patch.dict("sys.modules", fake_modules):
            answer = generator.generate_experiment("C", "Question?", [])
            self.assertEqual(generator.last_generation["raw_model_output"], " Answer <END_ANSWER>")
            self.assertEqual(generator.last_generation["stop_reason"], "end_answer_marker")
            with self.assertRaisesRegex(ValueError, "exceeds the input budget"):
                generator.generate_experiment("C", "q" * 3000, [])
        self.assertEqual(answer, "Answer")
        self.assertEqual(config.max_length, 20)
        self.assertEqual(clean_legacy_answer("Answer question ： another question"), "Answer")
        self.assertEqual(clean_legacy_answer("Answer<END_ANSWER>tail"), "Answer")

    def test_all_nine_audits_keep_demonstrations_out_of_verification(self):
        generator = ComparisonGenerator()
        unsupported = "සීගිරිය මාතලේ දිස්ත්‍රික්කයේ පිහිටා ඇත."

        def complete(experiment, question, sources):
            generator.config = SimpleNamespace(prompt_variant=SETTINGS[experiment]["prompt_variant"])
            generator.last_generation = {
                "raw_model_output": unsupported + END_MARKER, "stop_reason": "end_answer_marker",
            }
            return unsupported

        with patch.object(generator, "generate_experiment", side_effect=complete):
            results = list(run_comparison(self.rows, generator))
        self.assertEqual(len(results), 9)
        for result, review in zip(results, review_rows(results)):
            original = self.rows[result["question_id"]]
            self.assertEqual(result["selected_evidence"], original["sources"])
            self.assertEqual(result["generation_config"], SETTINGS[result["experiment"]])
            self.assertEqual(result["cleaned_model_answer"], unsupported)
            self.assertEqual(result["raw_model_output"], unsupported + END_MARKER)
            self.assertEqual(review["factually_correct_human"], "")
            for mode in ("strict", "experimental"):
                self.assertFalse(review[f"{mode}_accepted"])
                self.assertTrue(review[f"{mode}_fallback"])
                self.assertIsNone(result[mode]["verified_answer"])
                self.assertEqual(result[mode]["final_answer"], original["extractive_fallback"])
                self.assertEqual(result[mode]["fallback_answer"], original["extractive_fallback"])

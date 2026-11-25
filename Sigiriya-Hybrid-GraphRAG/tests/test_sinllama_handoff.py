"""Exercise the Colab handoff offline against the checked-in Sigiriya corpus."""

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4
from zipfile import ZipFile

from scripts.diagnose_retrieval import ManifestGraph
from scripts.sinllama_handoff import (
    COMPARISON_PROFILES, DEMO_QUESTIONS, RUNTIME_FILES, RUNTIME_NAME, SYSTEM_PROMPT_PROFILES,
    compare_generation, export_evidence,
    format_question_result, generate_batch, summarize_generation, write_handoff,
)
from src.chatbot import SigiriyaChatbot
from src.config import PROJECT_DIR, SINLLAMA_MODEL
from src.evidence import overlap_score
from src.generation import (
    GenerationConfig, SinLlamaGenerator, apply_generation, _quote_pieces, _repeated_token_tail,
)
from src.prompting import SINHALA_SYSTEM_PROMPT, build_prompt
from src.retrieval import RetrievalResult, decompose, graph_paths, resolve_anchors


class FixtureRetriever:
    """Use deterministic corpus ranks and a manifest graph without E5 or Neo4j."""

    def __init__(self):
        path, = (PROJECT_DIR / "data" / "raw").glob("SIGIRIYA*.txt")
        sections = re.findall(
            r"(?ms)^##\s+(\d+)\.\s+([^\n]+)\n(.*?)(?=^##\s+\d+\.|\Z)",
            path.read_text(encoding="utf-8-sig"),
        )
        chunks = {
            f"sigiriya_{int(n):03d}": [f"{int(n)}. {heading.strip()}", body.strip()]
            for n, heading, body in sections
        }
        self.sources = {
            cid: {"chunk_id": cid, "heading": heading, "evidence": body}
            for cid, (heading, body) in chunks.items()
        }
        manifest = PROJECT_DIR / "original_group_work" / "Sigiriya_KG_Expansion_Manifest.json"
        self.graph = ManifestGraph(json.loads(manifest.read_text(encoding="utf-8-sig")), chunks)

    def _ranking(self, question):
        return sorted(
            self.sources,
            key=lambda cid: -overlap_score(
                question, self.sources[cid]["heading"] + " " + self.sources[cid]["evidence"]
            ),
        )

    def retrieve(self, question):
        topics = decompose(question)
        ranks = self._ranking(question)
        entities = list(self.graph.entities.values())
        anchors = resolve_anchors(self.graph, question, entities, ranks)
        paths = graph_paths(self.graph, question, entities, anchors)
        return RetrievalResult(ranks, ranks, ranks, topics, [
            self._ranking(topic) for topic in topics
        ], anchors, paths[:8], [])


class SinLlamaHandoffTests(unittest.TestCase):
    def test_nine_question_round_trip_and_fallbacks(self):
        retriever = FixtureRetriever()
        payload = export_evidence(SigiriyaChatbot(retriever), DEMO_QUESTIONS)
        folder = Path(__file__).parent / f"handoff-test-{uuid4().hex}"
        evidence_path = folder / "sigiriya_sinllama_evidence.json"
        runtime_path = folder / RUNTIME_NAME
        self.addCleanup(folder.rmdir)
        for path in (evidence_path, runtime_path):
            self.addCleanup(path.unlink, missing_ok=True)
        write_handoff(folder, payload)
        self.assertEqual(json.loads(evidence_path.read_text(encoding="utf-8")), payload)
        with ZipFile(runtime_path) as archive:
            self.assertCountEqual(archive.namelist(), RUNTIME_FILES)
            self.assertEqual(archive.read("src/generation.py"), (PROJECT_DIR / "src/generation.py").read_bytes())
        by_id = {row["id"]: row for row in payload["questions"]}
        self.assertEqual(len(by_id), 9)
        self.assertEqual(len(by_id["multi_topic"]["sources"]), 2)
        self.assertTrue(any(
            len(path["edges"]) == 2 for path in by_id["multi_hop_route"]["graph_paths"]
        ))
        self.assertIn("සැලකේ", by_id["mechanism"]["extractive_fallback"])
        self.assertIn("බටහිර මුහුණත", by_id["paraphrase"]["extractive_fallback"])
        for row in payload["questions"]:
            for source in row["sources"]:
                stored = retriever.sources[source["chunk_id"]]["evidence"]
                self.assertTrue(all(piece in stored for piece in _quote_pieces(source["source_quote"])))

        def proposal(question, sources):
            if question == by_id["overview"]["question"]:
                return "බිතුසිතුවම් 500ක් අදත් ඉතිරිව ඇත."
            if question == by_id["multi_topic"]["question"]:
                return sources[0]["source_quote"]
            return " ".join(piece for source in sources for piece in _quote_pieces(source["source_quote"]))

        generator = Mock(model_id=SINLLAMA_MODEL, config=GenerationConfig())
        generator.propose.side_effect = proposal
        output = generate_batch(payload, generator)
        self.assertEqual(generator.propose.call_count, 7)
        outcomes = {row["id"]: row for row in output["results"]}
        for case in ("factual", "multi_hop_route", "mechanism", "paraphrase", "reason"):
            result = outcomes[case]
            self.assertEqual(result["verification_reason"], "exact_source_quotes_only")
            self.assertEqual(result["final_answer"], result["model_generated_answer"])
        for case, reason in (("overview", "not_an_exact_selected_source_sentence"),
                             ("multi_topic", "missing_selected_evidence")):
            result = outcomes[case]
            self.assertEqual(result["fallback_reason"], reason)
            self.assertEqual(result["final_answer"], result["extractive_fallback"])
        for case in ("unsupported_person", "live_data"):
            result = outcomes[case]
            self.assertEqual(result["generation_status"], "GENERATION_SKIPPED")
            self.assertIsNone(result["model_generated_answer"])
            self.assertEqual(result["final_answer"], result["extractive_fallback"])

    def test_qualifiers_and_sentence_order_are_preserved(self):
        baseline = {"status": "SOURCE_GROUNDED_EXTRACTIVE",
                    "answer": "Water enters the channel. Gravity probably drives it.",
                    "sources": [{"source_quote": "Water enters the channel. Gravity probably drives it."}]}
        for candidate, reason in (
            ("Water enters the channel. Gravity drives it.", "not_an_exact_selected_source_sentence"),
            ("Gravity probably drives it. Water enters the channel.", "reordered_or_repeated_source_sentences"),
        ):
            with self.subTest(candidate=candidate):
                result = apply_generation("How?", baseline, Mock(propose=Mock(return_value=candidate)))
                self.assertEqual(result["fallback_reason"], reason)
                self.assertEqual(result["final_answer"], baseline["answer"])

    def test_generation_error_returns_audited_fallback(self):
        baseline = {"status": "SOURCE_GROUNDED_EXTRACTIVE", "answer": "A source sentence.",
                    "sources": [{"source_quote": "A source sentence."}]}
        result = apply_generation("Question", baseline, Mock(propose=Mock(side_effect=RuntimeError)))
        self.assertEqual(result["generation_status"], "EXTRACTIVE_FALLBACK_GENERATION_ERROR")
        self.assertEqual(result["fallback_reason"], "generation_error:RuntimeError")
        self.assertEqual(result["final_answer"], baseline["answer"])

    def test_cpu_refuses_before_loading_transformers(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        generator = SinLlamaGenerator()
        with patch.dict(sys.modules, {"torch": torch, "transformers": None}):
            with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
                generator.propose("Question", [])
        self.assertIsNone(generator._model)

    def test_short_prompts_use_same_evidence_and_vary_one_setting(self):
        sources = [{"chunk_id": "source", "heading": "Heading", "source_quote": "A source."}]
        reference = build_prompt("Question text", sources)
        for profile in COMPARISON_PROFILES.values():
            prompt = build_prompt("Question text", sources, variant=profile.prompt_variant)
            self.assertEqual(prompt.count("A source."), 1)
            self.assertEqual(prompt.count("Question text"), 1)
        penalty = COMPARISON_PROFILES["source_qa_penalty_1_15"]
        self.assertEqual(build_prompt("Question text", sources, variant=penalty.prompt_variant), reference)
        self.assertEqual(penalty.repetition_penalty, 1.15)
        self.assertEqual(penalty.no_repeat_ngram_size, 0)

    def test_decode_uses_only_new_tokens_and_keeps_raw_inline_echo(self):
        from contextlib import nullcontext

        generator = SinLlamaGenerator()
        eos = SimpleNamespace(eos_token_id=[98, 99])
        tokens = Mock(shape=(1, 2))
        tokens.to.return_value = tokens
        generator._model = Mock(generation_config=eos)
        generator._model.generate.return_value = [[1, 2, 21, 22, 99]]
        raw = "A source.ප්‍රශ්නය: next question"
        generator._tokenizer = Mock(return_value={"input_ids": tokens}, pad_token_id=99, eos_token_id=99)
        generator._tokenizer.decode.return_value = raw
        generator._tokenizer.encode.return_value = [21, 22]
        torch = SimpleNamespace(inference_mode=nullcontext)
        transformer = SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list)
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformer}):
            proposal = generator.propose("A sufficiently long question?", [
                {"chunk_id": "source", "heading": "Heading", "source_quote": "A source."},
            ])
        self.assertEqual(proposal, "A source.")
        self.assertEqual(generator.last_generation["raw_model_output"], raw)
        self.assertEqual(generator.last_generation["stop_reason"], "prompt_marker_or_question_echo")
        generator._tokenizer.decode.assert_called_once_with(
            [21, 22, 99], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        settings = generator._model.generate.call_args.kwargs
        self.assertEqual(settings["eos_token_id"], [98, 99])
        self.assertEqual(settings["max_new_tokens"], 120)
        self.assertIn("A sufficiently long question?", settings["stop_strings"])

    def test_repeat_detection_uses_consecutive_token_blocks(self):
        phrase = list(range(8))
        self.assertTrue(_repeated_token_tail(phrase * 3))
        self.assertFalse(_repeated_token_tail(phrase * 2 + list(range(10, 18))))

    def test_single_display_and_comparison_share_the_exported_evidence(self):
        payload = export_evidence(SigiriyaChatbot(FixtureRetriever()), DEMO_QUESTIONS[:2])
        raw = "Repeated words. " * 80
        generator = Mock(model_id=SINLLAMA_MODEL, config=GenerationConfig())
        events = []

        def propose(question, sources):
            events.append(("proposal", question))
            return raw

        generator.propose.side_effect = propose
        generator.last_generation = {"raw_model_output": raw, "stop_reason": "repeated_tokens"}
        result = generate_batch(
            payload, generator, question_number=2,
            on_result=lambda number, row: events.append(("display", number, format_question_result(number, row))),
        )
        self.assertEqual([event[0] for event in events], ["proposal", "display"])
        self.assertIn(f"QUESTION 02:\n{payload['questions'][1]['question']}", events[1][2])
        self.assertIn("FINAL VERIFIED ANSWER:", events[1][2])
        self.assertIn("full completion in JSON", events[1][2])
        self.assertEqual(result["results"][0]["raw_model_output"], raw)
        self.assertEqual(summarize_generation(result["results"])["fallback_rate"], 1.0)
        baseline = generate_batch(payload, generator)
        comparison = compare_generation(payload, generator, baseline=baseline)
        self.assertEqual(set(comparison["runs"]), {"historical_baseline", *COMPARISON_PROFILES})
        self.assertEqual(generator.config, GenerationConfig())
        self.assertEqual(len(comparison["manual_review"]), 2 * (len(COMPARISON_PROFILES) + 1))

    def test_system_variant_uses_the_exact_sinhala_block_and_selected_evidence(self):
        self.assertEqual(
            hashlib.sha256(SINHALA_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "fcd4b7a36eae0031807140225068bc9d2258bbc97c76e27430e10578d283d4d2",
        )
        question = DEMO_QUESTIONS[0][1]
        sources = [{"chunk_id": "fixture", "heading": "සිතුවම්", "source_quote": "කාන්තා රූප."}]
        prompt = build_prompt(question, sources, variant="sinhala_system_grounded")
        self.assertEqual(
            prompt,
            f"{SINHALA_SYSTEM_PROMPT}\n\nමූලාශ්‍ර තොරතුරු:\n[fixture] සිතුවම්\n"
            f"කාන්තා රූප.\n\nප්‍රශ්නය: {question}\nපිළිතුර:",
        )
        self.assertEqual(prompt.count(SINHALA_SYSTEM_PROMPT), 1)
        self.assertEqual(SYSTEM_PROMPT_PROFILES, ("sinhala_qa", "sinhala_system_grounded"))
        for name in SYSTEM_PROMPT_PROFILES:
            self.assertEqual(
                COMPARISON_PROFILES[name],
                GenerationConfig(prompt_variant=name),
            )

    def test_system_variant_reaches_model_and_only_completion_is_decoded(self):
        from contextlib import nullcontext

        question = DEMO_QUESTIONS[0][1]
        source = {"chunk_id": "fixture", "heading": "සිතුවම්", "source_quote": "කාන්තා රූප."}
        generator = SinLlamaGenerator(config=COMPARISON_PROFILES["sinhala_system_grounded"])
        tokens = Mock(shape=(1, 3))
        tokens.to.return_value = tokens
        generator._tokenizer = Mock(return_value={"input_ids": tokens}, pad_token_id=9, eos_token_id=9)
        generator._tokenizer.encode.return_value = [6, 7]
        generator._tokenizer.decode.return_value = source["source_quote"]
        generator._model = Mock(generation_config=SimpleNamespace(eos_token_id=9))
        generator._model.generate.return_value = [[1, 2, 3, 6, 7, 9]]
        torch = SimpleNamespace(inference_mode=nullcontext)
        transformer = SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list)
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformer}):
            self.assertEqual(generator.propose(question, [source]), source["source_quote"])
        generator._tokenizer.assert_called_once_with(
            build_prompt(question, [source], variant="sinhala_system_grounded"), return_tensors="pt",
        )
        self.assertIs(generator._model.generate.call_args.kwargs["input_ids"], tokens)
        generator._tokenizer.decode.assert_called_once_with(
            [6, 7, 9], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        self.assertEqual(generator.last_generation["stop_reason"], "eos")

    def test_display_keeps_verification_reason_separate(self):
        row = {
            "question": DEMO_QUESTIONS[0][1], "raw_model_output": "කාන්තා රූප.",
            "final_answer": "කාන්තා රූප.", "generation_status": "GENERATED_EXPERIMENTALLY_VERIFIED",
            "answer_origin": "verified_recovered_prefix", "verification_reason": "complete_supported_prefix",
            "sources": [{"chunk_id": "fixture"}],
        }
        display = format_question_result(1, row)
        self.assertTrue(display.startswith(f"QUESTION 01:\n{row['question']}\n"))
        self.assertIn("STATUS:\nRecovered prefix\n\nVERIFICATION REASON:\ncomplete_supported_prefix", display)
        self.assertIn("SOURCE CHUNKS:\nfixture", display)

    def test_colab_code_cells_compile(self):
        path = Path(PROJECT_DIR, "notebooks", "sinllama_colab", "Sigiriya_SinLlama_Focused_Comparison_v3.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and not source.startswith("%"):
                compile(source, str(path), "exec")

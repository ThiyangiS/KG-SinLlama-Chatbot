import io
import json
import runpy
import sys
import unittest
from contextlib import nullcontext, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import evidence
from src.chatbot import SigiriyaChatbot
from src.config import HEADING_INDEX, TEXT_INDEX
from src.embeddings import E5Embeddings
from src.evidence import count_evidence_supported, refine_result
from src.generation import apply_generation, strict_quote_verification
from src.prompting import build_prompt
from src.retrieval import SigiriyaRetriever, rank_chunks, reciprocal_rank_fusion


class FakeSession:
    def __init__(self):
        self.calls = []
        self.quote = "මෙම සිතුවම්වල කාන්තා රූප නිරූපණය වේ."

    def run(self, query, **params):
        self.calls.append((query, params))
        if "collect(DISTINCT a.name)" in query:
            return [{
                "chunk_id": "frescoes", "heading": "සීගිරි බිතුසිතුවම්",
                "evidence": self.quote, "attractions": [],
            }]
        if "coalesce(e.aliases" in query:
            return [{"id": "art", "aliases": ["බිතුසිතුවම්"]}]
        if "db.index.vector.queryNodes" in query:
            return [{"chunk_id": "frescoes"}]
        if "LINKS_TO_ENTITY" in query:
            return []
        if "relationships(p)" in query:
            return [{
                "seed": "art", "entity_ids": ["art", "figures", "site"],
                "names": ["Art", "Figures", "Site"],
                "edges": [
                    {"fact_id": "f1", "chunk_id": "frescoes", "quote": self.quote},
                    {"fact_id": "f2", "chunk_id": "frescoes", "quote": self.quote},
                ],
            }]
        raise AssertionError(query)


class PipelineTests(unittest.TestCase):
    def test_module_cli_calls_refactored_evidence_selector(self):
        question = "සීගිරි බිතුසිතුවම් ගැන කියන්න."
        session = FakeSession()
        model = Mock()
        model.encode.return_value.tolist.return_value = [0.5]
        transformer = Mock(return_value=model)
        output = io.StringIO()
        # Run the module as __main__, substituting only the external boundaries.
        with (
            patch.dict(sys.modules, {
                "sentence_transformers": SimpleNamespace(SentenceTransformer=transformer)
            }),
            patch.object(sys, "argv", ["src.chatbot", question, "--debug"]),
            patch("src.graph.connect", return_value=nullcontext(session)),
            patch.object(evidence, "grounded_answer", wraps=evidence.grounded_answer) as grounded,
            patch.object(evidence, "extract_answer", wraps=evidence.extract_answer) as extract,
            patch("src.generation.SinLlamaGenerator") as generator,
            patch("src.generation.apply_generation") as generate,
            redirect_stdout(output),
        ):
            # runpy expects the target absent from the module cache, as in python -m.
            sys.modules.pop("src.chatbot", None)
            runpy.run_module("src.chatbot", run_name="__main__", alter_sys=True)

        grounded.assert_called_once()
        self.assertEqual(grounded.call_args.args[0], question)
        self.assertEqual(grounded.call_args.kwargs, {"debug": True})
        extract.assert_called_once_with(*grounded.call_args.args, debug=True)
        model.encode.assert_called_once_with(f"query: {question}", normalize_embeddings=True)
        generator.assert_not_called()
        generate.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "SOURCE_GROUNDED_EXTRACTIVE")
        self.assertEqual(result["sources"][0]["source_quote"], session.quote)
        self.assertIn("facet_routing", result["debug"])

    def test_e5_prefixes_and_normalization(self):
        model = Mock()
        model.encode.return_value.tolist.return_value = [0.5]
        embeddings = E5Embeddings(model=model)
        self.assertEqual(embeddings.encode_query("question"), [0.5])
        model.encode.assert_called_once_with("query: question", normalize_embeddings=True)
        model.encode.reset_mock()
        embeddings.encode_passages(["document", "heading"])
        model.encode.assert_called_once_with(
            ["passage: document", "passage: heading"], normalize_embeddings=True
        )

    def test_rrf_and_protected_hybrid_ranking(self):
        self.assertEqual(reciprocal_rank_fusion([["b", "a"], ["a", "b"]]), ["a", "b"])
        paths = [{"score": 9, "edges": [{"chunk_id": "d"}]}]
        top, ranked = rank_chunks(["a", "b", "c"], ["d", "b", "c"], paths)
        self.assertEqual(top, ["a", "b", "c"])
        self.assertEqual(ranked, ["a", "b", "c", "d"])

    def test_chatbot_uses_read_queries_and_returns_auditable_evidence(self):
        session = FakeSession()
        embeddings = Mock()
        embeddings.encode_query.return_value = [0.5]
        retriever = SigiriyaRetriever(session, embeddings)
        result = SigiriyaChatbot(retriever).answer("බිතුසිතුවම්වල දැකිය හැකි රූප මොනවාද?")
        self.assertEqual(result["status"], "SOURCE_GROUNDED_EXTRACTIVE")
        self.assertEqual(result["answer"], session.quote)
        self.assertEqual(result["sources"][0]["chunk_id"], "frescoes")
        self.assertEqual(len(result["retrieval"]["graph_paths"][0]["edges"]), 2)
        self.assertEqual(result["retrieval"]["hybrid"], ["frescoes"])
        vector_calls = [params for query, params in session.calls if "queryNodes" in query]
        self.assertEqual([p["name"] for p in vector_calls], [TEXT_INDEX, HEADING_INDEX])
        self.assertTrue(all(p["k"] == 22 for p in vector_calls))
        self.assertEqual(embeddings.encode_query.call_count, 1)

    def test_multi_query_keeps_topic_rankings(self):
        embeddings = Mock()
        embeddings.encode_query.return_value = [0.5]
        retriever = SigiriyaRetriever(FakeSession(), embeddings)
        result = retriever.retrieve("බිතුසිතුවම් සහ කැටපත් පවුර ගැන කියන්න")
        self.assertEqual(len(result.subquestions), 2)
        self.assertEqual(result.topic_rankings, [["frescoes"], ["frescoes"]])
        self.assertEqual(result.multi_query, ["frescoes"])
        self.assertEqual(embeddings.encode_query.call_count, 3)

    def test_live_data_gate_skips_retrieval_and_generation(self):
        retriever, generator = Mock(), Mock()
        result = SigiriyaChatbot(retriever, generator).answer("What is the ticket price today?")
        self.assertEqual(result["status"], "LIVE_DATA_UNSUPPORTED")
        retriever.retrieve.assert_not_called()
        generator.propose.assert_not_called()

    def test_exact_counts_and_refinement_guard(self):
        question = "How many exhibits are there?"
        self.assertFalse(count_evidence_supported(question, "The museum opens at 8.30."))
        self.assertTrue(count_evidence_supported(question, "The total is 24 exhibits."))
        question = "බිතුසිතුවම්වල දැකිය හැකි රූප මොනවාද?"
        baseline = {"status": "INSUFFICIENT_DOCUMENT_EVIDENCE", "sources": []}
        row = FakeSession().run("collect(DISTINCT a.name)")[0]
        sources = {"frescoes": row}
        self.assertIs(refine_result(question, baseline, sources, ["another"]), baseline)
        refined = refine_result(question, baseline, sources, ["frescoes"])
        self.assertEqual(refined["sources"][0]["source_quote"], row["evidence"])

    def test_sinhala_prompt_contains_the_question_and_source(self):
        question = "සීගිරිය ගැන කියන්න."
        row = {"chunk_id": "art", "heading": "සිතුවම්", "source_quote": FakeSession().quote}
        prompt = build_prompt(question, [row])
        self.assertIn(f"Question: {question}\nAnswer:", prompt)
        self.assertIn("[art] සිතුවම්", prompt)
        self.assertIn(row["source_quote"], prompt)

    def test_generation_accepts_verbatim_sentences_and_falls_back(self):
        source = {"source_quote": "One sentence. […] Another sentence."}
        self.assertTrue(strict_quote_verification("Another sentence.", [source])[0])
        self.assertFalse(strict_quote_verification("Source: Another sentence.", [source])[0])
        self.assertFalse(strict_quote_verification("මෙම වාක්‍යය වෙනස් කළා.", [source])[0])
        baseline = {"status": "SOURCE_GROUNDED_EXTRACTIVE", "answer": "One sentence.",
                    "sources": [source]}
        generator = SimpleNamespace(propose=lambda *_: "A novel claim.")
        result = apply_generation("question", baseline, generator)
        self.assertEqual(result["answer"], baseline["answer"])
        self.assertEqual(result["generation_status"], "EXTRACTIVE_FALLBACK_NOT_VERIFIED")


if __name__ == "__main__":
    unittest.main()

"""Console and Python entry points for the Sigiriya chatbot."""

import argparse
import json
from dataclasses import asdict

from .embeddings import E5Embeddings
from .evidence import grounded_answer, quality_gate
from .generation import SinLlamaGenerator, apply_generation
from .graph import connect
from .prompting import NO_LIVE_SI
from .retrieval import SigiriyaRetriever


class SigiriyaChatbot:
    """Combine independently usable retrieval and generation components."""

    def __init__(self, retriever, generator=None):
        self.retriever = retriever
        self.generator = generator

    def answer(self, question, *, debug=False):
        """Return a Sinhala answer, citations and retrieval metadata."""
        question = question.strip()
        if not question:
            raise ValueError("Enter a question.")
        gate = quality_gate(question)
        if gate["rejected"]:
            return {
                "question": question,
                "answer": NO_LIVE_SI,
                "status": "LIVE_DATA_UNSUPPORTED",
                "reason": gate["reason"],
                "sources": [],
                "subquestions": [],
                "retrieval": None,
            }

        retrieval = self.retriever.retrieve(question)
        result = grounded_answer(question, self.retriever.sources, retrieval, debug=debug)
        if self.generator is not None:
            result = apply_generation(question, result, self.generator)
        return {**result, "retrieval": asdict(retrieval)}


def main():
    parser = argparse.ArgumentParser(description="Ask the Sigiriya GraphRAG chatbot.")
    parser.add_argument("question")
    parser.add_argument("--sinllama", action="store_true", help="Use cached SinLlama on CUDA.")
    parser.add_argument("--debug", action="store_true", help="Include answer-selection diagnostics.")
    args = parser.parse_args()

    with connect() as session:
        retriever = SigiriyaRetriever(session, E5Embeddings())
        generator = SinLlamaGenerator() if args.sinllama else None
        result = SigiriyaChatbot(retriever, generator).answer(args.question, debug=args.debug)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Compare SinLlama formatting and topic isolation on the frozen evidence packet."""

import argparse
import re
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.sinllama_diagnostic_v1 import EVIDENCE, evaluate, load_packet
from scripts.sinllama_handoff import RUNTIME_FILES, write_json
from scripts.sinllama_legacy_rag_comparison_v1 import (
    SETTINGS as V1_SETTINGS,
    V4,
    ComparisonGenerator,
)
from src.config import PROJECT_DIR, SINLLAMA_MODEL
from src.generation import clean_experimental_completion


NAME = "sinllama_legacy_rag_comparison_v2"
OUTPUT = PROJECT_DIR / "results" / NAME
BUNDLE = "sigiriya_sinllama_legacy_rag_runtime_v2.zip"
PROFILES = {
    "A": V1_SETTINGS["A"],
    "B": V1_SETTINGS["B"],
    "C": {**V1_SETTINGS["B"], "cleanup": "terminal_markers_and_selected_metadata",
          "completion_from": "B"},
    "D": {**V1_SETTINGS["B"], "cleanup": "terminal_markers_and_selected_metadata",
          "approach": "decomposed_legacy_rag"},
}
_BRACKETED_ID = re.compile(r"\[[\w.-]+\]")
_QUESTION_LABEL = re.compile(r"(?:^|\n)\s*(?:question|ප්‍රශ්නය|ප්රශ්නය)\s*[:：]", re.I)


def question_echo(question, completion):
    question = " ".join(question.casefold().split()).rstrip(".?!？")
    output = " ".join(completion.casefold().split())
    return bool(question and question in output) or bool(_QUESTION_LABEL.search(completion))


def assess(row, generation, *, cleanup=False):
    raw = generation["raw_model_output"]
    candidate = generation["candidate"]
    if cleanup:
        candidate = clean_experimental_completion(raw or "", row["sources"])
    echo = question_echo(row["question"], raw or "")
    # Prevent the experimental verifier's prefix recovery from hiding malformed
    # source metadata or an echoed question that survives generation.
    rejection = generation.get("generation_error")
    if cleanup and not rejection:
        if _BRACKETED_ID.search(candidate):
            rejection = "unresolved_source_identifier"
        elif echo:
            rejection = "question_echo"
    decisions = evaluate(row, candidate)
    for decision in decisions.values():
        if rejection:
            decision.update(accepted=False, reason=rejection, verified_answer=None,
                            fallback_used=True, final_answer=row["extractive_fallback"])
            if "origin" in decision:
                decision.update(answer=None, origin="extractive_fallback")
        decision["accepted"] = decision["verified_answer"] is not None
        decision["fallback_answer"] = (
            row["extractive_fallback"] if decision["fallback_used"] else None
        )
    return {
        "question": row["question"],
        "selected_evidence": row["sources"],
        "extractive_fallback": row["extractive_fallback"],
        **generation,
        "cleaned_model_answer": candidate,
        "question_echo_detected": echo,
        **decisions,
        "final_answer": decisions["strict"]["final_answer"],
    }


def topic_rows(row):
    """Recover the ordered topic-to-excerpt mapping from the existing export."""
    subquestions = row["subquestions"]
    if len(subquestions) <= 1:
        return []
    sources = row["sources"]
    # The evidence selector exports complete topics in facet order, once each.
    # Confirm the original fallback has that same heading/quote layout.
    headings = [re.sub(r"^\s*\d+\.\s*", "", source["heading"]).strip() for source in sources]
    sections = [f"{heading}: {source['source_quote']}" for heading, source in zip(headings, sources)]
    if (len(sources) != len(subquestions) or not all(headings)
            or "\n".join(sections) != row["extractive_fallback"]):
        raise ValueError("Decomposition requires a complete ordered topic mapping in the evidence export.")
    return [
        {"question": question, "heading": heading, "sources": [source],
         "extractive_fallback": source["source_quote"]}
        for question, heading, source in zip(subquestions, headings, sources)
    ]


def compare(rows, generate):
    for question_id, row in rows.items():
        topics = topic_rows(row)
        baseline = generate("A", row)
        legacy = generate("B", row)
        for arm, generation in (("A", baseline), ("B", legacy), ("C", legacy)):
            yield {"question_id": question_id, "arm": arm, "profile": PROFILES[arm]["prompt_variant"],
                   **assess(row, generation, cleanup=arm == "C")}
        if topics:
            outputs = [
                {"heading": topic["heading"], **assess(topic, generate("B", topic), cleanup=True)}
                for topic in topics
            ]
            combined = {
                mode: "\n".join(f"{item['heading']}: {item[mode]['final_answer']}" for item in outputs)
                for mode in ("strict", "experimental")
            }
            yield {
                "question_id": question_id, "arm": "D", "question": row["question"],
                "profile": PROFILES["D"]["approach"],
                "selected_evidence": row["sources"], "subquestions": outputs,
                "combined_final_answers": combined, "final_answer": combined["strict"],
            }


def answer_counts(results):
    return {
        arm: {
            mode: {
                "accepted_model_outputs": sum(item[mode]["accepted"] for item in units),
                "extractive_fallbacks": sum(item[mode]["fallback_used"] for item in units),
            }
            for mode in ("strict", "experimental")
        }
        for arm in PROFILES
        if (units := [item for row in results if row["arm"] == arm
                      for item in row.get("subquestions", [row])])
    }


def offline_results(rows):
    """Use constructed completions to exercise formatting and mixed topic outcomes."""
    def fixture(arm, row):
        exact = " ".join(source["source_quote"] for source in row["sources"])
        if row.get("heading"):
            raw = (row["question"] if row["sources"][0]["chunk_id"] == "sigiriya_005"
                   else exact + " <end answer>")
        elif row["id"] == "mechanism":
            raw = f"{exact} [sigiriya_007] <end answer>"
        elif row["id"] == "multi_topic":
            raw = row["question"]
        else:
            raw = exact + " <endanswer>"
        return {"raw_model_output": raw, "candidate": raw}

    return list(compare(rows, fixture))


def write_bundle(path):
    runtime = (*RUNTIME_FILES, "scripts/sinllama_diagnostic_v1.py",
               "scripts/sinllama_legacy_rag_comparison_v1.py", f"scripts/{NAME}.py")
    with ZipFile(path, "x", compression=ZIP_DEFLATED) as package:
        for name in runtime:
            package.write(PROJECT_DIR / name, name)
        package.write(EVIDENCE, EVIDENCE.relative_to(PROJECT_DIR).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bundle", "offline", "gpu"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "bundle":
        destination = args.output or OUTPUT
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / BUNDLE
        write_bundle(path)
        print(path)
        return

    rows, digest = load_packet(EVIDENCE)
    destination = args.output or OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{NAME}_{args.mode}.json"
    if path.exists():
        parser.error(f"Output already exists: {path}. Use --output with a new directory.")
    report = {
        "schema_version": 2, "mode": args.mode, "evidence_sha256": digest,
        "checkpoint": SINLLAMA_MODEL, "profiles": PROFILES, "final_answer_policy": "strict",
        "results": [],
    }
    if args.mode == "offline":
        report["completion_source"] = "constructed fixtures; no model inference"
        report["results"] = offline_results(rows)
    else:
        import torch

        if not torch.cuda.is_available():
            parser.error("GPU mode requires a CUDA runtime.")
        generator = ComparisonGenerator(config=V4, local_files_only=False)
        generator._load(torch)
        report["checkpoint_revision"] = getattr(generator._model.config, "_commit_hash", None)
        report["runtime"] = {
            "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
            "packages": {name: version(name) for name in ("torch", "transformers", "accelerate", "bitsandbytes")},
        }

        def generate(arm, row):
            question, sources = row["question"], row["sources"]
            try:
                candidate = generator.generate_experiment(arm, question, sources)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                return {"raw_model_output": None, "candidate": "",
                        "generation_error": f"{type(exc).__name__}: {exc}"}
            return {**generator.last_generation, "candidate": candidate,
                    "prompt": generator._build_prompt(question, sources)}

        for result in compare(rows, generate):
            report["results"].append(result)
            report["answer_counts"] = answer_counts(report["results"])
            write_json(path, report)
            print(result["question_id"], result["arm"], flush=True)
    report["answer_counts"] = answer_counts(report["results"])
    write_json(path, report)
    print(path)


if __name__ == "__main__":
    main()

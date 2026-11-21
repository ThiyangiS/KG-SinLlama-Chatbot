"""Export local evidence and run the same generation verifier in Colab."""

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from zipfile import ZipFile

from src.config import PROJECT_DIR
from src.generation import (
    GenerationConfig, _quote_pieces, apply_generation, repetitive_output, verify_generated_answer,
)
from src.verification import has_repeated_claims


SCHEMA_VERSION = 1
DEMO_QUESTIONS = (
    ("factual", "සීගිරි බිතුසිතුවම්වල කාන්තා රූප ලෙස නිරූපණය කර ඇත්තේ කවුරුන්ද?"),
    ("overview", "සීගිරි බිතුසිතුවම් ගැන කියන්න."),
    ("multi_topic", "සීගිරි බිතුසිතුවම් සහ කැටපත් පවුර ගැන කියන්න."),
    ("multi_hop_route", "කොළඹ සිට දඹුල්ල හරහා සීගිරියට යන ගමන් මාර්ගය පැහැදිලි කරන්න."),
    ("mechanism", "දිය උල්පත් ක්‍රියාත්මක වන ආකාරය විස්තර කරන්න."),
    ("paraphrase", "පර්වතයේ බටහිර මුහුණතේ තිබූ චිත්‍ර ගැන කියන්න."),
    ("reason", "සීගිරිය නැරඹීමට උදෑසන පැමිණීම සුදුසු වන්නේ ඇයි?"),
    ("unsupported_person", "සීගිරි බිතුසිතුවම් ඇඳපු චිත්‍ර ශිල්පියා කවුද?"),
    ("live_data", "අද සීගිරියේ ප්‍රවේශපත්‍ර මිල කීයද?"),
)
RUNTIME_FILES = (
    "src/__init__.py", "src/config.py", "src/generation.py", "src/prompting.py", "src/verification.py",
    "scripts/sinllama_handoff.py", "src/legacy_rag_prompt.py",
)
RUNTIME_NAME = "sigiriya_sinllama_runtime_v4.zip"
ACCEPTED = "GENERATED_EXACT_SOURCE_QUOTE_REVIEW_REQUIRED"
EXPERIMENTAL_ACCEPTED = "GENERATED_EXPERIMENTALLY_VERIFIED"
SKIPPED = "GENERATION_SKIPPED"
FOCUS_QUESTIONS = (1, 3, 5)
COMPARISON_PROFILES = {
    "source_qa": GenerationConfig(),
    "sinhala_qa": replace(GenerationConfig(), prompt_variant="sinhala_qa"),
    "sinhala_system_grounded": replace(GenerationConfig(), prompt_variant="sinhala_system_grounded"),
    "short_grounding": replace(GenerationConfig(), prompt_variant="short_grounding"),
    "few_shot": replace(GenerationConfig(), prompt_variant="few_shot"),
    "source_qa_penalty_1_15": replace(GenerationConfig(), repetition_penalty=1.15),
}
SYSTEM_PROMPT_PROFILES = tuple(
    name for name in ("sinhala_qa", "sinhala_system_grounded", "system_grounded")
    if name in COMPARISON_PROFILES
)


def export_evidence(chatbot, questions):
    """Use the active local chatbot and keep only its selected citation text."""
    rows = []
    for question_id, question in questions:
        result = chatbot.answer(question)
        sources = result["sources"]
        for row in sources:
            stored = chatbot.retriever.sources[row["chunk_id"]]["evidence"]
            pieces = _quote_pieces(row["source_quote"])
            if not pieces or any(piece not in stored for piece in pieces):
                raise ValueError(f"{question_id}: selected quote is not traceable to its chunk.")
        selected = {row["chunk_id"] for row in sources}
        paths = [
            path for path in (result.get("retrieval") or {}).get("graph_paths", [])
            if path["edges"] and all(
                edge["chunk_id"] in selected and edge.get("fact_id") and edge.get("quote")
                and edge["quote"] in chatbot.retriever.sources[edge["chunk_id"]]["evidence"]
                for edge in path["edges"]
            )
        ]
        rows.append({
            "id": question_id, "question": result["question"], "status": result["status"],
            "reason": result["reason"], "subquestions": result["subquestions"],
            "extractive_fallback": result["answer"], "sources": sources,
            "graph_paths": paths,
        })
    return {"schema_version": SCHEMA_VERSION, "kind": "sigiriya_evidence", "questions": rows}


def generate_batch(
    payload, generator, *, question_number=None, question_numbers=None,
    on_result=None, verification_mode="strict",
):
    """Generate from selected excerpts only and keep the decision for every question."""
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != "sigiriya_evidence":
        raise ValueError("Expected a version 1 Sigiriya evidence export.")
    rows = list(enumerate(payload["questions"], 1))
    if question_number is not None:
        if question_numbers is not None:
            raise ValueError("Choose either one question or a set of questions.")
        question_numbers = (question_number,)
    if question_numbers is not None:
        if any(not 1 <= number <= len(rows) for number in question_numbers):
            raise ValueError(f"Choose question numbers from 1 to {len(rows)}.")
        rows = [rows[number - 1] for number in dict.fromkeys(question_numbers)]
    results = []
    for index, row in rows:
        result = {**row, "answer": row["extractive_fallback"]}
        completed = apply_generation(row["question"], result, generator, verification_mode=verification_mode)
        results.append(completed)
        if on_result is not None:
            on_result(index, completed)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sigiriya_generation_results",
        "model_id": generator.model_id,
        "generation_config": asdict(generator.config),
        "verification_mode": verification_mode,
        "results": results,
    }


def format_question_result(number, row, *, preview_limit=380):
    """Keep the final answer visible and shorten only the generated preview."""
    raw = row.get("raw_model_output")
    proposal = raw if raw is not None else row.get("model_generated_answer")
    preview = "[skipped or unavailable]" if proposal is None else " ".join(proposal.split()) or "[empty completion]"
    if len(preview) > preview_limit:
        preview = preview[:preview_limit].rstrip() + " … [full completion in JSON]"
    status = row["generation_status"]
    origin = row.get("answer_origin", "unrecorded")
    if origin == "verified_recovered_prefix":
        label = "Recovered prefix"
    elif status in {ACCEPTED, EXPERIMENTAL_ACCEPTED}:
        label = "Accepted"
    else:
        label = "Skipped" if status == SKIPPED else "Fallback"
    sources = ", ".join(dict.fromkeys(source["chunk_id"] for source in row["sources"]))
    reason = row.get("fallback_reason") or row.get("verification_reason")
    return (
        f"QUESTION {number:02d}:\n{row['question']}\n\n"
        f"SINLLAMA GENERATED ANSWER:\n{preview}\n\n"
        f"FINAL VERIFIED ANSWER:\n{row['final_answer']}\n\n"
        f"STATUS:\n{label}\n\n"
        f"VERIFICATION REASON:\n{reason}\n\n"
        f"SOURCE CHUNKS:\n{sources or '[none]'}\n"
    )


def summarize_generation(results):
    """Report verifier metrics separately from manual answer assessment."""
    counts = Counter(row["generation_status"] for row in results)
    attempted = [row for row in results if row["generation_status"] != SKIPPED]
    raw_bad = 0
    for row in attempted:
        raw = row.get("raw_model_output")
        if raw is None:
            raw = row.get("model_generated_answer")
        raw_bad += int(
            row.get("stop_reason") in {"repeated_tokens", "repeated_claims", "prompt_marker_or_question_echo"}
            or row.get("verification_reason") == "repetition_or_bad_format"
            or (raw is not None and (repetitive_output(raw) or has_repeated_claims(raw)))
        )
    rejected = sum(row["verification_reason"] == "repetition_or_bad_format" for row in attempted)
    total = len(attempted)
    accepted = counts[ACCEPTED] + counts[EXPERIMENTAL_ACCEPTED]
    return {
        "questions": len(results), "attempted": total, "accepted": accepted,
        "recovered_prefixes": sum(row.get("answer_origin") == "verified_recovered_prefix" for row in results),
        "strict_baseline_accepted_on_same_output": (
            sum(row["strict_baseline"]["accepted"] for row in results if "strict_baseline" in row)
            if any("strict_baseline" in row for row in results) else None
        ),
        "skipped": counts[SKIPPED], "fallback": total - accepted,
        "fallback_rate": (total - accepted) / total if total else None,
        "raw_repetition_or_bad_format_rate": raw_bad / total if total else None,
        "verifier_repetition_or_bad_format_rate": rejected / total if total else None,
    }


def compare_generation(
    payload, generator, *, baseline=None, on_result=None, question_numbers=None, profile_names=None,
):
    """Compare fixed profiles on one evidence packet, sharing the loaded model."""
    runs = {}
    if baseline is not None:
        if baseline["model_id"] != generator.model_id:
            raise ValueError("Baseline model differs from the configured generator.")
        historical = {row["id"]: row for row in baseline["results"]}
        selected = [
            row for i, row in enumerate(payload["questions"], 1)
            if question_numbers is None or i in question_numbers
        ]
        matched = []
        for row in selected:
            previous = historical.get(row["id"])
            if previous is None or any(
                previous[key] != row[key] for key in ("question", "sources", "extractive_fallback")
            ):
                raise ValueError("Baseline questions or selected evidence differ from the uploaded packet.")
            matched.append(previous)
        runs["historical_baseline"] = {**baseline, "results": matched, "summary": summarize_generation(matched)}
    original_config = generator.config
    try:
        for name in (COMPARISON_PROFILES if profile_names is None else profile_names):
            generator.config = COMPARISON_PROFILES[name]

            def display(index, result):
                if on_result is not None:
                    on_result(name, index, result)

            run = generate_batch(
                payload, generator, on_result=display, question_numbers=question_numbers,
                verification_mode="experimental",
            )
            runs[name] = {**run, "summary": summarize_generation(run["results"])}
    finally:
        generator.config = original_config
    return {
        "kind": "sigiriya_prompt_comparison", "schema_version": SCHEMA_VERSION,
        "runs": runs,
        "manual_review": [
            {
                "profile": profile, "id": row["id"],
                "direct_relevance": None, "factual_correctness": None,
                "multi_topic_completeness": None, "sinhala_readability": None,
                "review_possible_paraphrase": row.get("experimental_verification", {}).get(
                    "manual_review", row["verification_reason"] == "not_an_exact_selected_source_sentence"
                ),
                "notes": "",
            }
            for profile, run in runs.items() for row in run["results"]
            if row.get("model_generated_answer") is not None
        ],
    }


def replay_completions(payload, prior, *, question_numbers=FOCUS_QUESTIONS):
    """Recheck saved completions with both verifiers without GPU inference."""
    previous = {row["id"]: row for row in prior["results"]}
    results = []
    for number in question_numbers:
        evidence = payload["questions"][number - 1]
        saved = previous.get(evidence["id"])
        if saved is None:
            continue
        if any(saved[key] != evidence[key] for key in ("question", "sources", "extractive_fallback")):
            raise ValueError(f"Saved evidence differs for {evidence['id']}.")
        candidate = saved.get("model_generated_answer")
        if candidate is None:
            continue
        results.append(verify_generated_answer({**saved, "question_number": number}, candidate, "experimental"))
    return {"kind": "sigiriya_verification_replay", "schema_version": SCHEMA_VERSION, "results": results}


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_handoff(output, payload):
    """Write the UTF-8 evidence and the small runtime needed on Colab."""
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "sigiriya_sinllama_evidence.json", payload)
    write_runtime(output)


def write_runtime(output):
    """Repackage generation without opening a database connection."""
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(output / RUNTIME_NAME, "w") as archive:
        for name in RUNTIME_FILES:
            archive.write(PROJECT_DIR / name, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results" / "generation")
    parser.add_argument("--questions", type=Path, help="Optional UTF-8 file, one question per line.")
    parser.add_argument("--runtime-only", action="store_true", help="Repackage code without Neo4j.")
    parser.add_argument("--replay", type=Path, help="Recheck Q01, Q03, Q05 from saved results without GPU.")
    parser.add_argument("--evidence", type=Path, default=PROJECT_DIR / "results/generation/sigiriya_sinllama_evidence.json")
    args = parser.parse_args()
    if args.replay:
        payload = json.loads(args.evidence.read_text(encoding="utf-8-sig"))
        prior = json.loads(args.replay.read_text(encoding="utf-8-sig"))
        replay = replay_completions(payload, prior)
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output / f"{args.replay.stem}_verification_replay_v3.json"
        write_json(destination, replay)
        for row in replay["results"]:
            print(format_question_result(row["question_number"], row))
        print(f"Wrote {destination}")
        return
    if args.runtime_only:
        write_runtime(args.output)
        print(f"Wrote {args.output / RUNTIME_NAME}")
        return
    if not os.getenv("NEO4J_PASSWORD"):
        parser.error("Set NEO4J_PASSWORD in the local shell before exporting evidence.")
    if args.questions:
        lines = args.questions.read_text(encoding="utf-8-sig").splitlines()
        questions = [(f"Q{index:02d}", q.strip()) for index, q in enumerate(lines, 1) if q.strip()]
        if not questions:
            parser.error("The questions file must contain at least one nonempty line.")
    else:
        questions = DEMO_QUESTIONS

    from src.chatbot import SigiriyaChatbot
    from src.embeddings import E5Embeddings
    from src.graph import connect
    from src.retrieval import SigiriyaRetriever

    with connect() as session:
        chatbot = SigiriyaChatbot(SigiriyaRetriever(session, E5Embeddings(device="cpu")))
        payload = export_evidence(chatbot, questions)
    write_handoff(args.output, payload)
    print(f"Exported {len(questions)} questions and the Colab runtime to {args.output}")


if __name__ == "__main__":
    main()

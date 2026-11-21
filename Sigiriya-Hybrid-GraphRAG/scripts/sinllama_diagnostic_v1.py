"""Run the isolated SinLlama generation and verification diagnostic v1."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

from scripts.sinllama_handoff import COMPARISON_PROFILES, RUNTIME_FILES, write_json
from src.config import PROJECT_DIR
from src.generation import SinLlamaGenerator, has_repeated_claims, verify_generated_answer
from src.prompting import build_prompt


VERSION = 1
QUESTIONS = {"Q01": "factual", "Q03": "multi_topic", "Q05": "mechanism"}
PROFILES = ("question_only", "sinhala_qa", "sinhala_system_grounded")
CONFIG = COMPARISON_PROFILES["sinhala_system_grounded"]
EVIDENCE = PROJECT_DIR / "results/generation/sigiriya_sinllama_evidence.json"
SCRIPT = "scripts/sinllama_diagnostic_v1.py"


def load_packet(path):
    data = path.read_bytes()
    payload = json.loads(data.decode("utf-8-sig"))
    if (payload.get("kind"), payload.get("schema_version")) != ("sigiriya_evidence", 1):
        raise ValueError("Expected the existing version 1 Sigiriya evidence export.")
    rows = {row["id"]: row for row in payload["questions"]}
    focus = {}
    for question_id, original_id in QUESTIONS.items():
        row = rows[original_id]
        if not row["sources"] or row["status"] not in {
            "SOURCE_GROUNDED_EXTRACTIVE", "SOURCE_GROUNDED_FACT_EXTRACTED",
        }:
            raise ValueError(f"{question_id} has no eligible selected evidence.")
        focus[question_id] = row
    return focus, hashlib.sha256(data).hexdigest()


def diagnostic_prompt(profile, question, sources):
    if profile == "question_only":
        return question
    return build_prompt(question, sources, variant=profile)


class DiagnosticGenerator(SinLlamaGenerator):
    def _build_prompt(self, question, sources):
        return diagnostic_prompt(self.profile, question, sources)


def evaluate(row, candidate):
    decision = verify_generated_answer(
        {"sources": row["sources"], "extractive_fallback": row["extractive_fallback"]},
        candidate, "experimental",
    )
    strict = decision["strict_baseline"]
    experimental = decision["experimental_verification"]
    return {
        "strict": {
            **strict,
            "verified_answer": candidate if strict["accepted"] else None,
            "fallback_used": not strict["accepted"],
            "final_answer": candidate if strict["accepted"] else row["extractive_fallback"],
        },
        "experimental": {
            **experimental,
            "verified_answer": experimental["answer"],
            "fallback_used": experimental["answer"] is None,
            "final_answer": decision["final_answer"],
        },
    }


def manual_cases(rows):
    factual = rows["Q01"]["sources"][0]["source_quote"]
    frescoes = rows["Q03"]["sources"][0]["source_quote"]
    mechanism = rows["Q05"]["sources"][0]["source_quote"]
    unsupported = "මෙම දිය උල්පත් විදුලි පොම්ප මඟින් ක්‍රියා කරයි."
    return [
        ("exact_supported", "Q01", factual,
         "Exact selected source answer."),
        ("short_paraphrase", "Q01", "බිතුසිතුවම්වල දැකගත හැකි කාන්තා රූප අප්සරාවන් හෝ දිව්‍ය කාන්තාවන් ලෙස අර්ථකථනය කර ඇත.",
         "Preserves the interpretation but rewrites the source sentence."),
        ("incomplete", "Q03", frescoes,
         "Covers the frescoes and omits the Mirror Wall."),
        ("unsupported_fact", "Q05", unsupported,
         "The selected source gives no support for electric pumps."),
        ("certainty_changed", "Q05", mechanism.replace("වූ බව සැලකේ.", "විය."),
         "Turns a source interpretation into a definite statement."),
        ("mixed_suffix", "Q05", f"{mechanism} {unsupported}",
         "Complete supported answer followed by an unsupported claim."),
        ("mixed_prefix", "Q05", f"{unsupported} {mechanism}",
         "Unsupported claim before the complete supported answer."),
        ("allowlisted_paraphrase", "Q05", mechanism.replace("භාවිත කරමින්", "යොදා ගනිමින්"),
         "Uses the existing phrase equivalence and preserves the other claims."),
    ]


def offline_results(rows):
    return [
        {"case": case, "question_id": qid, "evidence_id": rows[qid]["id"],
         "assessment": assessment, "candidate": candidate,
         "extractive_fallback": rows[qid]["extractive_fallback"],
         **evaluate(rows[qid], candidate)}
        for case, qid, candidate, assessment in manual_cases(rows)
    ]


def generation_findings(question, raw, candidate, stop):
    findings = []
    question = question.strip().rstrip(".?!")
    if question and question in raw:
        findings.append("question_echo")
    if stop == "prompt_marker_or_question_echo" and not findings:
        findings.append("prompt_marker")
    if stop in {"repeated_tokens", "repeated_claims"} or has_repeated_claims(raw):
        findings.append("repetition")
    if not candidate.strip():
        findings.append("empty_cleaned_output")
    if stop == "token_limit":
        findings.append("output_budget_reached")
    return findings


def run_generation(rows, generator):
    for qid, row in rows.items():
        for profile in PROFILES:
            generator.profile = profile
            question, sources = row["question"], row["sources"]
            record = {
                "question_id": qid, "evidence_id": row["id"], "profile": profile,
                "prompt": diagnostic_prompt(profile, question, sources),
                "extractive_fallback": row["extractive_fallback"],
            }
            try:
                candidate = generator.propose(question, sources)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                yield {**record, "generation_error": f"{type(exc).__name__}: {exc}",
                       "raw_model_output": None, "cleaned_output": None,
                       "generation_findings": ["inference_error"],
                       "strict": None, "experimental": None,
                       "final_answer": row["extractive_fallback"]}
                continue
            metadata = generator.last_generation
            raw = metadata["raw_model_output"]
            yield {
                **record, **metadata, "cleaned_output": candidate,
                "generation_findings": generation_findings(
                    question, raw, candidate, metadata["stop_reason"],
                ),
                **evaluate(row, candidate),
            }


def write_review_template(path):
    fields = (
        "question_id", "profile", "direct_answer", "topics_covered", "all_claims_supported",
        "uncertainty_preserved", "sinhala_readability", "generation_issue",
        "strict_reason", "experimental_reason", "final_origin", "review_notes",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid in QUESTIONS:
            for profile in PROFILES:
                writer.writerow({"question_id": qid, "profile": profile})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("offline", "bundle", "gpu"))
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/sinllama_diagnostic_v1")
    args = parser.parse_args()
    rows, digest = load_packet(args.evidence)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "bundle":
        destination = args.output / "sigiriya_sinllama_diagnostic_v1.zip"
        with ZipFile(destination, "w") as package:
            for name in (*RUNTIME_FILES, SCRIPT):
                package.write(PROJECT_DIR / name, name)
            package.write(args.evidence, EVIDENCE.relative_to(PROJECT_DIR).as_posix())
        print(destination)
        return

    settings = {
        "schema_version": VERSION, "evidence_sha256": digest, "questions": rows,
        "profiles": PROFILES,
        "generation_config": {k: v for k, v in asdict(CONFIG).items() if k != "prompt_variant"},
    }
    if args.mode == "offline":
        write_json(args.output / "verification_v1.json", {**settings, "results": offline_results(rows)})
    else:
        import torch

        if not torch.cuda.is_available():
            parser.error("GPU mode requires a CUDA runtime.")
        generator = DiagnosticGenerator(config=CONFIG, local_files_only=False)
        settings["runtime"] = {
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "packages": {name: version(name) for name in ("torch", "transformers", "bitsandbytes", "accelerate")},
            "model_id": generator.model_id,
        }
        write_json(args.output / "generation_settings_v1.json", settings)
        with (args.output / "generation_v1.jsonl").open("w", encoding="utf-8") as handle:
            for result in run_generation(rows, generator):
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(result["question_id"], result["profile"],
                      result.get("generation_error") or result["experimental"]["reason"], flush=True)
    write_review_template(args.output / "manual_review_v1.csv")
    print(args.output)


if __name__ == "__main__":
    main()

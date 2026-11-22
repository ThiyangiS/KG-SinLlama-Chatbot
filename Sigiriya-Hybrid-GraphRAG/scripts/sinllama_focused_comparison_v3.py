"""Compare focused Sinhala prompts and replay the v2 Q03 formatting failures."""

import argparse
import csv
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.sinllama_diagnostic_v1 import EVIDENCE, load_packet
from scripts.sinllama_handoff import RUNTIME_FILES, write_json
from scripts.sinllama_legacy_rag_comparison_v1 import LEGACY_V4, SETTINGS, ComparisonGenerator
from scripts.sinllama_legacy_rag_comparison_v2 import assess, topic_rows
from src.config import PROJECT_DIR, SINLLAMA_MODEL
from src.generation import _normalize_quote, _quote_pieces, clean_experimental_completion
from src.prompting import build_prompt


NAME = "sinllama_focused_comparison_v3"
OUTPUT = PROJECT_DIR / "results" / NAME
BUNDLE = "sigiriya_sinllama_focused_runtime_v3.zip"
REVISION = "6c4f36c2a4dc71cf8258e3586daa3a51447b7088"
PROFILES = {
    "A": "decomposed_legacy_rag",
    "B": "focused_sinhala_qa",
    "C": "focused_sinhala_qa with clean evidence",
    "R-Q01": "sinhala_system_grounded",
    "R-Q05": "legacy_rag_prompt",
}
LABEL = r"(?:ප්‍රශ්නය|ප්රශ්නය|තොරතුරු|පිළිතුර|Question|Context|Answer|Sources?)"
LABEL_PATTERN = re.compile(rf"(?<![\w\u0d80-\u0dff]){LABEL}[^\S\r\n]*[:：]", re.I)
TERMINAL_LABEL = re.compile(rf"(?:^|(?<=[.!?।])|(?<=\n))\s*{LABEL}[^\S\r\n]*[:：]\s*\Z", re.I)


def format_completion(raw, sources):
    """Remove a bare trailing prompt label while keeping any following content."""
    candidate = clean_experimental_completion(raw, sources)
    label = TERMINAL_LABEL.search(candidate)
    if label and not any(label[0].strip() in source["source_quote"] for source in sources):
        return candidate[:label.start()].rstrip()
    return candidate


def audit(row, generation, *, v2_cleanup=False, label_cleanup=False):
    raw = generation["raw_model_output"]
    output = raw or ""
    prepared = generation
    if label_cleanup:
        prepared = {**generation, "raw_model_output": format_completion(output, row["sources"])}
    checked = assess(row, prepared, cleanup=v2_cleanup or label_cleanup)
    candidate = checked["cleaned_model_answer"]
    question = " ".join(_normalize_quote(row["question"]).split()).rstrip(".?!？")
    echoed = bool(question and question in " ".join(_normalize_quote(output).split()))
    expected = {_normalize_quote(piece) for source in row["sources"]
                for piece in _quote_pieces(source["source_quote"])}
    actual = _quote_pieces(candidate)
    actual_normalized = {_normalize_quote(piece) for piece in actual}
    return {
        **checked, "raw_model_output": raw, "question_echo_detected": echoed,
        "prompt_labels_leaked": list(dict.fromkeys(LABEL_PATTERN.findall(output))),
        "prompt_labels_remaining": list(dict.fromkeys(LABEL_PATTERN.findall(candidate))),
        "required_source_claims": len(expected),
        "matched_exact_source_claims": len(actual_normalized & expected),
        "unsupported_claims": [piece for piece in actual if _normalize_quote(piece) not in expected],
    }


class FocusedComparisonGenerator(ComparisonGenerator):
    focused_arm = None

    def _build_prompt(self, question, sources):
        if self.focused_arm:
            return build_prompt(question, sources, variant="focused_sinhala_qa",
                                clean_evidence=self.focused_arm == "C")
        return super()._build_prompt(question, sources)

    def generate_unit(self, arm, row):
        self.focused_arm = arm if arm in {"B", "C"} else None
        question, sources = row["question"], row["sources"]
        baseline = "A" if arm == "R-Q01" else "B"
        try:
            candidate = self.generate_experiment(baseline, question, sources)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return {"raw_model_output": None, "candidate": "",
                    "generation_error": f"{type(exc).__name__}: {exc}"}
        return {**self.last_generation, "candidate": candidate,
                "prompt": self._build_prompt(question, sources)}


def run_comparison(rows, generate):
    topics = topic_rows(rows["Q03"])
    for arm in ("A", "B", "C"):
        for topic in topics:
            generation = generate(arm, topic)
            record = {"question_id": "Q03", "arm": arm, "profile": PROFILES[arm],
                      "topic_heading": topic["heading"]}
            result = {
                **record,
                **audit(topic, generation, v2_cleanup=arm == "A", label_cleanup=arm != "A"),
            }
            if arm == "A":
                # Pair B and C with the same output formatting on A's saved completion.
                result["label_cleanup_control"] = audit(topic, generation, label_cleanup=True)
            yield result
    for qid in ("Q01", "Q05"):
        arm = f"R-{qid}"
        yield {"question_id": qid, "arm": arm, "profile": PROFILES[arm], "topic_heading": "",
               **audit(rows[qid], generate(arm, rows[qid]))}


def replay_v2(path, rows, evidence_digest):
    with ZipFile(path) as archive:
        raw_report = archive.read("sinllama_legacy_rag_comparison_v2_gpu.json")
    report = json.loads(raw_report)
    if (report["mode"], report["evidence_sha256"], report["checkpoint"], report["checkpoint_revision"]) != (
        "gpu", evidence_digest, SINLLAMA_MODEL, REVISION,
    ):
        raise ValueError("v2 GPU report does not match the frozen experiment.")
    q03 = next(row for row in report["results"] if row["question_id"] == "Q03" and row["arm"] == "D")
    results = []
    for topic, previous in zip(topic_rows(rows["Q03"]), q03["subquestions"], strict=True):
        if (topic["question"], topic["sources"]) != (previous["question"], previous["selected_evidence"]):
            raise ValueError("v2 topic evidence differs from the current packet.")
        original = audit(topic, previous, v2_cleanup=True)
        assert original["cleaned_model_answer"] == previous["cleaned_model_answer"]
        results.append({"question_id": "Q03", "arm": "v2-D-replay", "profile": PROFILES["A"],
                        "topic_heading": topic["heading"], "v2_original": original,
                        **audit(topic, previous, label_cleanup=True)})
    for qid, arm in (("Q01", "A"), ("Q05", "B")):
        previous = next(row for row in report["results"] if row["question_id"] == qid and row["arm"] == arm)
        results.append({"question_id": qid, "arm": f"v2-{arm}-replay", "profile": previous["profile"],
                        "topic_heading": "", **audit(rows[qid], previous)})
    return {
        "v2_gpu_report_sha256": hashlib.sha256(raw_report).hexdigest(),
        "v2_answer_counts": report["answer_counts"], "results": results,
    }


def write_review(path, results):
    table = []
    for item in results:
        strict, experimental = item["strict"], item["experimental"]
        table.append({
            "question_id": item["question_id"], "arm": item["arm"], "topic": item["topic_heading"],
            "question": item["question"], "raw_completion": item["raw_model_output"],
            "cleaned_answer": item["cleaned_model_answer"],
            "selected_evidence": json.dumps(item["selected_evidence"], ensure_ascii=False),
            "question_echoed": item["question_echo_detected"],
            "prompt_labels_leaked": ", ".join(item["prompt_labels_leaked"]),
            "prompt_labels_remaining": ", ".join(item["prompt_labels_remaining"]),
            "required_source_claims": item["required_source_claims"],
            "matched_exact_source_claims": item["matched_exact_source_claims"],
            "matched_ordered_experimental_claims": experimental["matched_claims"],
            "unsupported_claims_exact_policy": json.dumps(item["unsupported_claims"], ensure_ascii=False),
            "strict_accepted": strict["accepted"], "strict_reason": strict["reason"],
            "experimental_accepted": experimental["accepted"], "experimental_reason": experimental["reason"],
            "strict_fallback_used": strict["fallback_used"],
            "experimental_fallback_used": experimental["fallback_used"],
            "strict_accepted_answer": strict["verified_answer"], "strict_fallback": strict["fallback_answer"],
            "experimental_accepted_answer": experimental["verified_answer"],
            "experimental_fallback": experimental["fallback_answer"],
            "human_relevance": "", "human_factual_accuracy": "", "human_coverage": "", "human_notes": "",
        })
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def write_bundle(path):
    files = (*RUNTIME_FILES, "scripts/sinllama_diagnostic_v1.py",
             "scripts/sinllama_legacy_rag_comparison_v1.py", "scripts/sinllama_legacy_rag_comparison_v2.py",
             "scripts/sinllama_reporting.py", f"scripts/{NAME}.py",
             EVIDENCE.relative_to(PROJECT_DIR).as_posix())
    with ZipFile(path, "x") as archive:
        for name in files:
            archive.writestr(ZipInfo(name), (PROJECT_DIR / name).read_bytes(),
                             compress_type=ZIP_DEFLATED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gpu", "replay", "bundle"))
    parser.add_argument("--v2-zip", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.mode == "replay" and args.v2_zip is None:
        parser.error("Replay requires --v2-zip pointing to the v2 Colab output ZIP.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "bundle":
        write_bundle(args.output / BUNDLE)
        return
    path = args.output / f"{NAME}_{args.mode}.json"
    review_path = args.output / f"{NAME}_{args.mode}_human_review.csv"
    if path.exists() or review_path.exists():
        parser.error("Output already exists. Pass --output with a new directory.")
    rows, digest = load_packet(EVIDENCE)
    report = {"schema_version": 3, "mode": args.mode, "evidence_sha256": digest,
              "checkpoint": SINLLAMA_MODEL, "checkpoint_revision": REVISION,
              "profiles": PROFILES, "final_answer_policy": "strict", "results": []}
    if args.mode == "replay":
        report.update(replay_v2(args.v2_zip, rows, digest))
    else:
        import torch

        if not torch.cuda.is_available():
            parser.error("GPU mode requires a CUDA runtime.")
        model = FocusedComparisonGenerator(config=LEGACY_V4, revision=REVISION, local_files_only=False)
        model._load(torch)
        report["generation_settings"] = {
            arm: {
                **SETTINGS["A" if arm == "R-Q01" else "B"],
                "prompt_variant": "focused_sinhala_qa" if arm in {"B", "C"} else (
                    "sinhala_system_grounded" if arm == "R-Q01" else "legacy_rag_prompt"
                ),
                "clean_prompt_evidence": arm == "C",
            }
            for arm in PROFILES
        }
        report["runtime"] = {
            "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
            "packages": {name: version(name) for name in ("torch", "transformers", "accelerate", "bitsandbytes")},
        }
        for result in run_comparison(rows, model.generate_unit):
            report["results"].append(result)
            write_json(path, report)
            print(result["question_id"], result["arm"], result["topic_heading"],
                  result["strict"]["reason"], flush=True)
    write_json(path, report)
    write_review(review_path, report["results"])
    print(path, review_path, sep="\n")


if __name__ == "__main__":
    main()

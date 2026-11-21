"""Compare the v4 Sinhala prompt with the original RAG prompt on frozen evidence."""

import argparse
import csv
import re
from copy import deepcopy
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.sinllama_diagnostic_v1 import EVIDENCE, evaluate, load_packet
from scripts.sinllama_handoff import COMPARISON_PROFILES, RUNTIME_FILES, write_json
from src.config import PROJECT_DIR
from src.generation import SinLlamaGenerator, has_repeated_claims, repetitive_output
from src.legacy_rag_prompt import END_MARKER


NAME = "sinllama_legacy_rag_comparison_v1"
BUNDLE = "sigiriya_sinllama_legacy_rag_runtime_v1.zip"
OUTPUT = PROJECT_DIR / "results" / NAME
V4 = COMPARISON_PROFILES["sinhala_system_grounded"]
LEGACY_V4 = replace(V4, prompt_variant="legacy_rag_prompt")
LEGACY_SETTINGS = {
    "prompt_variant": "legacy_rag_prompt",
    "max_input_tokens": 3000,
    "max_new_tokens": 220,
    "do_sample": False,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 0,
    "use_cache": True,
    "max_length": None,
    "eos_and_pad": "tokenizer_eos_token_id",
}
BOUNDARIES = (END_MARKER, "ප්‍රශ්නය:", "තොරතුරු:", "Question:", "Context:")
LEGACY_SPLIT = re.compile(
    r"<END_ANSWER>|(?:ප්‍රශ්නය|තොරතුරු|Question|Context)\s*[:：]", re.IGNORECASE,
)
SETTINGS = {
    "A": {**asdict(V4), "stopping": "v4"},
    "B": {**asdict(LEGACY_V4), "stopping": "v4_plus_legacy_boundaries",
          "additional_boundaries": BOUNDARIES},
    "C": {**LEGACY_SETTINGS, "stopping": "legacy_generated_only", "boundaries": BOUNDARIES},
}


def decode_completion(tokenizer, token_ids, prompt_length):
    return tokenizer.decode(
        token_ids[prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )


def legacy_boundary(raw):
    return next((marker for marker in BOUNDARIES if marker in raw), None)


def clean_legacy_answer(raw):
    return LEGACY_SPLIT.split(raw, maxsplit=1)[0].strip()


class ComparisonGenerator(SinLlamaGenerator):
    def _prompt_markers(self):
        markers = super()._prompt_markers()
        if self.config.prompt_variant == "legacy_rag_prompt":
            return tuple(dict.fromkeys((*markers, *BOUNDARIES)))
        return markers

    def generate_experiment(self, experiment, question, sources):
        self.last_generation = {}
        self.config = V4 if experiment == "A" else LEGACY_V4
        if experiment == "C":
            return self._generate_legacy(question, sources)
        candidate = self.propose(question, sources)
        self.last_generation["pad_token_id"] = self._tokenizer.pad_token_id
        if experiment == "B":
            raw = self.last_generation["raw_model_output"]
            if END_MARKER in raw and self.last_generation["stop_reason"] == "prompt_marker_or_question_echo":
                self.last_generation["stop_reason"] = "end_answer_marker"
            candidate = clean_legacy_answer(candidate)
        return candidate

    def _generate_legacy(self, question, sources):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        tokenizer, model = self._tokenizer, self._model
        inputs = tokenizer(self._build_prompt(question, sources), return_tensors="pt")
        prompt_length = inputs["input_ids"].shape[-1]
        if prompt_length > LEGACY_SETTINGS["max_input_tokens"]:
            raise ValueError("Legacy prompt exceeds the input budget with the full selected evidence.")
        device = model.get_input_embeddings().weight.device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        config = deepcopy(model.generation_config)
        config.max_length = LEGACY_SETTINGS["max_length"]
        eos = tokenizer.eos_token_id
        budget = LEGACY_SETTINGS["max_new_tokens"]

        class StopAtAnswerBoundary(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return legacy_boundary(decode_completion(tokenizer, input_ids[0], prompt_length)) is not None

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                generation_config=config,
                max_new_tokens=budget,
                do_sample=LEGACY_SETTINGS["do_sample"],
                repetition_penalty=LEGACY_SETTINGS["repetition_penalty"],
                no_repeat_ngram_size=LEGACY_SETTINGS["no_repeat_ngram_size"],
                eos_token_id=eos,
                pad_token_id=eos,
                use_cache=LEGACY_SETTINGS["use_cache"],
                stopping_criteria=StoppingCriteriaList([StopAtAnswerBoundary()]),
            )
        raw = decode_completion(tokenizer, output[0], prompt_length)
        new_tokens = output[0][prompt_length:]
        boundary = legacy_boundary(raw)
        if boundary == END_MARKER:
            reason = "end_answer_marker"
        elif boundary:
            reason = "legacy_question_or_context_boundary"
        elif len(new_tokens) and int(new_tokens[-1]) == eos:
            reason = "eos"
        else:
            reason = "token_limit" if len(new_tokens) >= budget else "other"
        self.last_generation = {
            "raw_model_output": raw,
            "stop_reason": reason,
            "matched_boundary": boundary,
            "generated_tokens": len(new_tokens),
            "prompt_tokens": prompt_length,
            "max_new_tokens_used": budget,
            "eos_token_ids": [eos],
            "pad_token_id": eos,
        }
        return clean_legacy_answer(raw)


def run_comparison(rows, generator):
    for question_id, row in rows.items():
        question, sources = row["question"], row["sources"]
        for experiment, settings in SETTINGS.items():
            candidate = generator.generate_experiment(experiment, question, sources)
            strict_and_experimental = evaluate(row, candidate)
            for decision in strict_and_experimental.values():
                decision["accepted"] = decision["verified_answer"] is not None
                decision["fallback_answer"] = (
                    row["extractive_fallback"] if decision["fallback_used"] else None
                )
            yield {
                "question_id": question_id,
                "evidence_id": row["id"],
                "question": question,
                "experiment": experiment,
                "profile": settings["prompt_variant"],
                "checkpoint": generator.model_id,
                "generation_config": settings,
                "selected_evidence": sources,
                "prompt": generator._build_prompt(question, sources),
                **generator.last_generation,
                "cleaned_model_answer": candidate,
                **strict_and_experimental,
            }


def review_rows(results):
    for row in results:
        raw = row["raw_model_output"]
        yield {
            "question": row["question_id"],
            "experiment": row["experiment"],
            "sinhala_question": row["question"],
            "cleaned_model_answer": row["cleaned_model_answer"],
            "relevant_human": "",
            "factually_correct_human": "",
            "complete_human": "",
            "repetition_detected": (
                row["stop_reason"] in {"repeated_tokens", "repeated_claims"}
                or repetitive_output(raw) or has_repeated_claims(raw)
            ),
            "strict_accepted": row["strict"]["accepted"],
            "strict_fallback": row["strict"]["fallback_used"],
            "experimental_accepted": row["experimental"]["accepted"],
            "experimental_fallback": row["experimental"]["fallback_used"],
        }


def write_bundle(path):
    runtime = (*RUNTIME_FILES, "scripts/sinllama_diagnostic_v1.py",
               "scripts/sinllama_legacy_rag_comparison_v1.py")
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as package:
        for name in runtime:
            package.write(PROJECT_DIR / name, name)
        package.write(EVIDENCE, EVIDENCE.relative_to(PROJECT_DIR).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bundle", "gpu"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "bundle":
        destination = args.output or EVIDENCE.parent
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / BUNDLE
        write_bundle(path)
        print(path)
        return

    import torch

    if not torch.cuda.is_available():
        parser.error("GPU mode requires a CUDA runtime.")
    rows, digest = load_packet(EVIDENCE)
    destination = args.output or OUTPUT
    destination.mkdir(parents=True, exist_ok=False)
    generator = ComparisonGenerator(config=V4, local_files_only=False)
    generator._load(torch)
    report = {
        "schema_version": 1,
        "evidence_sha256": digest,
        "checkpoint": generator.model_id,
        "checkpoint_revision": getattr(generator._model.config, "_commit_hash", None),
        "runtime": {
            "loader": "Transformers bitsandbytes NF4 4-bit",
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "packages": {name: version(name) for name in ("torch", "transformers", "accelerate", "bitsandbytes")},
        },
        "experiments": SETTINGS,
        "results": [],
    }
    for row in run_comparison(rows, generator):
        report["results"].append(row)
        write_json(destination / f"{NAME}.json", report)
        print(f"\n{row['question_id']} / {row['experiment']}  {row['question']}", flush=True)
        print(row["cleaned_model_answer"], flush=True)
        for mode in ("strict", "experimental"):
            result = row[mode]
            print(f"{mode}: {result['reason']} | fallback={result['fallback_used']}", flush=True)

    table = list(review_rows(report["results"]))
    with (destination / f"{NAME}_human_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(destination)


if __name__ == "__main__":
    main()

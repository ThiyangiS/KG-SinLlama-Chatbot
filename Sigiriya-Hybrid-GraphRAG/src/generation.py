"""Optional SinLlama generation and exact-quote verification."""

import re
from dataclasses import dataclass

from .config import SINLLAMA_MODEL
from .prompting import build_prompt
from .verification import has_repeated_claims, verify_experimental


def _normalize_quote(text):
    return re.sub(r'[\u200b\u200c\u200d\ufeff]', '', str(text or '')).casefold()


def repetitive_output(text):
    """Flag notebook-defined length, format and repetition failures."""
    if any(label in text for label in ("Question:", "\nප්‍රශ්නය:")):
        return True
    text = " ".join(text.split())
    if not text or len(text) > 4000:
        return True
    words = text.split()
    if len(words) > 12:
        grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
        if len(set(grams)) / len(grams) < 0.72:
            return True
    return False


def _quote_pieces(text):
    parts = re.split(r"\s*\[…]\s*|(?<=[.!?।])\s+|\n+", text)
    return [part for raw in parts if (part := raw.strip(" \n\t-•"))]


def strict_quote_verification(candidate, evidence, *, require_complete=False):
    """Check exact source membership and optionally complete, ordered coverage."""
    if repetitive_output(candidate):
        return False, "repetition_or_bad_format"
    pieces = _quote_pieces(candidate)
    if not pieces:
        return False, "empty"
    expected = list(dict.fromkeys(
        _normalize_quote(piece).strip()
        for row in evidence
        for piece in _quote_pieces(row["source_quote"])
    ))
    actual = [_normalize_quote(piece).strip() for piece in pieces]
    if set(actual) - set(expected):
        return False, "not_an_exact_selected_source_sentence"
    if require_complete and set(expected) - set(actual):
        return False, "missing_selected_evidence"
    if require_complete and actual != expected:
        return False, "reordered_or_repeated_source_sentences"
    return True, "exact_source_quotes_only"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 320
    max_input_tokens: int = 3000
    simple_max_new_tokens: int = 120
    prompt_variant: str = "source_qa"
    do_sample: bool = False
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


PROMPT_MARKERS = (
    "Question:", "Answer:", "Sources:", "Source:", "ප්‍රශ්නය:", "ප්රශ්නය:",
    "පිළිතුර:", "මූලාශ්‍ර:", "[Illustrative example", "[End example",
)


def _repeated_token_tail(tokens):
    """Detect three consecutive copies of a substantial token block."""
    return any(
        tokens[-width:] == tokens[-2 * width:-width] == tokens[-3 * width:-2 * width]
        for width in range(8, min(32, len(tokens) // 3) + 1)
    )


def _completion_candidate(raw, stops):
    """Remove a detected continuation marker while preserving the raw completion."""
    cut = min((raw.find(marker) for marker in stops if marker in raw), default=len(raw))
    return raw[:cut].strip()


class SinLlamaGenerator:
    """Load SinLlama lazily via CUDA 4-bit inference, using the cache by default."""

    def __init__(self, model_id=SINLLAMA_MODEL, config=None, *, local_files_only=True):
        self.model_id = model_id
        self.config = config or GenerationConfig()
        self.local_files_only = local_files_only
        self._load_error = None
        self._model = None
        self._tokenizer = None
        self.last_generation = {}

    def _load(self, torch):
        if not torch.cuda.is_available():
            raise RuntimeError("The 4-bit SinLlama loader requires a suitable CUDA GPU.")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=self.local_files_only)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
            use_safetensors=True,
        )
        self._tokenizer, self._model = tokenizer, model.eval()

    def _build_prompt(self, question, sources):
        return build_prompt(question, sources, variant=self.config.prompt_variant)

    def propose(self, question, sources):
        """Generate a candidate from the selected source excerpts."""
        self.last_generation = {}
        import torch

        if self._load_error is not None:
            raise self._load_error
        if self._model is None:
            try:
                self._load(torch)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                # Avoid reattempting a failed model download/load for every question.
                self._load_error = exc
                raise
        from transformers import StoppingCriteria, StoppingCriteriaList

        tokenizer, model, config = self._tokenizer, self._model, self.config
        inputs = tokenizer(
            self._build_prompt(question, sources),
            return_tensors="pt",
        )
        if inputs["input_ids"].shape[-1] > config.max_input_tokens:
            raise ValueError("Evidence prompt exceeds the input budget.")
        device = model.get_input_embeddings().weight.device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        prompt_length = inputs["input_ids"].shape[-1]
        evidence = "\n".join(row["source_quote"] for row in sources)
        evidence_tokens = len(tokenizer.encode(evidence, add_special_tokens=False))
        budget = min(config.max_new_tokens, max(config.simple_max_new_tokens, evidence_tokens + 32))
        stops = list(PROMPT_MARKERS)
        if len(question.strip()) >= 20:
            stops.extend(dict.fromkeys((question.strip(), question.strip().rstrip(".?!"))))
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
            eos_ids.append(tokenizer.eos_token_id)

        class LoopStop(StoppingCriteria):
            reason = None

            def __call__(self, input_ids, scores, **kwargs):
                length = input_ids.shape[-1] - prompt_length
                if length >= 24 and length % 4 == 0:
                    tail = input_ids[0, max(prompt_length, input_ids.shape[-1] - 96):].tolist()
                    if _repeated_token_tail(tail):
                        self.reason = "repeated_tokens"
                    elif length % 12 == 0 and has_repeated_claims(
                        tokenizer.decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    ):
                        self.reason = "repeated_claims"
                return self.reason is not None

        loop = LoopStop()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                stop_strings=stops,
                stopping_criteria=StoppingCriteriaList([loop]),
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids or None,
            )
        new_tokens = output[0][prompt_length:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if loop.reason:
            stop_reason = loop.reason
        elif any(marker in raw for marker in stops):
            stop_reason = "prompt_marker_or_question_echo"
        elif len(new_tokens) and int(new_tokens[-1]) in eos_ids:
            stop_reason = "eos"
        else:
            stop_reason = "token_limit" if len(new_tokens) >= budget else "other"
        self.last_generation = {
            "raw_model_output": raw,
            "stop_reason": stop_reason,
            "generated_tokens": len(new_tokens),
            "max_new_tokens_used": budget,
            "selected_evidence_tokens": evidence_tokens,
            "eos_token_ids": eos_ids,
        }
        return _completion_candidate(raw, stops)


def apply_generation(question, result, generator, *, verification_mode="strict"):
    """Accept complete ordered source copies or preserve the local answer."""
    if verification_mode not in {"strict", "experimental"}:
        raise ValueError("Unknown verification mode.")
    fallback = result["answer"]
    audit = {
        **result,
        "extractive_fallback": fallback,
        "model_generated_answer": None,
        "raw_model_output": None,
        "generation_status": "GENERATION_SKIPPED",
        "verification_reason": "not_run",
        "fallback_reason": result["status"],
        "final_answer": fallback,
        "answer_origin": "extractive_fallback",
    }
    if result["status"] not in (
        "SOURCE_GROUNDED_EXTRACTIVE", "SOURCE_GROUNDED_FACT_EXTRACTED"
    ) or not result["sources"]:
        return audit
    try:
        candidate = generator.propose(question, result["sources"])
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {
            **audit,
            "generation_status": "EXTRACTIVE_FALLBACK_GENERATION_ERROR",
            "fallback_reason": f"generation_error:{type(exc).__name__}",
        }
    metadata = getattr(generator, "last_generation", None)
    return verify_generated_answer(
        {**audit, **(metadata if isinstance(metadata, dict) else {})},
        candidate, verification_mode,
    )


def verify_generated_answer(audit, candidate, verification_mode):
    """Apply both policies to one completion when running the experiment."""
    passed, reason = strict_quote_verification(candidate, audit["sources"], require_complete=True)
    fallback = audit["extractive_fallback"]
    experiment = {}
    if verification_mode == "experimental":
        checked = verify_experimental(candidate, audit["sources"])
        experiment = {
            "strict_baseline": {"accepted": passed, "reason": reason},
            "experimental_verification": checked,
        }
        answer = checked["answer"] if checked["answer"] is not None else fallback
        return {
            **audit, **experiment,
            "answer": answer, "final_answer": answer, "model_generated_answer": candidate,
            "answer_origin": checked["origin"],
            "generation_status": (
                "GENERATED_EXPERIMENTALLY_VERIFIED"
                if checked["answer"] is not None else "EXTRACTIVE_FALLBACK_NOT_VERIFIED"
            ),
            "verification_reason": checked["reason"],
            "fallback_reason": None if checked["answer"] is not None else checked["reason"],
        }
    answer = candidate if passed else fallback
    return {
        **audit,
        "answer": answer,
        "final_answer": answer,
        "model_generated_answer": candidate,
        "answer_origin": "full_generated_answer" if passed else "extractive_fallback",
        "generation_status": (
            "GENERATED_EXACT_SOURCE_QUOTE_REVIEW_REQUIRED"
            if passed else "EXTRACTIVE_FALLBACK_NOT_VERIFIED"
        ),
        "verification_reason": reason,
        "fallback_reason": None if passed else reason,
    }

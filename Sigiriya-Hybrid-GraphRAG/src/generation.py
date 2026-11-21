"""Optional SinLlama generation and exact-quote verification."""

import re
from dataclasses import dataclass

from .config import SINLLAMA_MODEL
from .prompting import build_prompt


def _normalize_quote(text):
    return re.sub(r'[\u200b\u200c\u200d\ufeff]', '', str(text or '')).casefold()


def repetitive_output(text):
    """Flag notebook-defined length, format and repetition failures."""
    if any(label in text for label in ("Question:", "\nප්‍රශ්නය:")):
        return True
    text = " ".join(text.split())
    if not text or len(text) > 500:
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


def strict_quote_verification(candidate, evidence):
    """Check that each generated sentence matches a selected source sentence."""
    if repetitive_output(candidate):
        return False, "repetition_or_bad_format"
    pieces = _quote_pieces(candidate)
    if not pieces:
        return False, "empty"
    allowed = {
        _normalize_quote(piece).strip()
        for row in evidence
        for piece in _quote_pieces(row["source_quote"])
    }
    if any(_normalize_quote(piece).strip() not in allowed for piece in pieces):
        return False, "not_an_exact_selected_source_sentence"
    return True, "exact_source_quotes_only"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 110
    max_input_tokens: int = 1850
    temperature: float = 0.45
    top_p: float = 0.85
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 4


class SinLlamaGenerator:
    """Load cached SinLlama weights on demand using the CUDA 4-bit path."""

    def __init__(self, model_id=SINLLAMA_MODEL, config=None):
        self.model_id = model_id
        self.config = config or GenerationConfig()
        self._model = None
        self._tokenizer = None

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
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self._tokenizer, self._model = tokenizer, model.eval()

    def propose(self, question, sources):
        """Generate a candidate from the selected source excerpts."""
        import torch

        if self._model is None:
            self._load(torch)
        tokenizer, model, config = self._tokenizer, self._model, self.config
        inputs = tokenizer(
            build_prompt(question, sources),
            return_tensors="pt",
            truncation=True,
            max_length=config.max_input_tokens,
        )
        device = model.get_input_embeddings().weight.device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        candidate = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return re.split(
            r"\n\s*(?:Question|Source|ප්‍රශ්නය|මූලාශ්‍ර)\s*:", candidate, maxsplit=1
        )[0].strip()


def apply_generation(question, result, generator):
    """Keep the extractive answer when the proposal fails verification."""
    if result["status"] not in (
        "SOURCE_GROUNDED_EXTRACTIVE", "SOURCE_GROUNDED_FACT_EXTRACTED"
    ) or not result["sources"]:
        return result
    candidate = generator.propose(question, result["sources"])
    passed, reason = strict_quote_verification(candidate, result["sources"])
    return {
        **result,
        "answer": candidate if passed else result["answer"],
        "generation_status": (
            "GENERATED_EXACT_SOURCE_QUOTE_REVIEW_REQUIRED"
            if passed else "EXTRACTIVE_FALLBACK_NOT_VERIFIED"
        ),
        "verification_reason": reason,
    }

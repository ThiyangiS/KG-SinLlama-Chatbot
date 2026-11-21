"""Conservative, experimental Sinhala claim matching and prefix recovery."""

import re
from difflib import SequenceMatcher


# These are narrow phrase equivalences. Open-ended semantic similarity
# cannot authorize an answer or a recovered prefix.
_EQUIVALENCES = (
    ("යොදා ගනිමින්", "භාවිත කරමින්"),
    ("යැයි සැලකේ", "බව සැලකේ"),
)
_BOUNDARY = re.compile(r"[.!?।](?=\s|$|[\u0d80-\u0dff]|\d)")
_NUMBER = re.compile(r"\d+(?:[.,:/–-]\d+)*")
_NEGATION = re.compile(r"නො[\u0d80-\u0dff]+|නැත\S*|නොවේ|නොහැකි|නැති\S*|නොමැත")
_UNCERTAINTY = re.compile(r"සැලකේ|හැකිය|හැකි|පමණ|ඇතැම්|බොහෝ|විය හැක|ආසන්න")


def _claims(text):
    """Yield (claim, end offset) across sentence endings and elisions."""
    start = 0
    for match in _BOUNDARY.finditer(text):
        index = match.start()
        if match[0] == "." and (
            (index and text[index - 1].isdigit() and text[index + 1:index + 2].isdigit())
            or re.search(r"\b[A-Za-z]$", text[start:index])
        ):
            continue
        part = text[start:match.end()].strip()
        if part:
            yield part, match.end()
        start = match.end()
    tail = text[start:].strip()
    if tail:
        yield tail, len(text)


def _normalize(text):
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return " ".join(text.split())


def _canonical(text):
    normalized = _normalize(text).removesuffix(".")
    for alternative, canonical in _EQUIVALENCES:
        normalized = re.sub(
            rf"(?<![\w\u0d80-\u0dff]){re.escape(alternative)}(?![\w\u0d80-\u0dff])",
            canonical, normalized,
        )
    return normalized


def _invariants(text):
    """Explicitly audit numbers, polarity and uncertainty on each claim."""
    text = _normalize(text)
    return tuple(tuple(pattern.findall(text)) for pattern in (_NUMBER, _NEGATION, _UNCERTAINTY))


def has_repeated_claims(text):
    """Find repeated sentences even when Sinhala words or sentences touch."""
    claims = [re.sub(r"\s+", "", _normalize(claim)) for claim, _ in _claims(text)]
    for index, current in enumerate(claims):
        if len(current) < 20:
            continue
        if any(
            current == previous
            or (min(len(current), len(previous)) >= 32
                and SequenceMatcher(None, current, previous).ratio() >= 0.90)
            for previous in claims[max(0, index - 3):index]
        ):
            return True
    # A completion can loop without emitting any sentence punctuation.
    tail = "".join(claims)[-216:]
    for width in range(28, min(72, len(tail) // 3) + 1, 2):
        first, second, third = tail[-3 * width:-2 * width], tail[-2 * width:-width], tail[-width:]
        if all(SequenceMatcher(None, left, right).ratio() >= 0.90
               for left, right in ((first, second), (second, third))):
            return True
    return False


def verify_experimental(candidate, evidence):
    """Accept only complete ordered claim matches or a verified leading answer.

    Only the listed equivalent phrases may change. All other words, including
    entity names and content words, must match the source claim exactly.
    """
    expected = {}
    for row in evidence:
        for excerpt in row["source_quote"].split(" […] "):
            for claim, _ in _claims(excerpt):
                expected.setdefault(_canonical(claim), (claim, row["chunk_id"]))

    actual = list(_claims(candidate))
    matches = []
    for (claim, end), (signature, (source, chunk_id)) in zip(actual, expected.items()):
        if _canonical(claim) != signature or _invariants(claim) != _invariants(source):
            break
        matches.append({"claim": claim, "source_claim": source, "chunk_id": chunk_id})

    covered = len(matches)
    audit = {
        "claims": matches, "required_claims": len(expected), "matched_claims": covered,
        "unverified_claims": [claim for claim, _ in actual[covered:]],
    }
    if covered < len(expected) and covered < len(actual):
        source = list(expected.values())[covered][0]
        claim = actual[covered][0]
        expected_invariants, actual_invariants = _invariants(source), _invariants(claim)
        audit["first_mismatch_checks"] = {
            label: expected_value == actual_value
            for label, expected_value, actual_value in zip(
                ("numbers_preserved", "negation_preserved", "uncertainty_preserved"),
                expected_invariants, actual_invariants,
            )
        }
    if not expected or covered < len(expected):
        return {
            **audit, "answer": None, "origin": "extractive_fallback",
            "reason": "incomplete_coverage_or_unverified_claim", "manual_review": bool(candidate),
        }

    if len(actual) == covered and not has_repeated_claims(candidate):
        return {
            **audit, "answer": candidate, "origin": "full_generated_answer",
            "reason": "complete_supported_claims", "manual_review": False,
        }

    prefix_end = actual[covered - 1][1]
    prefix = candidate[:prefix_end].rstrip()
    if len(actual) > covered and prefix.endswith((".", "!", "।")) and not has_repeated_claims(prefix):
        return {
            **audit, "answer": prefix, "origin": "verified_recovered_prefix",
            "reason": "complete_supported_prefix", "manual_review": True,
            "discarded_continuation": candidate[prefix_end:],
        }
    return {
        **audit, "answer": None, "origin": "extractive_fallback",
        "reason": "no_safe_complete_prefix", "manual_review": True,
    }

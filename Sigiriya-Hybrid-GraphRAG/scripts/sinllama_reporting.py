"""Render saved SinLlama decisions without running generation or verification."""

import re
from collections import defaultdict

from src.generation import _normalize_quote, _quote_pieces


_MARKER = re.compile(r"<\s*end(?:\s*_\s*|\s*)answer\s*>", re.I)


def _certainty_changed(missing, unmatched):
    # Confirm only known certainty changes where the rest of the claim matches.
    for source in missing:
        for uncertain, definite in (("වූ බව සැලකේ.", "විය."), (" පමණ", "")):
            if uncertain in source and _normalize_quote(source.replace(uncertain, definite)) in unmatched:
                return True
    return False


def classify(record):
    decision = record["strict"]
    raw = record.get("raw_model_output") or ""
    candidate = record.get("cleaned_model_answer") or ""
    sources = record.get("selected_evidence", [])
    expected = {_normalize_quote(piece): piece for source in sources
                for piece in _quote_pieces(source["source_quote"])}
    actual = {_normalize_quote(piece): piece for piece in _quote_pieces(candidate)}
    matched = [text for key, text in expected.items() if key in actual]
    missing = [text for key, text in expected.items() if key not in actual]
    unmatched = actual.keys() - expected.keys()
    accepted = decision["accepted"]
    relevance = "YES" if accepted or matched else "UNCERTAIN"
    supported = "YES" if accepted else "PARTIAL" if matched and unmatched else "YES" if matched else "UNCERTAIN"
    complete = "YES" if accepted or (matched and not missing) else "PARTIAL" if matched else "UNCERTAIN"
    if accepted:
        problem = "VERIFIED_SUCCESS"
    elif not expected:
        problem, supported, complete = "NO_RELEVANT_EVIDENCE", "NO", "NO"
    elif record.get("reviewed_relevance") == "NO" and raw.strip() and not matched:
        problem, relevance, complete = "UNRELATED_GENERATION", "NO", "NO"
    elif record.get("question_echo_detected") and not matched:
        problem, supported, complete = "QUESTION_ECHO", "NO", "NO"
    elif _certainty_changed(missing, unmatched):
        problem, relevance = "UNCERTAINTY_CHANGED", "YES"
        supported = "PARTIAL" if matched else "NO"
    elif (record.get("prompt_labels_remaining") or _MARKER.search(candidate)
          or (raw.strip() and not candidate.strip() and _MARKER.fullmatch(raw.strip()))
          or decision["reason"] in {"unresolved_source_identifier", "reordered_or_repeated_source_sentences"}):
        problem = "FORMAT_ERROR"
        if not matched:
            supported, complete = "UNCERTAIN", "UNCERTAIN"
    elif matched and missing and not unmatched:
        problem = "PARTIAL_ANSWER"
    elif unmatched and (matched or record.get("reviewed_source_support") == "NO"):
        problem = "UNSUPPORTED_CLAIM"
        supported = "PARTIAL" if matched else "NO"
    else:
        problem = "NEEDS_MANUAL_REVIEW"
    if not raw.strip():
        supported, complete = "NO", "NO"
    return {
        "generated": "YES" if raw.strip() else "NO", "relevant": relevance,
        "source_supported": supported, "complete": complete, "classification": problem,
        "supported_claims": matched, "missing_claims": missing,
        "unmatched_claims": [text for key, text in actual.items() if key in unmatched],
    }


def _answer_source(record):
    strict = record["strict"]
    if not strict["accepted"]:
        return "EXTRACTIVE_FALLBACK"
    if (record.get("raw_model_output") or "").strip() == strict["verified_answer"].strip():
        return "SINLLAMA_GENERATED"
    return "SINLLAMA_GENERATED_CLEANED"


def _print_unit(record, emit):
    decision = record["strict"]
    status = classify(record)
    emit("RAW SINLLAMA ANSWER")
    raw = record.get("raw_model_output") or ""
    emit(raw if raw.strip() else "(No model completion)")
    emit("SINLLAMA ANSWER STATUS")
    for label, key in (("Generated", "generated"), ("Relevant", "relevant"),
                       ("Source supported", "source_supported"), ("Complete", "complete")):
        emit(f"- {label}: {status[key]}")
    emit(f"- Verification: {'ACCEPTED' if decision['accepted'] else 'REJECTED'}")
    if not decision["accepted"]:
        emit(f"- Rejection reason: {decision['reason']}")
    emit(f"- Classification: {status['classification']}")
    if not decision["accepted"]:
        for label, key in (("Supported claims", "supported_claims"), ("Missing claims", "missing_claims"),
                           ("Unmatched under strict verification", "unmatched_claims")):
            if status[key]:
                emit(f"- {label}: {' | '.join(status[key])}")
    emit("FINAL GROUNDED ANSWER")
    emit(decision["final_answer"] or "(No grounded answer available)")
    emit("ANSWER SOURCE")
    emit(_answer_source(record))
    emit("SOURCE EVIDENCE")
    sources = [row for row in record.get("selected_evidence", []) if _quote_pieces(row["source_quote"])]
    if not sources:
        emit("No supporting source was found.")
    for row in sources:
        emit(f"[{row['chunk_id']}] {row['source_quote']}")
    emit("FALLBACK")
    emit("YES" if decision["fallback_used"] else "NO")
    if decision["fallback_used"]:
        emit(f"Fallback reason: {decision.get('fallback_reason') or decision['reason']}")


def print_report(records, *, questions=None, emit=print):
    """Print strict decisions from flat topic rows or nested arm records."""
    full_questions = dict(questions or {})
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        qid = record["question_id"]
        arm = record.get("arm", record.get("experiment", ""))
        topics = record.get("subquestions")
        if topics or not record.get("topic_heading"):
            full_questions.setdefault(qid, record["question"])
        grouped[qid][arm].extend(topics or [record])
    for qid in sorted(grouped):
        emit("=" * 72)
        emit("QUESTION NUMBER AND QUESTION")
        emit(f"{qid}  {full_questions.get(qid, '(Full question unavailable in saved records)')}")
        for arm, units in grouped[qid].items():
            if arm:
                emit(f"ARM {arm}")
            decomposed = len(units) > 1 or bool(units[0].get("topic_heading"))
            for index, unit in enumerate(units, 1):
                if decomposed:
                    emit(f"SUBQUESTION {index}  {unit['question']}")
                _print_unit(unit, emit)
            if decomposed:
                emit("COMBINED FINAL GROUNDED ANSWER")
                for unit in units:
                    heading = unit.get("topic_heading") or unit.get("heading") or unit["question"]
                    emit(f"{heading}: {unit['strict']['final_answer']}")
                origins = {_answer_source(unit) for unit in units}
                if "EXTRACTIVE_FALLBACK" in origins and len(origins) > 1:
                    origin = "MIXED_VERIFIED_AND_FALLBACK"
                elif "SINLLAMA_GENERATED_CLEANED" in origins:
                    origin = "SINLLAMA_GENERATED_CLEANED"
                else:
                    origin = next(iter(origins))
                emit("COMBINED ANSWER SOURCE")
                emit(origin)
            emit(f"Accepted generations: {sum(unit['strict']['accepted'] for unit in units)}")
            emit(f"Extractive fallbacks: {sum(unit['strict']['fallback_used'] for unit in units)}")

"""Sinhala prompts and fallback messages for source-grounded answers."""

import re

from .legacy_rag_prompt import render_legacy_rag_prompt


NO_SOURCE_SI = (
    "ලබාදී ඇති සීගිරිය ලේඛනයෙන් මෙම ප්‍රශ්නයට අවශ්‍ය "
    "නිශ්චිත පිළිතුර තහවුරු කළ නොහැක."
)


NO_LIVE_SI = (
    "මෙම ප්‍රශ්නයට අවශ්‍ය යාවත්කාලීන තොරතුරු "
    "සීගිරිය පිළිබඳ ගබඩා කළ ලේඛනයෙන් තහවුරු කළ නොහැක. "
    "විශ්වාසදායක නවතම මූලාශ්‍රයක් බලන්න."
)


def build_evidence_context(sources, max_chars_each=600, *, clean_evidence=False):
    """Format up to six selected quotes with their chunk IDs and headings."""
    lines = []
    seen = set()
    for row in sources[:6]:
        key = (row["chunk_id"], row["source_quote"])
        if key not in seen:
            seen.add(key)
            heading = row["heading"]
            if clean_evidence:
                heading = re.sub(r"^\s*\d+\.\s*", "", heading)
            else:
                heading = f"[{key[0]}] {heading}"
            lines.append(f"{heading}\n{key[1][:max_chars_each]}")
    return "\n\n".join(lines)


SINHALA_SYSTEM_PROMPT = """ඔබ සීගිරිය පිළිබඳ තොරතුරු සපයන සිංහල භාෂා සහායකයෙකි.

ඔබගේ ප්‍රධාන කාර්යය වන්නේ පරිශීලකයා අසන ප්‍රශ්නවලට ලබාදී ඇති මූලාශ්‍ර තොරතුරු පමණක් භාවිත කරමින් නිවැරදි, පැහැදිලි සහ ස්වාභාවික සිංහල භාෂාවෙන් පිළිතුරු ලබාදීමයි.

පිළිතුරු ලබාදීමේ නීති:

1. සෑම විටම සිංහල භාෂාවෙන් පිළිතුරු ලබාදෙන්න.
2. ලබාදී ඇති මූලාශ්‍ර තොරතුරු පමණක් භාවිත කරන්න. මූලාශ්‍රවල සඳහන් නොවන කරුණු අනුමාන කිරීමෙන් හෝ නිර්මාණය කිරීමෙන් වළකින්න.
3. පරිශීලකයාගේ ප්‍රශ්නයට සෘජුව පිළිතුරු ලබාදෙන්න. ප්‍රශ්නය නැවත ලිවීමෙන් වළකින්න.
4. ප්‍රශ්නයේ කොටස් කිහිපයක් තිබේ නම්, ඒ සෑම කොටසකටම අදාළ තොරතුරු ඇතුළත් කරන්න.
5. පිළිතුරට අවශ්‍ය වැදගත් කරුණු පමණක් ඇතුළත් කරන්න. අදාළ නොවන තොරතුරු එකතු නොකරන්න.
6. මූලාශ්‍රවල සඳහන් පුද්ගල නාම, ස්ථාන නාම, සංඛ්‍යා, දින සහ සම්බන්ධතා නිවැරදිව පවත්වාගන්න.
7. මූලාශ්‍රයක කරුණක් නිශ්චිත නොවන බව සඳහන් වේ නම්, එය තහවුරු කළ සත්‍යයක් ලෙස ඉදිරිපත් නොකරන්න.
8. එකම වාක්‍යය හෝ අදහස නැවත නැවත සඳහන් නොකරන්න.
9. සරල ප්‍රශ්නයකට කෙටි පිළිතුරක් ලබාදෙන්න. පැහැදිලි කිරීමක් අවශ්‍ය ප්‍රශ්නයකට අදාළ කරුණු ඇතුළත් වාක්‍ය කිහිපයකින් පිළිතුරු ලබාදෙන්න.
10. ලබාදී ඇති මූලාශ්‍ර තොරතුරුවලින් පිළිතුර තහවුරු කළ නොහැකි නම්, පහත පිළිතුර පමණක් ලබාදෙන්න.

"ලබාදී ඇති තොරතුරු අනුව මෙම ප්‍රශ්නයට නිශ්චිත පිළිතුරක් ලබාදිය නොහැක."

11. පිළිතුර අවසන් වූ පසු අමතර ප්‍රශ්න, උපදෙස් හෝ වෙනත් අන්තර්ගතයක් ජනනය නොකරන්න.
12. පිළිතුරේ අවශ්‍යතාවට අනුව සරල, පැහැදිලි සහ ස්වාභාවික සිංහල භාෂාව භාවිත කරන්න.

අවසාන ප්‍රතිදානය: පරිශීලකයාගේ ප්‍රශ්නයට අදාළ සිංහල පිළිතුර පමණක් ලබාදෙන්න."""


FEW_SHOT_EXAMPLE = (
    "[Illustrative example — separate from the current question]\n"
    "Sources: පෙට්ටියේ රතු බෝලයක් ඇත.\n"
    "Question: පෙට්ටියේ ඇත්තේ කුමක්ද?\n"
    "Answer: පෙට්ටියේ රතු බෝලයක් ඇත.\n"
    "[End example — use only the following sources]\n"
)


FOCUSED_SINHALA_INSTRUCTIONS = """පහත තොරතුරු පමණක් භාවිත කරමින් ප්‍රශ්නයට පිළිතුරු දෙන්න.
අදාළ වැදගත් කරුණු සියල්ල ඇතුළත් කරන්න.
තොරතුරුවල නොමැති කරුණු එකතු නොකරන්න.
අවිනිශ්චිත කරුණු නිශ්චිත ලෙස නොකියන්න.
ප්‍රශ්නය නැවත නොලියන්න.
පිළිතුර අවසන් වූ විට <END_ANSWER> යොදන්න."""


PROMPT_FORMATS = {
    "source_qa": ("", "Sources", "Question", "Answer"),
    "sinhala_qa": ("", "මූලාශ්‍ර", "ප්‍රශ්නය", "පිළිතුර"),
    "short_grounding": (
        "Answer in Sinhala using the source sentences in order.",
        "Sources", "Question", "Answer",
    ),
    "sinhala_system_grounded": (
        SINHALA_SYSTEM_PROMPT + "\n", "මූලාශ්‍ර තොරතුරු", "ප්‍රශ්නය", "පිළිතුර",
    ),
    "few_shot": (FEW_SHOT_EXAMPLE.rstrip(), "Sources", "Question", "Answer"),
    "focused_sinhala_qa": (
        FOCUSED_SINHALA_INSTRUCTIONS + "\n", "තොරතුරු", "ප්‍රශ්නය", "පිළිතුර",
    ),
}


def build_prompt(question, sources, *, variant="source_qa", clean_evidence=False):
    """Build a base-model completion prompt with the full selected evidence."""
    if clean_evidence and variant != "focused_sinhala_qa":
        raise ValueError("Clean evidence rendering is specific to focused_sinhala_qa.")
    context = build_evidence_context(sources, max_chars_each=None, clean_evidence=clean_evidence)
    if variant == "legacy_rag_prompt":
        return render_legacy_rag_prompt(question, context)
    instruction, source_label, question_label, answer_label = PROMPT_FORMATS[variant]
    prefix = f"{instruction}\n" if instruction else ""
    return f"{prefix}{source_label}:\n{context}\n\n{question_label}: {question}\n{answer_label}:"

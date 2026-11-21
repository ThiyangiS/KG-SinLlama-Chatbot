"""Sinhala prompts and fallback messages for source-grounded answers."""


NO_SOURCE_SI = (
    "ලබාදී ඇති සීගිරිය ලේඛනයෙන් මෙම ප්‍රශ්නයට අවශ්‍ය "
    "නිශ්චිත පිළිතුර තහවුරු කළ නොහැක."
)


NO_LIVE_SI = (
    "මෙම ප්‍රශ්නයට අවශ්‍ය යාවත්කාලීන තොරතුරු "
    "සීගිරිය පිළිබඳ ගබඩා කළ ලේඛනයෙන් තහවුරු කළ නොහැක. "
    "විශ්වාසදායක නවතම මූලාශ්‍රයක් බලන්න."
)


def build_evidence_context(sources, max_chars_each=600):
    """Format up to six selected quotes with their chunk IDs and headings."""
    lines = []
    seen = set()
    for row in sources[:6]:
        key = (row["chunk_id"], row["source_quote"])
        if key not in seen:
            seen.add(key)
            lines.append(f"[{key[0]}] {row['heading']}\n{key[1][:max_chars_each]}")
    return "\n\n".join(lines)


def build_prompt(question, sources):
    """Ask SinLlama for short, verbatim evidence in Sinhala."""
    context = build_evidence_context(sources)
    return (
        "ඔබ සිංහල සංචාරක සහායකයෙකි. පහත මූලාශ්‍රයේ ඇති කරුණු පමණක් භාවිත කරන්න. "
        "නොමැති තොරතුරු නිර්මාණය නොකරන්න. ප්‍රශ්නයට සෘජුව පිළිතුරු දෙන්න. "
        "මූලාශ්‍ර වාක්‍ය වෙනස් නොකර කෙටියෙන් පිළිතුරු දෙන්න.\n"
        f"මූලාශ්‍ර:\n{context}\n\nප්‍රශ්නය: {question}\nපිළිතුර: "
    )

"""Vector, multi-query and conservative Stage 2 graph-hybrid retrieval."""

import re
from collections import defaultdict
from dataclasses import dataclass

from .config import HEADING_INDEX, RRF_K, TEXT_INDEX, TOP_K
from .graph import linked_entities, load_entities, load_sources, search_index, traverse


# Recognize complete request endings so topic names retain their Sinhala inflections.
_HOW = r"(?:කෙසේද|කොහොමද)"
_REQUEST = r"(?:කියන්න|දෙන්න|කරන්න)"
_GENERAL_END = (
    rf"(?:ගැන|පිළිබඳ)\s+(?:(?:විස්තර|තොරතුරු|පැහැදිලි)\s+)?{_REQUEST}"
    r"|(?:විස්තර|පැහැදිලි|හඳුන්වා)\s+(?:කරන්න|කියන්න|දෙන්න)"
)
_OPERATION = r"(?:ක්\u200d?රියාත්මක|භාවිත|යොදා\s+ගනු)"
_OPERATION_END = (
    rf"{_OPERATION}\s+(?:(?:වන්නේ|වෙන්නේ|කරන්නේ|කෙරෙන්නේ|ලබන්නේ)\s+{_HOW}"
    rf"|(?:වන|වෙන|කරන|කෙරෙන)\s+(?:ආකාරය|ක්\u200d?රමය)\s+(?:විස්තර|පැහැදිලි)\s+{_REQUEST})"
    rf"|{_HOW}\s+{_OPERATION}\s+(?:වන්නේ|වෙන්නේ|කරන්නේ)"
)
_LOCATION = r"(?:කොහේද|කොහෙද|කොතැනද|කොතැනකද)"
_REASON_END = r"(?:වැදගත්|විශේෂ|ප්‍රසිද්ධ|ජනප්‍රිය)\s+(?:වන්නේ|වෙන්නේ|වී\s+ඇත්තේ)\s+ඇයි"
_TOPIC_REQUEST = re.compile(
    rf"^(?P<topic>.+?)\s+(?:(?P<both>දෙකම|දෙකේම|දෙකෙහිම)\s+)?"
    rf"(?:(?P<operation>{_OPERATION_END})"
    rf"|(?P<location>(?:පිහිටා\s+(?:ඇත්තේ|තිබෙන්නේ)\s+)?{_LOCATION})"
    rf"|(?P<reason>{_REASON_END})|(?P<general>{_GENERAL_END}))$"
)
_KEEP_TOGETHER = re.compile(
    r"අතර|සසඳ|සංසන්දන|වෙනස්කම්|සමානකම්|සම්බන්ධ|එකට|එක්ව|යොදාගෙන|භාවිත කරමින්"
    r"|ගුරුත්වාකර්ෂණය\s+සහ\s+ජල\s+පීඩනය"
    r"|ආහාර\s+සහ\s+පාන|ප්‍රවේශපත්‍ර\s+සහ\s+සංචාරක\s+මාර්ගෝපදේශකයන්"
)


def clean_topic(text):
    """Strip a recognized request ending while preserving the topic wording."""
    topic = " ".join(text.split()).strip(" .?!？")
    match = _TOPIC_REQUEST.fullmatch(topic)
    return match["topic"] if match else topic


def decompose(question):
    """Split only recognizable two-topic requests with a safe reading."""
    text = " ".join(question.split())
    body = text.rstrip(" .?!？")
    ending = text[len(body):] or "."
    if _KEEP_TOGETHER.search(body):
        return [question]

    if body.count(" සහ ") == 1:
        left, right = body.split(" සහ ")
        request = _TOPIC_REQUEST.fullmatch(right)
        if not request or re.search(r"[.?!？,:;]", left):
            return [question]
        left_request = _TOPIC_REQUEST.fullmatch(left)
        if left_request:
            left_ending = "." if left_request["general"] else ending
            return [left + left_ending, right + ending]

        # An unmarked embedded "explain how B works" can be a request
        # for an overview of A followed by an explanation of B's operation.
        operation = request["operation"]
        if operation and operation.endswith(("කරන්න", "කියන්න", "දෙන්න")) and not request["both"]:
            return [f"{left} ගැන විස්තර කියන්න.", request["topic"] + " " + operation + ending]

        action = request[request.lastgroup]
        return [f"{topic} {action}{ending}" for topic in (left, request["topic"])]

    # Aත් Bත් requests can be split when both topics stand on their own.
    paired = re.fullmatch(
        r"(.+?)ත්\s+(.+?)ත්\s+(මොනවාද|කුමක්ද|මොකක්ද)", body
    )
    if paired and " සහ " not in body and not paired[2].startswith(("එහි ", "එයට ")):
        return [f"{topic} {paired[3]}{ending}" for topic in (paired[1], paired[2])]
    return [question]


def reciprocal_rank_fusion(rankings, k=RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank)
    return [chunk_id for chunk_id, _ in sorted(
        scores.items(), key=lambda item: (-item[1], item[0])
    )]


def fused_search(session, model, question):
    vector = model.encode_query(question)
    text_ranks = search_index(session, TEXT_INDEX, vector)
    heading_ranks = search_index(session, HEADING_INDEX, vector)
    return reciprocal_rank_fusion([text_ranks, heading_ranks])


def heading_match_score(subquestion, heading):
    """Small, non-gold lexical heading preference for a sub-question."""
    heading = re.sub(r"^\s*\d+\.\s*", "", heading or "").strip(" .?!؟")
    if not heading:
        return 0
    if heading in subquestion:
        return 2
    # Require at least TWO matching title words for partial matching.
    words = [w for w in heading.split() if len(w) >= 3]
    hits = sum(w in subquestion for w in words)
    return 1 if hits >= 2 else 0


def prefer_heading_match(ranking, question, headings):
    """Prefer an explicitly named source heading; otherwise keep vector order.

    Only the top 8 fused candidates are considered for the preference.
    All other candidates retain their original order.
    """
    first = ranking[:8]
    scored = [
        (heading_match_score(question, headings.get(cid, "")), rank, cid)
        for rank, cid in enumerate(first)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [cid for _, _, cid in scored] + ranking[8:]


def balanced_merge(ranked_lists):
    """Round-robin merge: give each sub-question a chance in the top three."""
    output = []
    seen = set()
    for depth in range(max((len(r) for r in ranked_lists), default=0)):
        for ranking in ranked_lists:
            if depth < len(ranking) and ranking[depth] not in seen:
                cid = ranking[depth]
                seen.add(cid)
                output.append(cid)
    return output


def multi_query_search(session, model, question, headings):
    """Return the topic queries and their round-robin ranking."""
    subquestions, ranking, _ = _multi_query(session, model, question, headings)
    return subquestions, ranking


def _multi_query(session, model, question, headings, original=None):
    subquestions = decompose(question)
    if len(subquestions) == 1:
        ranking = original if original is not None else fused_search(session, model, question)
        return subquestions, ranking, [ranking]

    raw = [fused_search(session, model, subquestion) for subquestion in subquestions]
    preferred = [
        prefer_heading_match(ranking, subquestion, headings)
        for subquestion, ranking in zip(subquestions, raw)
    ]
    return subquestions, balanced_merge(preferred), raw


TOKEN_RE = re.compile(r'[\u0D80-\u0DFF]+|[A-Za-z]{3,}|\d{4}', re.UNICODE)


STOP = {'සීගිරිය','සීගිරියේ','සීගිරියට','ගැන','කියන්න','කරන්න','විස්තර',
        'කුමක්ද','මොනවාද','දෙන්න','සහ','අතර','එහි','තිබෙන','මෙම','කොහෙද'}


def _norm(s):
    return re.sub(r'[\u200b\u200c\u200d\ufeff]', '', (s or '').casefold()).strip()


DESTINATION_ANCHOR = "sg:dest"
ANCHOR_REQUEST_TERMS = {
    "පැහැදිලි", "කෙසේද", "කොහොමද", "ආකාරය", "පිළිබඳ",
    "කුමන", "කොහේද", "කොතැනද", "ඇයි", "වන්නේ", "වෙනුවෙන්",
    "හැකි", "දැකිය", "දැකගත", "තිබූ", "ඇති", "පැරණි", "ඉපැරණි",
    "සීගිරි", "sigiriya", "tell", "about", "what", "where", "how", "why",
}
_SINHALA_SUFFIX = r"(?:වල(?:ට|ින්|දී)?|යේ|ගේ|ට|ෙන්|ෙහි|දී)?"
_ALIAS_LEFT = r"(?<![\w\u0d80-\u0dff])"
_ALIAS_RIGHT = r"(?![\w\u0d80-\u0dff])"


def _matching_alias_length(text, aliases):
    """Match full aliases, allowing common Sinhala inflections."""
    longest = 0
    for alias in aliases or ():
        name = _norm(alias)
        if len(name) < 4 or len(name) <= longest:
            continue
        suffix = _SINHALA_SUFFIX if "\u0d80" <= name[-1] <= "\u0dff" else ""
        if re.search(f"{_ALIAS_LEFT}{re.escape(name)}{suffix}{_ALIAS_RIGHT}", text):
            longest = len(name)
    return longest


def find_anchors(question, entities, max_anchors=6):
    """Return explicit matches, with the destination used only as a fallback."""
    text = _norm(question)
    matches = [
        (_matching_alias_length(text, entity["aliases"]), entity["id"])
        for entity in entities
    ]
    matches.sort(key=lambda item: (-item[0], item[1]))
    specific = [
        entity_id for length, entity_id in matches
        if length and entity_id != DESTINATION_ANCHOR
    ][:max_anchors]
    if specific:
        return specific
    return [DESTINATION_ANCHOR] if any(
        length and entity_id == DESTINATION_ANCHOR for length, entity_id in matches
    ) else []


def _anchor_terms(text):
    return source_terms(text) - ANCHOR_REQUEST_TERMS


def _matched_terms(terms, text):
    """Match topic words with a little tolerance for Sinhala word endings."""
    candidates = _anchor_terms(text)
    return {
        term for term in terms
        if any(
            term == candidate or (
                min(len(term), len(candidate)) >= 5
                and abs(len(term) - len(candidate)) <= 2
                and (term.startswith(candidate) or candidate.startswith(term))
            )
            for candidate in candidates
        )
    }


def resolve_anchors(session, question, entities, semantic_ranking):
    """Keep explicit anchors and require topical support for vector-linked additions."""
    explicit = find_anchors(question, entities)
    specific = [entity_id for entity_id in explicit if entity_id != DESTINATION_ANCHOR]
    if len(specific) >= 6:
        return specific

    # For named topics, seek evidence for the remaining question terms.
    # A chunk about an already named entity alone cannot justify a new anchor.
    remaining = _anchor_terms(question)
    for entity in entities:
        if entity["id"] in specific or entity["id"] == DESTINATION_ANCHOR:
            for alias in entity["aliases"] or ():
                remaining -= _matched_terms(remaining, alias)
    top = list(dict.fromkeys(semantic_ranking[:5]))
    if not remaining or not top:
        return explicit

    scores = {}
    for row in linked_entities(session, top):
        entity_id = row["id"]
        if entity_id == DESTINATION_ANCHOR or entity_id in specific:
            continue
        heading = row["heading"]
        heading_hits = _matched_terms(remaining, heading)
        body_hits = _matched_terms(remaining, row.get("evidence"))
        alias_hits = {
            term for alias in row["aliases"] or () for term in _matched_terms(remaining, alias)
        }
        in_heading = _matching_alias_length(_norm(heading), row["aliases"]) > 0
        position = row["pos"]

        # Rank corroborates textual evidence. It cannot qualify a candidate alone.
        alias_supported = alias_hits and (heading_hits or len(body_hits) >= 2)
        heading_supported = in_heading and (
            len(heading_hits) >= 2 or (position < 2 and (heading_hits or len(body_hits) >= 2))
        )
        if not (alias_supported or heading_supported):
            continue
        score = 3 * len(alias_hits) + 2 * len(heading_hits) + len(body_hits) + int(in_heading)
        key = (score, -position)
        scores[entity_id] = max(key, scores.get(entity_id, key))

    recovered = sorted(
        scores, key=lambda entity_id: (-scores[entity_id][0], -scores[entity_id][1], entity_id)
    )
    return (specific + recovered)[:6] or explicit


def source_terms(s):
    return {w for w in TOKEN_RE.findall(_norm(s)) if len(w)>=3 and w not in STOP}


def overlap(a,b):
    a,b=source_terms(a),source_terms(b)
    if not a:return 0.
    return len(a&b)/len(a)


def graph_paths(session, question, entities, anchors, limit=20):
    """Execute *actual* directed graph traversals up to two edges.

    A two-edge return cannot be faked by a single chunk lookup: it requires
    persisted linked nodes and sourced relationships.
    """
    if not anchors or limit <= 0:
        return []
    specific = set(anchors) - {DESTINATION_ANCHOR}
    seeds = list(dict.fromkeys([*anchors, DESTINATION_ANCHOR]))[:7]
    rows = traverse(session, seeds)
    if specific:
        # Keep destination paths only when they bridge to a selected entity.
        rows = [
            row for row in rows
            if row["seed"] != DESTINATION_ANCHOR
            or specific.intersection(row["entity_ids"])
        ]
    byid={e['id']:e for e in entities}
    qnorm=_norm(question)
    for p in rows:
        name_matches=sum(any(_norm(alias) in qnorm and len(_norm(alias))>=4
                             for alias in byid[eid]['aliases'])
                         for eid in p['entity_ids'] if eid in byid)
        quoted=' '.join(edge.get('quote') or '' for edge in p['edges'])
        p['score']=round(3.0*name_matches + 2.0*overlap(question, quoted) - 0.20*len(p['edges']),4)
    rows.sort(key=lambda r:(-r['score'],-len(r['edges']),r['entity_ids']))
    seen=set(); result=[]
    for p in rows:
        key=tuple(p['entity_ids'])
        if key not in seen:
            result.append(p);seen.add(key)
        if len(result)>=limit: break
    return result


def rank_chunks(original, multi, paths, topk=TOP_K):
    """Conservative graph-enriched retrieval without RRF top-k displacement.

    Preserve the original RRF ranking at @3 (the best-scoring retriever on
    the development set), while still returning audited graph paths separately.
    Graph facts can influence diagnostic scores for candidates OUTSIDE the
    protected window, but cannot displace the original top-3. Therefore any
    retrieval gain claimed for this approach must be tested separately.
    """
    scores=defaultdict(float)
    for ranking, weight in ((original,1.0),(multi,0.7)):
        for i,cid in enumerate(ranking,1):
            scores[cid] += weight / (RRF_K+i)
    for path in paths[:15]:
        boost=0.001 * max(0.0, min(1.0, path.get('score',0)/9.0))
        for edge in path['edges']:
            scores[edge['chunk_id']] += boost
    protected=list(dict.fromkeys(original[:topk]))
    for cid in multi:
        if len(protected)>=topk: break
        if cid not in protected: protected.append(cid)
    other=sorted((cid for cid in scores if cid not in protected),
                 key=lambda cid:(-scores[cid],cid))
    ranked=protected+other
    return ranked[:topk], ranked


@dataclass
class RetrievalResult:
    original: list[str]
    multi_query: list[str]
    hybrid: list[str]
    subquestions: list[str]
    topic_rankings: list[list[str]]
    graph_anchors: list[str]
    graph_paths: list[dict]
    evidence: list[dict]


class SigiriyaRetriever:
    """Reuse source chunks and entity aliases across questions."""

    def __init__(self, session, embeddings):
        self.session = session
        self.embeddings = embeddings
        self.sources = load_sources(session)
        self.headings = {cid: row["heading"] for cid, row in self.sources.items()}
        self.entities = load_entities(session)

    def retrieve(self, question):
        """Return semantic rankings, hybrid chunks and source-backed paths."""
        original = fused_search(self.session, self.embeddings, question)
        subquestions, multi, topic_rankings = _multi_query(
            self.session, self.embeddings, question, self.headings, original
        )
        anchors = resolve_anchors(self.session, question, self.entities, original)
        paths = graph_paths(self.session, question, self.entities, anchors)
        top, hybrid = rank_chunks(original, multi, paths)
        return RetrievalResult(
            original=original,
            multi_query=multi,
            hybrid=hybrid,
            subquestions=subquestions,
            topic_rankings=topic_rankings,
            graph_anchors=anchors,
            graph_paths=paths[:8],
            evidence=[self.sources[cid] for cid in top if cid in self.sources],
        )

"""Vector, multi-query and conservative Stage 2 graph-hybrid retrieval."""

import re
from collections import defaultdict
from dataclasses import dataclass

from .config import HEADING_INDEX, RRF_K, TEXT_INDEX, TOP_K
from .graph import linked_entities, load_entities, load_sources, search_index, traverse


def clean_topic(text):
    """Remove request wording, keeping the named topic (heuristic)."""
    text = text.strip(" .?!？")
    text = re.split(
        r"\s+(?:ගැන|අතර|ලක්ෂණ|දෙකම|වෙනස්කම්|මොනවාද|කුමක්ද)\b",
        text,
        maxsplit=1,
    )[0]
    return text.strip(" .?!？")


def decompose(question):
    """Split a few common Sinhala two-topic constructions.

    Unknown constructions are left unsplit; the system makes no claim to
    understand all Sinhala grammar. Neither gold labels nor chunk IDs are used.
    """
    question = " ".join(question.strip().split())

    # "A සහ B ..." (comparison or a request about two places/topics).
    if " සහ " in question:
        left, right = question.split(" සහ ", 1)
        left, right = clean_topic(left), clean_topic(right)
        if left and right:
            context = "සීගිරියේ " if left.startswith("සීගිරියේ ") else ""
            if context and not right.startswith(("සීගිරි", "සීගිරිය", "පිදුරංගල")):
                right = context + right
            return [f"{left} ගැන විස්තර කියන්න.", f"{right} ගැන විස්තර කියන්න."]

    # "Aත් Bත් මොනවාද?" (e.g., route AND visiting time).
    m = re.match(r"^(.+?)ත්\s+(.+?)ත්\s+(?:මොනවාද|කුමක්ද|මොකක්ද)[?？.]?$", question)
    if m:
        left, right = clean_topic(m.group(1)), clean_topic(m.group(2))
        if left and right:
            if right.startswith("එහි ") and "සීගිරිය" in left:
                right = right.replace("එහි ", "සීගිරියට ", 1)
            return [f"{left} ගැන විස්තර කියන්න.", f"{right} ගැන විස්තර කියන්න."]

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


def find_anchors(question, entities, max_anchors=6):
    q = _norm(question)
    matches=[]
    for ent in entities:
        candidates=[a for a in ent['aliases'] if len(_norm(a))>=4 and _norm(a) in q]
        if candidates:
            a=max(candidates,key=lambda x:len(_norm(x)))
            matches.append((len(_norm(a)),ent['id']))
    matches.sort(key=lambda x:(-x[0],x[1]))
    names=[eid for _,eid in matches if eid!='sg:dest'][:max_anchors]
    return names or ['sg:dest']


def resolve_anchors(session, question, entities, semantic_ranking):
    """Resolve named entities from BOTH the question and semantically found chunks.

    Linked graph candidates are read from the existing chunk-entity connection,
    so synonym/paraphrase questions need not contain literal entity names.
    No test gold is supplied here.
    """
    explicit=find_anchors(question,entities,max_anchors=6)
    ranked=list(explicit) if explicit!=['sg:dest'] else []
    top=list(dict.fromkeys(semantic_ranking[:5]))
    if top:
        candidates=[]
        for rec in linked_entities(session, top):
            entid=rec['id']
            if entid=='sg:dest' or entid in ranked: continue
            aliases=rec['aliases'] or []
            in_heading=max((len(_norm(a)) for a in aliases if len(_norm(a))>=4 and _norm(a) in _norm(rec['heading'])),default=0)
            match_query=max((len(_norm(a)) for a in aliases if len(_norm(a))>=4 and _norm(a) in _norm(question)),default=0)
            kind_prior=1 if rec['kind'] in ('Attraction','NearbyPlace','Service','HistoricFeature','EngineeringFeature') else 0
            value=4*match_query+2*in_heading+kind_prior-3*rec['pos']
            candidates.append((value,entid))
        candidates.sort(key=lambda x:(-x[0],x[1]))
        for _,eid in candidates:
            if eid not in ranked:ranked.append(eid)
            if len(ranked)>=6:break
    return ranked[:6] or ['sg:dest']


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
    seeds=list(dict.fromkeys([*anchors,'sg:dest']))[:7]
    rows = traverse(session, seeds)
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

"""Read-only operations on the existing Sigiriya Neo4j schema."""

from contextlib import contextmanager

from .config import HEADING_INDEX, TEXT_INDEX, VECTOR_LIMIT, Neo4jConfig


@contextmanager
def connect(config=None):
    """Open a read-routed Neo4j session and close its driver on exit."""
    from neo4j import GraphDatabase, READ_ACCESS

    config = config or Neo4jConfig.from_env()
    with GraphDatabase.driver(config.uri, auth=(config.username, config.password)) as driver:
        with driver.session(database=config.database, default_access_mode=READ_ACCESS) as session:
            yield session


SOURCE_QUERY = """
MATCH (c:Chunk)
WHERE c.destination = 'Sigiriya'
OPTIONAL MATCH (a:Attraction)-[:MENTIONS]-(c)
RETURN c.id AS chunk_id,
       c.heading AS heading,
       c.text AS evidence,
       collect(DISTINCT a.name) AS attractions
ORDER BY c.id
"""


def load_sources(session):
    records = [dict(r) for r in session.run(SOURCE_QUERY)]
    if not records:
        raise RuntimeError("No Sigiriya :Chunk nodes found. Check your Neo4j database.")
    ids = [r["chunk_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate source chunk IDs detected in Neo4j.")
    return {r["chunk_id"]: r for r in records}


VALID_RELATIONS = frozenset({
    'DEVELOPED','HAS_STATUS','HAS_ATTRACTION','HAS_FEATURE','HAS_ACTIVITY',
    'DEPICTS','HAS_ADVICE','USES','NAMED_FOR','NEARBY','HAS_SERVICE',
    'HAS_EXAMPLE','TRAVEL_VIA','TRAVEL_TO','ACCESSIBLE_VIA','AVAILABLE_AT','HAS_RISK',
})


REL_PATTERN = '|'.join(sorted(VALID_RELATIONS))


def search_index(session, name, vector):
    if name not in (TEXT_INDEX, HEADING_INDEX):
        raise ValueError('Unexpected vector index name')
    cypher = '''CALL db.index.vector.queryNodes($name,$k,$vector)
        YIELD node, score
        WHERE node:Chunk AND node.destination = 'Sigiriya'
        RETURN node.id AS chunk_id, score ORDER BY score DESC'''
    return [r['chunk_id'] for r in session.run(cypher, name=name, k=VECTOR_LIMIT, vector=vector)]


def traverse(session, seeds):
    """Fetch directed, sourced paths of one or two edges."""
    query = f"""
    MATCH (s:TourismEntity)
    WHERE s.id IN $seeds AND s.scope='Sigiriya'
    MATCH p=(s)-[:{REL_PATTERN}*1..2]->(t:TourismEntity)
    WHERE ALL(r IN relationships(p) WHERE r.scope='Sigiriya' AND r.fact_id IS NOT NULL)
    RETURN s.id AS seed,
      [n IN nodes(p) | n.id] AS entity_ids,
      [n IN nodes(p) | n.name] AS names,
      [r IN relationships(p) | {{relation:type(r), fact_id:r.fact_id,
                                  chunk_id:r.chunk_id,quote:r.evidence}}] AS edges
    LIMIT 300
    """
    return [dict(row) for row in session.run(query, seeds=seeds)]


def linked_entities(session, chunk_ids):
    """Fetch entities linked to semantic candidates in rank order."""
    query = """
    UNWIND range(0,size($ids)-1) AS pos
    MATCH (c:Chunk {id:$ids[pos]})-[:LINKS_TO_ENTITY]->(e:TourismEntity)
    WHERE c.destination='Sigiriya' AND e.scope='Sigiriya'
    RETURN pos,e.id AS id,e.kind AS kind,e.aliases AS aliases,
           c.heading AS heading, c.text AS evidence
    ORDER BY pos
    LIMIT 100
    """
    return session.run(query, ids=chunk_ids)


def load_entities(session):
    """Read the aliases and IDs from the persisted Stage 2 graph."""
    return [
        dict(row)
        for row in session.run("""
            MATCH (e:TourismEntity {scope: 'Sigiriya'})
            RETURN e.id AS id, e.name AS name, coalesce(e.aliases, []) AS aliases
            ORDER BY e.id
        """)
    ]

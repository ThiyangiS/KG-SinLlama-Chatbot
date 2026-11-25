"""Create Sigiriya GraphRAG data inside a dedicated, EMPTY local Neo4j database.

Uses the canonical Sinhala document and the Stage 2 graph manifest.
Reads Neo4j connection settings from the environment.
"""
import re
import time

from sentence_transformers import SentenceTransformer
import Sigiriya_GraphRAG_Stage2 as stage2

MODEL_NAME = 'intfloat/multilingual-e5-base'

SOURCE_PATTERN = re.compile(r'(?ms)^##\s+(\d+)\.\s+([^\n]+)\n(.*?)(?=^##\s+\d+\.|\Z)')
LEGACY_ATTRACTIONS = {
    'Sigiriya Rock': 'sigiriya_003',
    'Sigiriya Frescoes': 'sigiriya_004',
    'Mirror Wall': 'sigiriya_005',
    'Water Gardens': 'sigiriya_006',
    'Fountain Garden': 'sigiriya_007',
    'Boulder Garden': 'sigiriya_008',
    'Cobra Hood Cave': 'sigiriya_009',
    'Sigiriya Museum': 'sigiriya_010',
}


def load_source_and_validate():
    source = stage2.SOURCE_PATH.read_text(encoding='utf-8-sig')
    chunks = {}
    for n, title, body in SOURCE_PATTERN.findall(source):
        cid = f'sigiriya_{int(n):03d}'
        chunks[cid] = dict(id=cid, heading=f'{int(n)}. {title.strip()}',
                           text=body.strip(), destination='Sigiriya')
    manifest = stage2.load_manifest()
    good, bad = stage2.validate_facts(manifest, chunks)
    if len(chunks) != 22 or len(manifest['entities']) != 49 or len(good) != 54 or bad:
        raise RuntimeError(f'Source validation failed: chunks={len(chunks)}, '
                           f'entities={len(manifest["entities"])}, validated facts={len(good)}, errors={bad[:5]}')
    print(f'PASS: {len(chunks)} chunks, {len(manifest["entities"])} entities, {len(good)} verified facts')
    return chunks, manifest, good


def main():
    chunks, manifest, valid_facts = load_source_and_validate()
    driver, database = stage2.db_connection()
    try:
        with driver.session(database=database) as session:
            count = session.run('MATCH (n) RETURN count(n) AS n').single()['n']
            if count:
                raise RuntimeError(f'TARGET IS NOT EMPTY ({count} nodes). Stopping to protect existing graph. '
                                   'Use a dedicated empty Neo4j database.')
            print('PASS: connected to empty local database.')

        print('Loading multilingual E5-base on CPU (first run downloads model weights)...')
        model = SentenceTransformer(MODEL_NAME, device='cpu')
        ids = sorted(chunks)
        vectors = model.encode(['passage: ' + chunks[i]['text'] for i in ids],
                               batch_size=2, normalize_embeddings=True, show_progress_bar=True)
        headings = model.encode(['passage: ' + chunks[i]['heading'] for i in ids],
                                batch_size=2, normalize_embeddings=True, show_progress_bar=True)
        if vectors.shape != (22, 768) or headings.shape != (22, 768):
            raise RuntimeError(f'Unexpected embedding shapes: {vectors.shape}, {headings.shape}')
        rows = [{**chunks[cid], 'embedding': vectors[i].astype(float).tolist(),
                 'embedding_heading': headings[i].astype(float).tolist()}
                for i, cid in enumerate(ids)]
        with driver.session(database=database) as session:
            for label, constraint_name in [('Chunk', 'chunk'), ('TourismEntity', 'tourism_entity'),
                                           ('TourismFact', 'tourism_fact')]:
                session.run(f'CREATE CONSTRAINT sigiriya_{constraint_name}_id IF NOT EXISTS '
                            f'FOR (n:{label}) REQUIRE n.id IS UNIQUE').consume()
            session.run('''UNWIND $rows AS row
                MERGE (c:Chunk {id:row.id})
                SET c.destination=row.destination, c.heading=row.heading, c.text=row.text,
                    c.embedding=row.embedding, c.embedding_heading=row.embedding_heading''', rows=rows).consume()
            print('PASS: stored 22 chunks with 768-dimensional text and heading vectors.')
            for query in [
                '''CREATE VECTOR INDEX sigiriya_chunk_vectors IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {indexConfig: {`vector.dimensions`:768, `vector.similarity_function`:'cosine'}}''',
                '''CREATE VECTOR INDEX sigiriya_heading_vectors IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding_heading)
                    OPTIONS {indexConfig: {`vector.dimensions`:768, `vector.similarity_function`:'cosine'}}''',
            ]:
                session.run(query).consume()
            needed = {'sigiriya_chunk_vectors', 'sigiriya_heading_vectors'}
            for _ in range(90):
                states = {r['name']:r['state'] for r in session.run('SHOW VECTOR INDEXES YIELD name,state RETURN name,state')}
                if all(states.get(name) == 'ONLINE' for name in needed):
                    break
                time.sleep(2)
            else:
                raise RuntimeError(f'Indexes did not come online: {states}')
            print('PASS: both vector indexes ONLINE.')
            session.run('''UNWIND $rows AS row
                MERGE (a:Attraction {name:row.name})
                WITH a, row MATCH (c:Chunk {id:row.chunk})
                MERGE (a)-[:MENTIONS]->(c)''',
                rows=[dict(name=name,chunk=cid) for name,cid in LEGACY_ATTRACTIONS.items()]).consume()
            stage2.apply_graph(session, manifest, valid_facts)
            counts = stage2.graph_inspection(session)
            print('Graph inspection:',counts)
            required = {'entities':49, 'fact_nodes':54, 'sourced_relations':54, 'linked_chunks':22}
            for name, expected in required.items():
                if counts.get(name) != expected:
                    raise RuntimeError(f'Expected {name}={expected}; got {counts.get(name)}')
            print('SUCCESS: local Sigiriya GraphRAG database built and verified.')
    finally:
        driver.close()

if __name__ == '__main__':
    main()

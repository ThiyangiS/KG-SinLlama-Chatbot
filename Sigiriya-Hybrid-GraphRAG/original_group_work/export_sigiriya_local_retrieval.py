"""Export source-grounded local Neo4j retrieval results for Colab SinLlama.

Place beside the scripts in the extracted Sigiriya_Hybrid_GraphRAG_Stage2 folder.
Install: pip install neo4j sentence-transformers
Run: python export_sigiriya_local_retrieval.py
"""
import json
from pathlib import Path
from getpass import getpass

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import Sigiriya_Multi_Query_Retrieval as multi
import Sigiriya_GraphRAG_Stage2 as stage2
import Sigiriya_GraphRAG_Stage2_Answer_Refinement as refinement
from Sigiriya_Retrieval_Improvement_Batch_Reviewed import load_sources

QUESTION = 'සීගිරියේ ජල උද්‍යානවල ජල මාර්ග ගැන විස්තර කරන්න.'
URI = 'bolt://127.0.0.1:7687'
USERNAME = 'neo4j'
DATABASE = 'neo4j'

# Compatible with Neo4j versions that support db.index.vector.queryNodes.
def local_vector_search(session, name, vector):
    if name not in (multi.TEXT_INDEX, multi.HEADING_INDEX):
        raise ValueError('Unexpected vector index name')
    cypher = '''CALL db.index.vector.queryNodes($name,$k,$vector)
        YIELD node, score
        WHERE node:Chunk AND node.destination = 'Sigiriya'
        RETURN node.id AS chunk_id, score ORDER BY score DESC'''
    return [r['chunk_id'] for r in session.run(cypher, name=name, k=22, vector=vector)]

multi.search_index = local_vector_search


def main():
    password = getpass('Local Neo4j password: ')
    driver = GraphDatabase.driver(URI, auth=(USERNAME, password))
    try:
        driver.verify_connectivity()
        print('Connected to local Neo4j')
        model = SentenceTransformer(multi.MODEL_NAME, device='cpu')
        manifest = stage2.load_manifest()
        with driver.session(database=DATABASE) as s:
            headings = {r['chunk_id']: r['heading'] for r in s.run(multi.HEADINGS_QUERY)}
            sources = load_sources(s)
            print('Stored source chunks:', len(sources))
            assert len(sources) == 22, 'Expected 22 source chunks; verify your database'
            result = refinement.quality_answer(QUESTION, s, model, sources, headings)
            base = {
                'question': QUESTION,
                'status': result['status'],
                'extractive_answer': result.get('answer', ''),
                'source_quotes': result.get('sources', []),
            }
            if result['status'] == 'LIVE_DATA_UNSUPPORTED':
                payload = {**base, 'rrf_top3': [], 'multi_query_top3': [], 'hybrid_top3': [],
                           'anchors': [], 'graph_paths': [], 'retrieved_chunks': []}
            else:
                original = multi.fused_search(s, model, QUESTION)
                _, multi_rank = multi.multi_query_search(s, model, QUESTION, headings)
                anchors = refinement.resolve_anchors(s, QUESTION, manifest['entities'], original)
                paths = refinement.graph_paths(s, QUESTION, manifest['entities'], anchors, limit=12)
                hybrid, _ = refinement.rank_chunks(original, multi_rank, paths)
                chunks = [{'id': cid, 'heading': sources[cid]['heading'],
                           'text': sources[cid]['evidence']}
                          for cid in hybrid if cid in sources]
                payload = {**base, 'rrf_top3': original[:3], 'multi_query_top3': multi_rank[:3],
                           'hybrid_top3': hybrid, 'anchors': anchors,
                           'graph_paths': paths[:6], 'retrieved_chunks': chunks}
        path = Path('sigiriya_retrieval_result.json')
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        print('Saved:', path.resolve())
        print('Status:', payload['status'])
        print('Retrieved chunks:', len(payload['retrieved_chunks']))
        print('Graph paths:', len(payload['graph_paths']))
    finally:
        driver.close()


if __name__ == '__main__':
    main()

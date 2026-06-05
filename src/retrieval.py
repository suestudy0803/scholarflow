# src/retrieval.py
import math
import os
import pickle
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def load_data():
    with open(os.path.join(DATA_DIR, 'embeddings.pkl'), 'rb') as f:
        emb_data = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'papers.json'), 'r') as f:
        papers = json.load(f)
    emb_data['papers'] = papers
    return emb_data


LANDMARK_SIM_THRESHOLD = 0.45


def rank_score(sim_score, citation_count, is_landmark):
    cite_score = math.log1p(citation_count) / math.log1p(200000)
    score = 0.75 * sim_score + 0.25 * cite_score
    if is_landmark and sim_score >= LANDMARK_SIM_THRESHOLD:
        score += 0.15
    return score


def _apply_tag_boost(sim_scores, papers, query):
    """쿼리가 원조 논문의 개념 태그와 부분 문자열 매칭되면 sim_score 하한을 0.5로 보정.
    태그는 CONCEPT_ANCHORS 논문에만 존재하므로 검색 누락을 방지하는 최소 개입."""
    query_lower = query.lower().strip()
    boosted = sim_scores.copy()
    for i, p in enumerate(papers):
        for tag in p.get('tags', []):
            tag_lower = tag.lower()
            if query_lower in tag_lower or tag_lower in query_lower:
                boosted[i] = max(boosted[i], 0.50)
                break
    return boosted


def search(query, data, model, top_k=10):
    query_emb = model.encode(query, normalize_embeddings=True)
    sim_scores = data['embeddings'] @ query_emb
    sim_scores = _apply_tag_boost(sim_scores, data['papers'], query)

    top_candidates = np.argsort(sim_scores)[::-1][:50]

    results = []
    for idx in top_candidates:
        p = data['papers'][idx]
        score = rank_score(
            float(sim_scores[idx]),
            p.get('citation_count', 0),
            p.get('is_landmark', False),
        )
        results.append({
            'idx':            int(idx),
            'title':          p['title'],
            'abstract':       p['abstract'],
            'authors':        p.get('authors', []),
            'published':      p.get('published', ''),
            'url':            p.get('url', ''),
            'citation_count': p.get('citation_count', 0),
            'venue':          p.get('venue', ''),
            'is_landmark':    p.get('is_landmark', False),
            'sim_score':      float(sim_scores[idx]),
            'final_score':    score,
        })

    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results[:top_k]


def similar_papers(idx, data, top_k=5):
    query_emb = data['embeddings'][idx]
    scores = data['embeddings'] @ query_emb
    scores[idx] = -1

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in top_indices:
        p = data['papers'][i]
        results.append({
            'idx':            int(i),
            'title':          p['title'],
            'abstract':       p['abstract'],
            'authors':        p.get('authors', []),
            'published':      p.get('published', ''),
            'url':            p.get('url', ''),
            'citation_count': p.get('citation_count', 0),
            'venue':          p.get('venue', ''),
            'is_landmark':    p.get('is_landmark', False),
            'sim_score':      float(scores[i]),
        })
    return results


if __name__ == '__main__':
    from sentence_transformers import SentenceTransformer
    print("로드 중...")
    data = load_data()
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    print("\n=== 검색 테스트: transformer ===")
    results = search("transformer", data, model, top_k=5)
    for i, r in enumerate(results):
        print(f"{i+1}. [{r['final_score']:.4f}] {r['title'][:60]}")
        print(f"   sim={r['sim_score']:.3f} | 인용수: {r['citation_count']:,} | venue: {r['venue'][:30]}")

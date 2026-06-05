# src/enrich.py
import os
import json
import time
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 신뢰 학회 목록
TRUSTED_VENUES = {
    'NeurIPS', 'ICML', 'ICLR', 'ACL', 'EMNLP', 'NAACL',
    'CVPR', 'ICCV', 'ECCV', 'AAAI', 'IJCAI', 'ACM',
    'IEEE', 'Nature', 'Science', 'arXiv'
}

# 원조 논문 목록 (수동 등록)
LANDMARK_PAPERS = {
    'attention is all you need',
    'bert: pre-training of deep bidirectional transformers',
    'deep residual learning for image recognition',
    'generative adversarial nets',
    'dropout: a simple way to prevent neural networks from overfitting',
    'adam: a method for stochastic optimization',
    'language models are few-shot learners',
    'an image is worth 16x16 words',
}

def enrich_papers():
    with open(os.path.join(DATA_DIR, 'papers.json'), 'r') as f:
        papers = json.load(f)

    print(f"총 {len(papers)}개 논문 enrichment 시작...")
    enriched = []

    for i, paper in enumerate(papers):
        # arXiv ID 추출
        arxiv_id = paper['id'].split('/')[-1].split('v')[0]

        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
            params = {"fields": "citationCount,venue,publicationVenue"}
            res = requests.get(url, params=params, timeout=10)

            if res.status_code == 200:
                data = res.json()
                citation_count = data.get('citationCount', 0)
                venue = data.get('venue', '') or ''
            else:
                citation_count = 0
                venue = ''

        except Exception:
            citation_count = 0
            venue = ''

        # 원조 논문 여부
        is_landmark = paper['title'].lower().strip() in LANDMARK_PAPERS

        # 신뢰 학회 여부
        is_trusted = any(v.lower() in venue.lower() for v in TRUSTED_VENUES) or venue == ''

        enriched.append({
            **paper,
            'citation_count': citation_count,
            'venue':          venue,
            'is_landmark':    is_landmark,
            'is_trusted':     is_trusted,
        })

        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(papers)} 완료...")

        time.sleep(0.5)  # rate limit

    # 저장
    save_path = os.path.join(DATA_DIR, 'papers_enriched.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {save_path}")
    return enriched

if __name__ == '__main__':
    enrich_papers()
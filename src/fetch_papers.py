# src/fetch_papers.py
import os
import json
import time
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

API_KEY = "s2k-R2ysT7fyvqfmi60iUbvHXNRCFA4dgULEFxvft7kY"
HEADERS = {"x-api-key": API_KEY}

QUERIES = [
    "machine learning",
    "deep learning",
    "natural language processing",
    "large language models",
    "reinforcement learning",
    "transformer",
    "BERT language model",
    "retrieval augmented generation",
]

# 쿼리 검색으로 빠질 수 있는 원조 논문 목록 (개념 → Semantic Scholar paper ID)
# 논문 제목에 해당 개념 단어가 없어서 키워드 검색에서 누락되는 경우를 보완
CONCEPT_ANCHORS = {
    "transformer":  "204e3073870fae3d05bcbc2f6a8e263d9b72e776",  # Attention is All You Need
    "resnet":       "2c03df8b48bf3fa39054345bafabfeff15bfd11d",  # Deep Residual Learning for Image Recognition
    "gan":          "13bc4e683075bdd6a3f0155241c276a772d4aa06",  # Generative Adversarial Nets
    "word2vec":     "87f40e6f3022adbc1f1905e3e506abad05a9964f",  # Distributed Representations of Words
    "adam":         "a6cb366736791bcccc5c8639de5a8f9636bf87e8",  # Adam: A Method for Stochastic Optimization
    "attention":    "fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5",  # Neural Machine Translation by Jointly Learning to Align
    "seq2seq":      "cea967b59209c6be22829699f05b8b1ac4dc092d",  # Sequence to Sequence Learning with Neural Networks
    "lstm":         "2e9d221c206e9503ceb452302d68d10e293f2a10",  # Long Short-Term Memory
    "gpt2":         "9405cc0d6169988371b2755e573cc28650d14dfe",  # Language Models are Unsupervised Multitask Learners
    "instructgpt":  "d766bffc357127e0dc86dd69561d5aeb520d6f4c",  # Training language models to follow instructions
    "clip":         "6f870f7f02a8c59c3e23f407f3ef00dd1dcf8fc4",  # Learning Transferable Visual Models (CLIP)
    "ddpm":         "5c126ae3421f05768d8edd97ecd44b1364e2c99a",  # Denoising Diffusion Probabilistic Models
    "ppo":          "dce6f9d4017b1785979e7520fd0834ef8cf02f4b",  # Proximal Policy Optimization Algorithms
    "dropout":      "34f25a8704614163c4095b3ee2fc969b60de4698",  # Dropout: A Simple Way to Prevent Overfitting
    "batchnorm":    "995c5f5e62614fcb4d2796ad2faab969da51713e",  # Batch Normalization
    "rag":          "659bf9ce7175e1ec266ff54359e2bd76e0b7ff31",  # RAG for Knowledge-Intensive NLP Tasks
    "vit":          "268d347e8a55b5eb82fb5e7d2f800e33c75ab18a",  # An Image is Worth 16x16 Words (ViT)
}

# 앵커 논문에 부여할 개념 태그 (embed.py가 임베딩 텍스트에 포함)
ANCHOR_TAGS = {
    "204e3073870fae3d05bcbc2f6a8e263d9b72e776": ["transformer", "self-attention", "multi-head attention", "encoder decoder", "attention mechanism"],
    "2c03df8b48bf3fa39054345bafabfeff15bfd11d": ["resnet", "residual learning", "skip connection", "deep residual network", "image recognition"],
    "13bc4e683075bdd6a3f0155241c276a772d4aa06": ["GAN", "generative adversarial network", "generative model", "adversarial training"],
    "87f40e6f3022adbc1f1905e3e506abad05a9964f": ["word2vec", "word embedding", "word representation", "distributed representation"],
    "a6cb366736791bcccc5c8639de5a8f9636bf87e8": ["adam optimizer", "adaptive learning rate", "gradient descent optimization"],
    "fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5": ["attention mechanism", "bahdanau attention", "neural machine translation", "alignment"],
    "cea967b59209c6be22829699f05b8b1ac4dc092d": ["seq2seq", "sequence to sequence", "encoder decoder", "neural machine translation"],
    "2e9d221c206e9503ceb452302d68d10e293f2a10": ["LSTM", "long short-term memory", "recurrent neural network", "sequence modeling", "vanishing gradient"],
    "9405cc0d6169988371b2755e573cc28650d14dfe": ["GPT-2", "language model", "unsupervised learning", "text generation", "zero-shot"],
    "d766bffc357127e0dc86dd69561d5aeb520d6f4c": ["InstructGPT", "RLHF", "reinforcement learning from human feedback", "instruction tuning", "alignment"],
    "6f870f7f02a8c59c3e23f407f3ef00dd1dcf8fc4": ["CLIP", "contrastive learning", "vision language model", "zero-shot image classification", "multimodal"],
    "5c126ae3421f05768d8edd97ecd44b1364e2c99a": ["diffusion model", "DDPM", "denoising diffusion", "generative model", "image generation"],
    "dce6f9d4017b1785979e7520fd0834ef8cf02f4b": ["PPO", "proximal policy optimization", "reinforcement learning", "policy gradient"],
    "34f25a8704614163c4095b3ee2fc969b60de4698": ["dropout", "regularization", "overfitting", "neural network training"],
    "995c5f5e62614fcb4d2796ad2faab969da51713e": ["batch normalization", "training stability", "internal covariate shift", "deep learning optimization"],
    "659bf9ce7175e1ec266ff54359e2bd76e0b7ff31": ["RAG", "retrieval augmented generation", "knowledge-intensive NLP", "dense retrieval"],
    "268d347e8a55b5eb82fb5e7d2f800e33c75ab18a": ["ViT", "vision transformer", "image recognition", "patch embedding", "image classification"],
}

TRUSTED_VENUES = {
    'NeurIPS', 'ICML', 'ICLR', 'ACL', 'EMNLP', 'NAACL',
    'CVPR', 'ICCV', 'ECCV', 'AAAI', 'IJCAI',
    'IEEE', 'Nature', 'Science', 'arXiv'
}


def _paper_to_dict(p, is_landmark=False):
    venue = p.get('venue', '') or ''
    is_trusted = (
        any(v.lower() in venue.lower() for v in TRUSTED_VENUES)
        or venue == ''
        or 'arxiv' in venue.lower()
    )
    arxiv_id = (p.get('externalIds') or {}).get('ArXiv', '')
    url_paper = (
        f"https://arxiv.org/abs/{arxiv_id}"
        if arxiv_id
        else f"https://www.semanticscholar.org/paper/{p['paperId']}"
    )
    tags = ANCHOR_TAGS.get(p['paperId'], [])
    return {
        'id':             p['paperId'],
        'title':          p.get('title', ''),
        'abstract':       p.get('abstract', ''),
        'tags':           tags,
        'authors':        [a['name'] for a in (p.get('authors') or [])[:5]],
        'published':      str(p.get('year', '')),
        'citation_count': p.get('citationCount', 0) or 0,
        'venue':          venue,
        'is_trusted':     is_trusted,
        'is_landmark':    is_landmark,
        'url':            url_paper,
        'categories':     [],
    }


def fetch_anchor_papers(seen_ids):
    """CONCEPT_ANCHORS에 정의된 원조 논문을 paper ID로 직접 수집"""
    anchors = []
    fields = "paperId,title,abstract,authors,year,citationCount,venue,externalIds"

    for concept, paper_id in CONCEPT_ANCHORS.items():
        if paper_id in seen_ids:
            continue
        try:
            res = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
                params={"fields": fields},
                headers=HEADERS,
                timeout=15
            )
            if res.status_code != 200:
                print(f"  [{concept}] 오류: {res.status_code}")
                continue
            p = res.json()
            if not p.get('abstract'):
                print(f"  [{concept}] 초록 없음, 건너뜀")
                continue
            anchors.append(_paper_to_dict(p, is_landmark=True))
            seen_ids.add(paper_id)
            print(f"  + [{concept}] {p['title'][:60]}")
        except Exception as e:
            print(f"  [{concept}] 예외: {e}")
        time.sleep(0.3)

    return anchors


def fetch_papers(max_per_query=100):
    papers = []
    seen_ids = set()

    for query in QUERIES:
        print(f"\n[{query}] 논문 수집 중...")
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_per_query,
            "fields": "paperId,title,abstract,authors,year,citationCount,venue,externalIds"
        }

        try:
            res = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if res.status_code != 200:
                print(f"  오류: {res.status_code}")
                time.sleep(2)
                continue

            data = res.json()
            for p in data.get('data', []):
                if not p.get('paperId') or p['paperId'] in seen_ids:
                    continue
                if not p.get('abstract'):
                    continue
                seen_ids.add(p['paperId'])
                papers.append(_paper_to_dict(p, is_landmark=False))

        except Exception as e:
            print(f"  예외: {e}")

        print(f"  누적 {len(papers)}개")
        time.sleep(1)

    # 원조 논문 추가 (쿼리 검색에서 빠진 것만)
    print("\n[앵커 논문] 원조 논문 수집 중...")
    anchors = fetch_anchor_papers(seen_ids)
    papers.extend(anchors)
    print(f"  앵커 {len(anchors)}개 추가")

    papers.sort(key=lambda x: x['citation_count'], reverse=True)

    save_path = os.path.join(DATA_DIR, 'papers.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(papers)}개 논문 저장 완료: {save_path}")
    return papers


if __name__ == '__main__':
    fetch_papers()

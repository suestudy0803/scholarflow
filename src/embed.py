# src/embed.py
import os
import json
import pickle
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 논문 로드
with open(os.path.join(DATA_DIR, 'papers.json'), 'r') as f:
    papers = json.load(f)

print(f"총 {len(papers)}개 논문 로드")
print(f"GPU: {torch.cuda.get_device_name(0)}")

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# title + abstract 합쳐서 임베딩
def build_text(p):
    base = f"{p['title']}. {p['abstract']}"
    tags = p.get('tags', [])
    if tags:
        base += f" Keywords: {', '.join(tags)}."
    return base

texts = [build_text(p) for p in papers]

print("임베딩 중...")
embeddings = model.encode(
    texts,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"임베딩 완료: {embeddings.shape}")

# 저장
save = {
    'embeddings': embeddings,
    'papers':     papers,
}

save_path = os.path.join(DATA_DIR, 'embeddings.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(save, f)

print(f"저장 완료: {save_path}")
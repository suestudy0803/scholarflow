# ScholarFlow

AI/ML/NLP 논문 검색 및 추천 시스템

**[scholarflow-project.streamlit.app](https://scholarflow-project.streamlit.app)**

---

## 소개

처음 AI/ML을 공부할 때 "어떤 논문부터 읽어야 하지?"라는 고민에서 출발한 프로젝트입니다.  
단순 키워드 검색이 아닌 **의미 기반 Dense Retrieval**로 논문을 찾고,  
인용수와 원조 논문 여부를 결합한 **하이브리드 랭킹**으로 학습에 가장 유용한 논문을 상위에 노출합니다.

---

## 주요 기능

- **의미 기반 검색** — `all-MiniLM-L6-v2` Sentence Transformer로 699편 논문 임베딩, 코사인 유사도 검색
- **하이브리드 랭킹** — 유사도(75%) + 인용수(25%) + 원조 논문 보너스(+0.15)
- **Concept Anchor** — `transformer`, `GAN`, `ViT` 등 17개 약어 쿼리에 원조 논문 자동 연결
- **유사 논문 추천** — 논문 상세 페이지에서 임베딩 기반 유사 논문 5편 추천
- **학회명 정규화** — Semantic Scholar API의 전체 학회명을 ACL, NeurIPS, CVPR 등 약어로 표시

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 데이터 수집 | Semantic Scholar Academic Graph API |
| 임베딩 모델 | `all-MiniLM-L6-v2` (Sentence Transformers) |
| 유사도 검색 | NumPy 행렬 내적 |
| Landmark 분류 | Claude Haiku API |
| 프론트엔드 | Streamlit |
| 배포 | Streamlit Community Cloud |

---

## 프로젝트 구조

```
scholarflow/
├── main.py              # Streamlit 앱 진입점
├── requirements.txt
├── src/
│   ├── retrieval.py     # Dense Retrieval + 하이브리드 랭킹
│   ├── fetch_papers.py  # Semantic Scholar API 수집
│   ├── embed.py         # 임베딩 생성
│   └── landmark.py      # Claude API 원조 논문 분류
└── data/
    ├── papers.json      # 699편 논문 메타데이터
    └── embeddings.pkl   # 699×384 임베딩 행렬
```

---

## 로컬 실행

```bash
git clone https://github.com/suestudy0803/scholarflow.git
cd scholarflow
pip install -r requirements.txt
streamlit run main.py
```

---

## 검색 예시

| 쿼리 | 1위 논문 |
|------|---------|
| transformer | Attention is All You Need |
| GAN | Generative Adversarial Nets |
| LSTM | Long Short-Term Memory |
| diffusion model | Denoising Diffusion Probabilistic Models |
| word2vec | Distributed Representations of Words |

---

## 데이터셋

- **699편** AI/ML/NLP 논문 (Semantic Scholar API 수집)
- **17편** Concept Anchor 원조 논문 (수동 등록)
- 수집 쿼리: `machine learning`, `deep learning`, `NLP`, `large language models`, `reinforcement learning`, `transformer`, `BERT`, `retrieval augmented generation`

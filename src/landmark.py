# src/landmark.py
import os
import json
import anthropic
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def evaluate_landmarks():
    with open(os.path.join(DATA_DIR, 'papers.json'), 'r') as f:
        papers = json.load(f)

    client = anthropic.Anthropic()

    # 인용수 상위 100개만 평가
    top_papers = sorted(papers, key=lambda x: x['citation_count'], reverse=True)[:100]
    top_ids = {p['id'] for p in top_papers}

    print(f"상위 100개 논문 landmark 평가 시작...")

    for i, paper in enumerate(papers):
        if paper['id'] not in top_ids:
            paper['is_landmark'] = False
            continue

        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": f"""Is this paper the ORIGINAL paper that FIRST introduced a major concept in AI/ML/NLP?
NOT follow-up work, NOT improvements, NOT applications, NOT surveys.
Only answer yes if this paper is THE paper that first proposed the concept (e.g. Transformer, BERT, ResNet, GAN).
Title: {paper['title']}
Citations: {paper['citation_count']}
Answer only: yes or no"""
                }]
            )
            answer = message.content[0].text.strip().lower()
            paper['is_landmark'] = answer.startswith('yes')

            if paper['is_landmark']:
                print(f"  ⭐ [{paper['citation_count']:,}회] {paper['title'][:60]}")

        except Exception as e:
            print(f"  오류: {e}")
            paper['is_landmark'] = False

        time.sleep(0.5)

    # 저장
    save_path = os.path.join(DATA_DIR, 'papers.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    landmarks = [p for p in papers if p['is_landmark']]
    print(f"\n총 {len(landmarks)}개 landmark 논문 확인")
    print("저장 완료")

if __name__ == '__main__':
    evaluate_landmarks()
import json, random
import statistics
from openai import OpenAI

client = OpenAI(api_key="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")

# ---------------------------
# 1) 평가 함수
# ---------------------------
def evaluate_triple(triple):
    prompt = f"""
당신은 사건-관계-결과 트리플을 평가하는 전문 평가자입니다.
주어진 트리플의 품질을 다음 3가지 기준으로 평가하세요.

===========================
[평가 기준 설명]
===========================

1) **일관성(Consistency, 1~5점)**
- Head 문장과 Tail 문장이 Relation에 따라 논리적으로 이어지는가?
- 사건의 주체·객체·행동이 충돌하지 않는가?
- “이 Relation이면 일반적으로 이런 Tail이 나와야 한다”에 부합하는가?

예시:
- Head: "A가 B에게 사과했다." / relation=xIntent → Tail: "A는 잘못을 인정하고자 했다." → 5점
- Head: "A가 밥을 먹었다." / relation=oReact → Tail: "사람들은 그 소식에 충격을 받았다." → 1점

2) **상식성(Commonsense Plausibility, 1~5점)**
- 일반적인 인간 경험, 감정, 인과성에 맞는 자연스러운 내용인가?
- Tail이 과도하게 억지·비약·비현실적이지 않은가?

예시:
- Head: "A는 시험을 망쳤다." / Tail: "A는 속상함을 느꼈다." → 5점
- Head: "A는 물을 마셨다." / Tail: "A는 하늘을 날 수 있게 되었다." → 1점

3) **사실성(Factuality, 1~5점)**
- Tail 문장이 현실 세계의 사실이나 가능한 사실과 충돌하지 않는가?
- 사건의 속성, 감정, 욕구가 인간 행동의 기본 패턴을 위반하지 않는가?

예시:
- "A는 배가 아팠다." / Tail: "A는 불편함을 느꼈다." → 5점
- "A는 배가 아팠다." / Tail: "A는 저절로 치유되는 마법을 사용했다." → 1점

===========================
[평가 지침]
===========================
- 세 항목은 서로 독립적으로 평가한다.
- 점수는 반드시 **1~5 정수**만 사용한다.
- 너무 과한 추론은 하지 말고 Head·Relation·Tail 안에서 판단한다.
- 한국어 문장의 어색함이 아닌 논리적·개념적 품질을 평가한다.

===========================
[평가할 트리플]
===========================
Head: {triple['head']}
Relation: {triple['relation']}
Tail: {triple['tail']}

===========================
[출력 형식]
===========================
아래 JSON 구조만 출력하세요. 다른 설명 금지.

{{
  "consistency": (1~5),
  "commonsense": (1~5),
  "factuality": (1~5)
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)


# ---------------------------
# 2) JSON 로드
# ---------------------------
with open("/home/jaesang/kg_project/output/sampled_triples_100.json", "r", encoding="utf-8") as f:
    triples = json.load(f)

# ---------------------------
# 3) 100개 랜덤 샘플링
# ---------------------------
sampled = random.sample(triples, 300)

print("=== 선택된 300개 샘플 ===", flush=True)
for t in sampled:
    print(t, flush=True)
print("=====================\n", flush=True)

# ---------------------------
# 4) 평가 수행
# ---------------------------
results = []
for i, tri in enumerate(sampled):
    print(f"[{i+1}/300] 평가 중...", flush=True)   # ★ 진행률 즉시 표시
    scores = evaluate_triple(tri)
    results.append({**tri, **scores})

# ---------------------------
# 5) 평균 계산
# ---------------------------
cons_avg = statistics.mean([r["consistency"] for r in results])
com_avg  = statistics.mean([r["commonsense"] for r in results])
fact_avg = statistics.mean([r["factuality"] for r in results])

overall_avg = statistics.mean([cons_avg, com_avg, fact_avg])

print("===== 최종 평균 =====", flush=True)
print(f"일관성: {cons_avg:.2f}", flush=True)
print(f"상식성: {com_avg:.2f}", flush=True)
print(f"사실성: {fact_avg:.2f}", flush=True)
print(f"종합 평균: {overall_avg:.2f}", flush=True)

# ---------------------------
# 6) 결과 저장
# ---------------------------
save_path = "/home/jaesang/kg_project/output/triple_eval_results.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n평가 결과 저장 완료 → {save_path}", flush=True)

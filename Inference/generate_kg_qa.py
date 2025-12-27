import json
import random
import time
import re
from openai import OpenAI

# =========================
# OpenAI client
# =========================
client = OpenAI(
    api_key="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"
)

# =========================
# Relation definition
# =========================
RELATION_DEFINITION = {
    "xIntent": "행위자가 이 발화나 행동을 하게 만든 목적 또는 이유",
    "xEffect": "사건 이후 행위자 자신에게 직접적으로 발생한 결과",
    "xReact": "사건 이후 행위자가 느낄 가능성이 높은 감정",
    "oReact": "이 사건을 들은 다른 사람들이 느낄 가능성이 높은 감정",
    "oWant": "이 사건 이후 주변 사람들이 원하게 되는 행동 또는 상태",
}

RELATION_QUESTION = {
    "xIntent": "왜 행위자는 이런 말을 하거나 행동을 했을까?",
    "xEffect": "이 사건 이후, 행위자에게 어떤 변화가 일어났을까?",
    "xReact": "이 사건 이후, 행위자는 어떤 감정을 느꼈을까?",
    "oReact": "이 사건을 들은 다른 사람들은 어떤 감정을 느꼈을까?",
    "oWant": "이 사건 이후, 주변 사람들은 무엇을 원하게 되었을까?",
}

# =========================
# Prompt builder (FULL VERSION 유지)
# =========================
def build_prompt(head, relation):
    return f"""
당신은 사건 기반 상식 인과 추론 모델입니다.

[사건]
{head}

[관계 정의]
{relation}: {RELATION_DEFINITION[relation]}

[질문]
{RELATION_QUESTION[relation]}

[출력 규칙]
- 반드시 한 문장으로 답할 것
- 사건을 그대로 반복하거나 바꿔 말하지 말 것
- {relation}의 정의를 벗어나는 내용은 금지
- 즉각적이고 직접적인 인과/감정만 허용
- 모호한 표현(“어떤”, “무언가”) 사용 금지
- 단일 결과 / 단일 감정만 허용
"""

# =========================
# Generic intent / want filter
# =========================
ABSTRACT_NOUNS = [
    "사회", "정의", "도덕", "윤리", "가치", "안정", "질서",
    "중요성", "필수", "필요성", "책임", "의무", "원칙"
]

NORMATIVE_PHRASES = [
    "중요하다고", "필수적", "바람직", "옳다고",
    "해야 한다", "믿었기 때문에", "생각했기 때문에"
]

CAMPAIGN_EXPRESSIONS = [
    "모두", "사람들은", "우리 사회",
    "누구나", "일반적으로", "사회적으로"
]

PURPOSE_WEAK_ENDINGS = [
    "알리기 위해", "전달하기 위해", "공유하기 위해"
]

def is_generic_intent_or_want(answer: str) -> bool:
    if sum(1 for w in ABSTRACT_NOUNS if w in answer) >= 2:
        return True
    if any(p in answer for p in NORMATIVE_PHRASES):
        return True
    if any(p in answer for p in CAMPAIGN_EXPRESSIONS):
        return True
    if any(answer.strip().endswith(p) for p in PURPOSE_WEAK_ENDINGS):
        return True
    if not re.search(r"(행위자|주변 사람|상대방|다른 사람)", answer):
        return True
    return False

# =========================
# Extra strict ATOMIC filters (A-mode)
# =========================
def violates_xeffect_subject(answer: str) -> bool:
    return any(w in answer for w in ["상대방", "다른 사람", "주변 사람"])

def has_multiple_effects(answer: str) -> bool:
    return any(w in answer for w in ["그리고", "고 ", "며 ", "및"])

def is_redundant_react(head: str, answer: str) -> bool:
    redundant_keywords = ["배고프", "졸리", "취하", "아프"]
    return any(k in head and k in answer for k in redundant_keywords)

COGNITIVE_EMOTION_WORDS = [
    "심각성", "문제의식", "위험성", "경각심",
    "인식", "깨달", "이해"
]

def is_over_reasoned_oreact(answer: str) -> bool:
    return any(w in answer for w in COGNITIVE_EMOTION_WORDS)

def has_multiple_emotions(answer: str) -> bool:
    return any(w in answer for w in ["과", "와", "및"])

# =========================
# Quality filter (FINAL–A)
# =========================
def is_low_quality(head, answer, relation):
    head_tokens = set(re.findall(r"\w+", head))
    ans_tokens = set(re.findall(r"\w+", answer))
    if len(head_tokens & ans_tokens) / max(len(head_tokens), 1) > 0.6:
        return True

    if any(b in answer for b in ["알게 되었다", "말했다", "언급했다", "소개했다"]):
        return True

    if relation == "xEffect" and any(w in answer for w in ["느꼈", "감정"]):
        return True

    if relation in ["xReact", "oReact"] and any(w in answer for w in ["원하게", "결심", "행동"]):
        return True

    if relation in ["xIntent", "oWant"] and is_generic_intent_or_want(answer):
        return True

    if relation == "xEffect":
        if violates_xeffect_subject(answer) or has_multiple_effects(answer):
            return True

    if relation == "xReact":
        if is_redundant_react(head, answer) or has_multiple_emotions(answer):
            return True

    if relation == "oReact" and is_over_reasoned_oreact(answer):
        return True

    return False

# =========================
# GPT call
# =========================
def generate_answer(head, relation):
    prompt = build_prompt(head, relation)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# =========================
# Main (dialog_triples.json 대응)
# =========================
def main():
    INPUT_PATH = "/home/jaesang/kg_project/output/dialog_triples.json"
    OUTPUT_PATH = "/home/jaesang/kg_project/output/kg_qa_samples_FINAL.jsonl"

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        dialog_files = json.load(f)

    # 🔥 dialog_triples 구조 flatten
    triples = []
    for doc in dialog_files:
        for tri in doc["triples"]:
            if tri["relation"] in RELATION_DEFINITION:
                triples.append(tri)

    random.shuffle(triples)

    saved = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for idx, tri in enumerate(triples):
            head = tri["head"]
            relation = tri["relation"]
            tail = tri["tail"]

            try:
                answer = generate_answer(head, relation)

                if is_low_quality(head, answer, relation):
                    print(f"[SKIP] {relation} :: {answer}", flush=True)
                    continue

                sample = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{head}\n\n{RELATION_QUESTION[relation]}"
                        },
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    ],
                    "source_triple": {
                        "head": head,
                        "relation": relation,
                        "tail": tail
                    }
                }

                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                out_f.flush()

                saved += 1
                print(f"[OK] {saved} saved", flush=True)
                time.sleep(0.4)

            except Exception as e:
                print(f"[ERROR] {idx}: {e}", flush=True)

    print(f"\n=== DONE: {saved} ATOMIC-strict QA samples generated ===", flush=True)

if __name__ == "__main__":
    main()

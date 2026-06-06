from __future__ import annotations

import re


EVENT_PROMPT_V3 = """
You are an advanced Korean SNS dialogue event extractor.

GOAL:
Extract key events from the conversation. Each event must contain:
- a natural event_sentence (speaker identity implicit, using expressions like “한 참여자는”, “다른 친구는”, “상대방은”)
- a causal explanation referencing previous dialogue context.

EVENT SENTENCE RULES:
- Never use “발화자”.
- Use natural and varied Korean expressions for the actor.
- Must summarize the action or intention clearly.

CAUSE RULES:
- Must incorporate conversational context.
- Avoid generic or shallow reasons.
- Provide meaningful human-level reasoning.

OUTPUT FORMAT:
{
  "events": [
    {
      "id": "E1",
      "event_sentence": "...",
      "event_cause": "..."
    }
  ]
}

STRICT:
- JSON only.
""".strip()


TRIPLE_PROMPT_V4 = """
You are an advanced Korean commonsense reasoning model following the ATOMIC framework.
Your job is to generate 9 high-quality ATOMIC relations for the given event.

INPUT EVENT:
- event_sentence: 자연스러운 사건 설명
- event_cause: 사건이 발생한 이유 또는 대화 흐름

IMPORTANT RELATION RULES (STRICT):

1) xIntent
- 행위자가 왜 그런 행동/말을 했는지 목적 또는 숨겨진 이유
- 감정 금지
- 1문장

2) xNeed
- 사건이 일어나기 *전에* 충족되어야 했던 조건
- 사건 내용 반복 금지
- 감정/욕구 금지
- 1문장

3) xAttr
- 사건으로부터 추론되는 행위자의 성격/특성
- 1문장

4) xEffect
- 사건 이후 행위자에게 일어나는 상태 변화 (감정 금지)
- “무엇을 하게 된다”, “상태가 어떻게 변한다"와 같은 형태
- 감정이 포함되면 안 됨
- 1문장

5) xReact
- 행위자의 감정만 표현
- "기뻤다 / 불안했다 / 민망함을 느꼈다" 등
- 감정 외 요소 금지
- 1문장

6) xWant
- 사건 직후 행위자가 원하는 것
- 1문장

7) oEffect
- 주변 사람들이 사건으로 인해 겪는 상태 변화
- 감정 금지
- 1문장

8) oReact
- 주변 사람들이 느끼는 감정
- 1문장

9) oWant
- 주변 사람들이 통상적으로 바라게 되는 후속 행동/상태
- 1문장

OUTPUT FORMAT (STRICT JSON):
{
  "triples": [
    {"relation": "xIntent", "tail": "...", "event_id": "E1", "head": "..."},
    ...
  ]
}

REQUIREMENTS:
- 모든 tail은 자연스러운 한국어 ‘완전한 문장’이어야 함.
- 감정이 허용된 relation(xReact, oReact) 외에는 감정 단어 사용 금지.
- xEffect와 oEffect는 반드시 actor vs others를 구분할 것.
- 반복, 모호한 단어, 단일명사 출력 금지.
""".strip()


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


ABSTRACT_NOUNS = [
    "사회", "정의", "도덕", "윤리", "가치", "안정", "질서",
    "중요성", "필수", "필요성", "책임", "의무", "원칙",
]


NORMATIVE_PHRASES = [
    "중요하다고", "필수적", "바람직", "옳다고",
    "해야 한다", "믿었기 때문에", "생각했기 때문에",
]


CAMPAIGN_EXPRESSIONS = [
    "모두", "사람들은", "우리 사회",
    "누구나", "일반적으로", "사회적으로",
]


PURPOSE_WEAK_ENDINGS = [
    "알리기 위해", "전달하기 위해", "공유하기 위해",
]


COGNITIVE_EMOTION_WORDS = [
    "심각성", "문제의식", "위험성", "경각심",
    "인식", "깨달", "이해",
]


def estimate_event_count(num_utterances: int) -> int:
    if num_utterances <= 5:
        return 3
    if num_utterances <= 12:
        return 4
    return 5


def build_legacy_event_prompt(conversation_text: str, n_events: int) -> str:
    return f"{EVENT_PROMPT_V3}\n\nN = {n_events}\n===DIALOG===\n{conversation_text}"


def build_legacy_triple_prompt(event_text: str, event_cause: str) -> str:
    input_block = f"event_sentence: {event_text}\nevent_cause: {event_cause}"
    return f"{TRIPLE_PROMPT_V4}\n{input_block}"


def build_legacy_qa_prompt(head: str, relation: str) -> str:
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
""".strip()


def is_generic_intent_or_want(answer: str) -> bool:
    if sum(1 for word in ABSTRACT_NOUNS if word in answer) >= 2:
        return True
    if any(phrase in answer for phrase in NORMATIVE_PHRASES):
        return True
    if any(phrase in answer for phrase in CAMPAIGN_EXPRESSIONS):
        return True
    if any(answer.strip().endswith(ending) for ending in PURPOSE_WEAK_ENDINGS):
        return True
    if not re.search(r"(행위자|주변 사람|상대방|다른 사람)", answer):
        return True
    return False


def violates_xeffect_subject(answer: str) -> bool:
    return any(word in answer for word in ["상대방", "다른 사람", "주변 사람"])


def has_multiple_effects(answer: str) -> bool:
    return any(word in answer for word in ["그리고", "고 ", "며 ", "및"])


def is_redundant_react(head: str, answer: str) -> bool:
    redundant_keywords = ["배고프", "졸리", "취하", "아프"]
    return any(keyword in head and keyword in answer for keyword in redundant_keywords)


def is_over_reasoned_oreact(answer: str) -> bool:
    return any(word in answer for word in COGNITIVE_EMOTION_WORDS)


def has_multiple_emotions(answer: str) -> bool:
    return any(word in answer for word in ["과", "와", "및"])


def is_low_quality_qa_answer(head: str, answer: str, relation: str) -> bool:
    head_tokens = set(re.findall(r"\w+", head))
    answer_tokens = set(re.findall(r"\w+", answer))
    if len(head_tokens & answer_tokens) / max(len(head_tokens), 1) > 0.6:
        return True

    if any(bad in answer for bad in ["알게 되었다", "말했다", "언급했다", "소개했다"]):
        return True

    if relation == "xEffect" and any(word in answer for word in ["느꼈", "감정"]):
        return True

    if relation in ["xReact", "oReact"] and any(word in answer for word in ["원하게", "결심", "행동"]):
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

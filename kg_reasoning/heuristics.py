from __future__ import annotations

from kg_reasoning.schema import RELATION_TYPES
from kg_reasoning.text import normalize_text, tokenize, unique_preserving_order


TAG_RULES: dict[str, tuple[str, ...]] = {
    "pain_rest": ("배가", "복통", "아파", "조퇴", "쉬고 싶", "보건실"),
    "release_delay": ("출시", "안 하는 거", "아직도 안 하", "여름에 출시", "연기"),
    "self_doubt_retry": ("자신 없는", "다시 시도", "의심", "할 수 있을까", "걱정"),
    "wildfire_concern": ("산불", "불이 번", "피해", "바람이 세", "캘리포니아", "대피"),
    "meal_decision": ("뭐 먹", "메뉴", "배 너무 고파", "맛있는 거", "버거", "라면", "국밥"),
    "umbrella_rain": ("우산", "비를 맞", "젖었", "비 온다"),
    "exam_failure": ("시험", "망쳤", "불합격", "결과", "낙담"),
    "overwork_exhaustion": ("야근", "퇴근 못", "지쳤", "피곤", "번아웃", "쉴 틈"),
}


STOPWORDS = {
    "그리고",
    "근데",
    "그냥",
    "진짜",
    "너무",
    "정말",
    "아직도",
    "약간",
    "오늘",
    "내일",
    "이번",
    "그때",
    "이제",
}


TAG_TO_EVENT = {
    "pain_rest": "복통 때문에 일정을 줄이고 쉬고 싶어 한다",
    "release_delay": "출시 지연 때문에 기다림이 길어져 답답해한다",
    "self_doubt_retry": "다시 시도하고 싶지만 스스로를 의심한다",
    "wildfire_concern": "대형 산불 피해 확대를 걱정한다",
    "meal_decision": "배가 고프지만 무엇을 먹을지 고민한다",
    "umbrella_rain": "우산을 잊고 비를 맞아 곤란해한다",
    "exam_failure": "시험 결과 때문에 낙담하고 다음 계획을 고민한다",
    "overwork_exhaustion": "과도한 야근으로 지쳐 회복이 필요하다",
    "generic": "대화 속 핵심 상황을 정리하고 다음 행동을 고민한다",
}


TAG_TO_ANCHORS = {
    "pain_rest": ["아프다", "쉬다", "조퇴하다"],
    "release_delay": ["출시하다", "기다리다", "연기되다"],
    "self_doubt_retry": ["시도하다", "의심하다", "도전하다"],
    "wildfire_concern": ["번지다", "대피하다", "걱정하다"],
    "meal_decision": ["먹다", "고르다", "고민하다"],
    "umbrella_rain": ["잊다", "맞다", "곤란하다"],
    "exam_failure": ["망치다", "낙담하다", "준비하다"],
    "overwork_exhaustion": ["야근하다", "지치다", "회복하다"],
    "generic": ["말하다", "생각하다"],
}


TAG_TO_RELATIONS = {
    "pain_rest": {
        "xIntent": "몸 상태를 솔직하게 알리고 휴식을 얻고 싶어 한다.",
        "xNeed": "이미 복통이 심해져 일정을 조정해야 할 상황이어야 한다.",
        "xEffect": "당장의 일정이나 활동량을 줄이게 된다.",
        "xReact": "불편함과 지침을 느낀다.",
        "xWant": "바로 쉬거나 상태를 돌보고 싶어 한다.",
        "oEffect": "주변 사람은 일정을 조정하거나 상태를 확인하게 된다.",
        "oReact": "주변 사람은 걱정된다.",
        "oWant": "주변 사람은 무리하지 말고 쉬기를 바란다.",
        "xAttr": "자신의 몸 상태를 비교적 솔직하게 드러내는 사람이다.",
    },
    "release_delay": {
        "xIntent": "기다리던 일정이 왜 미뤄지는지 확인하고 싶어 한다.",
        "xNeed": "이미 약속된 출시 시점이 있었고 그 일정이 어긋나야 한다.",
        "xEffect": "기다리는 시간이 길어져 계획을 다시 잡게 된다.",
        "xReact": "답답함과 실망을 느낀다.",
        "xWant": "정확한 일정 공지나 확실한 설명을 원한다.",
        "oEffect": "함께 기다리던 사람들도 기대 일정을 다시 조정하게 된다.",
        "oReact": "주변 사람도 아쉬움을 느낀다.",
        "oWant": "주변 사람은 일정이 빨리 확정되기를 바란다.",
        "xAttr": "관심 있는 일을 오래 추적하는 편이다.",
    },
    "self_doubt_retry": {
        "xIntent": "다시 도전해 보고 싶지만 실패를 반복하고 싶지 않아 한다.",
        "xNeed": "이전에 만족스럽지 않은 결과를 경험했어야 한다.",
        "xEffect": "다음 시도를 더 조심스럽게 준비하게 된다.",
        "xReact": "불안함과 망설임을 느낀다.",
        "xWant": "확신을 얻고 다시 시도할 힘을 갖고 싶어 한다.",
        "oEffect": "주변 사람은 격려하거나 현실적인 조언을 하게 된다.",
        "oReact": "주변 사람은 안타까움과 응원하는 마음을 느낀다.",
        "oWant": "주변 사람은 다시 시도해 보기를 바란다.",
        "xAttr": "자기평가가 엄격한 사람이다.",
    },
    "wildfire_concern": {
        "xIntent": "피해 규모를 파악하고 더 큰 위험을 대비하고 싶어 한다.",
        "xNeed": "이미 대형 화재나 재난 소식이 빠르게 번지고 있어야 한다.",
        "xEffect": "위험 지역과 향후 상황을 계속 주시하게 된다.",
        "xReact": "불안함과 긴장감을 느낀다.",
        "xWant": "피해가 더 커지지 않고 빨리 진정되길 바란다.",
        "oEffect": "주변 사람도 안전 정보와 대피 상황을 확인하게 된다.",
        "oReact": "주변 사람은 걱정과 충격을 느낀다.",
        "oWant": "주변 사람은 빠른 진화와 안전 확보를 바란다.",
        "xAttr": "사회적 위험에 민감하게 반응하는 사람이다.",
    },
    "meal_decision": {
        "xIntent": "허기를 빨리 해결하면서도 만족스러운 선택을 하고 싶어 한다.",
        "xNeed": "이미 배가 고프고 선택할 음식 후보가 있어야 한다.",
        "xEffect": "메뉴 비교나 동행과의 조율을 하게 된다.",
        "xReact": "허기와 기대감을 함께 느낀다.",
        "xWant": "지금 가장 끌리는 음식을 먹고 싶어 한다.",
        "oEffect": "주변 사람은 메뉴를 함께 조정하거나 추천하게 된다.",
        "oReact": "주변 사람은 가볍게 들뜨거나 고민하게 된다.",
        "oWant": "주변 사람은 모두가 만족할 메뉴를 고르길 바란다.",
        "xAttr": "사소한 선택도 즐겁게 고민하는 사람이다.",
    },
    "umbrella_rain": {
        "xIntent": "비를 피하거나 빨리 몸을 말리고 싶어 한다.",
        "xNeed": "우산 없이 비를 맞게 되는 상황이 먼저 발생해야 한다.",
        "xEffect": "젖은 옷을 정리하거나 이동 계획을 바꾸게 된다.",
        "xReact": "당황함과 불편함을 느낀다.",
        "xWant": "비를 피할 곳이나 마른 옷을 빨리 찾고 싶어 한다.",
        "oEffect": "주변 사람은 우산을 함께 쓰거나 도움을 주게 된다.",
        "oReact": "주변 사람은 안쓰럽게 느낀다.",
        "oWant": "주변 사람은 감기 걸리지 않게 바로 챙기길 바란다.",
        "xAttr": "예상치 못한 실수를 겪어도 빠르게 수습하려는 사람이다.",
    },
    "exam_failure": {
        "xIntent": "무엇이 부족했는지 정리하고 다음 기회를 준비하고 싶어 한다.",
        "xNeed": "이미 시험 결과가 기대에 못 미쳐야 한다.",
        "xEffect": "학습 계획이나 진로 계획을 다시 조정하게 된다.",
        "xReact": "속상함과 낙담을 느낀다.",
        "xWant": "다음에는 더 나은 결과를 얻고 싶어 한다.",
        "oEffect": "주변 사람은 위로하거나 공부 계획을 같이 점검하게 된다.",
        "oReact": "주변 사람은 안타까움을 느낀다.",
        "oWant": "주변 사람은 다시 준비해서 만회하길 바란다.",
        "xAttr": "결과를 오래 곱씹는 성향이 있다.",
    },
    "overwork_exhaustion": {
        "xIntent": "밀린 일을 끝내려 했지만 동시에 회복의 필요도 느낀다.",
        "xNeed": "이미 업무량이 과도하거나 휴식이 부족한 상태여야 한다.",
        "xEffect": "집중력이 떨어지고 회복 일정을 따로 잡게 된다.",
        "xReact": "지침과 무력감을 느낀다.",
        "xWant": "충분히 쉬면서 리듬을 회복하고 싶어 한다.",
        "oEffect": "주변 사람은 업무 분담이나 일정 조정을 고민하게 된다.",
        "oReact": "주변 사람은 걱정스럽게 느낀다.",
        "oWant": "주변 사람은 무리하지 말고 쉬길 바란다.",
        "xAttr": "책임감이 강해서 쉽게 쉬지 못하는 사람이다.",
    },
    "generic": {
        "xIntent": "현재 상황을 설명하고 도움이나 공감을 얻고 싶어 한다.",
        "xNeed": "이미 말로 꺼낼 만한 변화나 고민이 있어야 한다.",
        "xEffect": "상황을 정리하고 다음 행동을 고민하게 된다.",
        "xReact": "복합적인 감정을 느낀다.",
        "xWant": "상황이 더 나아질 방향을 찾고 싶어 한다.",
        "oEffect": "주변 사람은 상황을 파악하고 반응하게 된다.",
        "oReact": "주변 사람은 관심이나 걱정을 느낀다.",
        "oWant": "주변 사람은 문제없이 잘 풀리길 바란다.",
        "xAttr": "현재 상황을 언어로 정리하려는 사람이다.",
    },
}


def detect_tags(text: str) -> list[str]:
    normalized = normalize_text(text)
    matches: list[str] = []
    for tag, cues in TAG_RULES.items():
        if any(cue in normalized for cue in cues):
            matches.append(tag)
    return matches or ["generic"]


def choose_primary_tag(text: str) -> str:
    return detect_tags(text)[0]


def choose_event_text(text: str) -> str:
    return TAG_TO_EVENT[choose_primary_tag(text)]


def choose_anchors(text: str) -> list[str]:
    tag = choose_primary_tag(text)
    lexical = [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1][:3]
    return unique_preserving_order(TAG_TO_ANCHORS[tag] + lexical)[:5]


def build_relation_bundle(text: str) -> dict[str, str]:
    tag = choose_primary_tag(text)
    bundle = TAG_TO_RELATIONS[tag]
    for relation in RELATION_TYPES:
        if relation not in bundle:
            raise ValueError(f"Missing relation template for {tag}:{relation}")
    return bundle

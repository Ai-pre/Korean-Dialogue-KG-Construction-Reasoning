from __future__ import annotations

from typing import Any

from kg_reasoning.schema import (
    INDIRECTNESS_LABELS,
    MULTIHOP_QUERY_TYPES,
    POLITENESS_LABELS,
    POWER_RELATION_LABELS,
    PRAGMATIC_SIGNAL_TYPES,
    SOCIAL_DISTANCE_LABELS,
    SOCIAL_RELATION_LABELS,
    SPEECH_ACT_LABELS,
    STANCE_LABELS,
)


PRAGMATIC_SIGNAL_TAXONOMY: dict[str, dict[str, Any]] = {
    "speechAct": {
        "description": "발화가 수행하는 직접적 행위. 한국어 대화의 질문, 공감, 거절, 권유, 불만을 구분한다.",
        "labels": SPEECH_ACT_LABELS,
        "examples": ("간접거절", "공감", "권유", "불만"),
    },
    "stance": {
        "description": "화자의 태도와 입장. 모호한 한국어 표현에서 숨은 긍정/부정 방향을 읽기 위해 사용한다.",
        "labels": STANCE_LABELS,
        "examples": ("머뭇거리는 동의", "완곡한 부정", "장난스러운 동조"),
    },
    "politeness": {
        "description": "반말/존댓말/격식체의 층위를 표시한다.",
        "labels": POLITENESS_LABELS,
        "examples": ("반말", "존댓말", "격식체"),
    },
    "indirectness": {
        "description": "직접적 표현인지, 완곡한 암시인지, 비꼼인지 구분한다.",
        "labels": INDIRECTNESS_LABELS,
        "examples": ("직설", "완곡", "암시", "비꼼"),
    },
    "emotionCue": {
        "description": "ㅠㅠ, 키키, 하..., ... 같은 표면 신호가 드러내는 감정 단서.",
        "labels": ("sadness", "relief", "playfulness", "frustration", "awkwardness"),
    },
    "listenerPressure": {
        "description": "상대가 응답해야 한다는 압박이나 눈치를 유발하는 표현.",
        "labels": ("none", "low", "medium", "high"),
    },
    "facework": {
        "description": "체면 살리기, 완곡거절, 책임 회피처럼 face-saving과 관련된 행위.",
        "labels": ("face_saving", "face_threatening", "self_protective", "other_protective"),
    },
    "humor": {
        "description": "농담, 티키타카, 자기비하성 유머 같은 한국어 대화의 친밀도 신호.",
        "labels": ("none", "light_banter", "teasing", "self_deprecating"),
    },
    "ellipsisSubject": {
        "description": "생략된 주체를 복원하기 위한 슬롯.",
        "labels": ("speaker", "listener", "third_party", "group", "unknown"),
    },
    "ellipsisTarget": {
        "description": "생략된 대상, 화제, 감정 방향을 복원하기 위한 슬롯.",
        "labels": ("self", "listener", "event", "object", "social_role", "unknown"),
    },
    "agreementShift": {
        "description": "동의에서 반대, 반대에서 수긍으로 바뀌는 흐름.",
        "labels": ("stable", "softened", "reversed", "intensified"),
    },
    "certainty": {
        "description": "말끝과 표현이 드러내는 확신 정도.",
        "labels": ("certain", "probable", "uncertain", "speculative"),
    },
}


SOCIAL_RELATION_TAXONOMY: dict[str, dict[str, Any]] = {
    "relation_label": {
        "description": "화자 쌍의 기본 사회적 관계.",
        "labels": SOCIAL_RELATION_LABELS,
    },
    "social_distance": {
        "description": "발화에서 드러나는 심리적 거리감.",
        "labels": SOCIAL_DISTANCE_LABELS,
    },
    "power_relation": {
        "description": "한국어 높임말과 발화 선택에 영향을 주는 위계.",
        "labels": POWER_RELATION_LABELS,
    },
}


MULTIHOP_QUERY_TEMPLATES: dict[str, dict[str, Any]] = {
    "intent_from_context": {
        "description": "사건과 stance를 연결해 숨은 의도를 추론한다.",
        "required_signals": ("speechAct", "stance"),
        "question_template": "이 발화의 숨은 의도는 무엇인가?",
        "answer_shape": "화자의 숨은 의도 한 문장",
    },
    "indirect_speech_act": {
        "description": "완곡한 표현이 실제로는 거절/불만/부담 전가인지 해석한다.",
        "required_signals": ("indirectness", "speechAct"),
        "question_template": "이 발화는 직접 의미보다 어떤 간접 화행으로 읽혀야 하는가?",
        "answer_shape": "간접 화행 라벨 + 짧은 근거",
    },
    "ellipsis_recovery": {
        "description": "생략된 주체나 대상을 복원한다.",
        "required_signals": ("ellipsisSubject", "ellipsisTarget"),
        "question_template": "이 발화에서 생략된 주체/대상은 무엇인가?",
        "answer_shape": "복원된 주체/대상",
    },
    "stance_grounded_response": {
        "description": "사건과 태도를 함께 보고 다음 적절한 반응을 생성한다.",
        "required_signals": ("stance", "emotionCue"),
        "question_template": "이 태도에 맞는 다음 반응은 무엇인가?",
        "answer_shape": "태도 일관성이 있는 응답 한 문장",
    },
    "relation_aware_response": {
        "description": "사회적 거리와 위계를 반영해 같은 의미를 다른 톤으로 생성한다.",
        "required_signals": ("politeness", "speechAct"),
        "question_template": "이 관계와 말투를 유지하면서 적절한 응답은 무엇인가?",
        "answer_shape": "관계/말투 일관 응답",
    },
    "emotion_cascade": {
        "description": "사건의 효과와 감정 단서를 함께 보고 감정 전이를 추론한다.",
        "required_signals": ("emotionCue", "certainty"),
        "question_template": "이 대화에서 감정은 어떤 방향으로 전이되고 있는가?",
        "answer_shape": "감정 전이 설명 한 문장",
    },
}


def build_pragmatic_schema_snapshot() -> dict[str, Any]:
    return {
        "signal_types": PRAGMATIC_SIGNAL_TYPES,
        "signal_taxonomy": PRAGMATIC_SIGNAL_TAXONOMY,
        "social_taxonomy": SOCIAL_RELATION_TAXONOMY,
        "multihop_query_types": MULTIHOP_QUERY_TYPES,
        "multihop_query_templates": MULTIHOP_QUERY_TEMPLATES,
    }

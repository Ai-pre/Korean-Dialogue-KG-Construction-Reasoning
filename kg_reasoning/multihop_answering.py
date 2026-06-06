from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


HOP_RELATION_RE = re.compile(r"--(?P<relation>[A-Za-z_]+)-->")
LABEL_MARKER = " | label="


@dataclass(slots=True)
class PromptHop:
    relation: str
    label: str
    evidence: str = ""


def rewrite_multihop_training_answer(row: dict[str, Any]) -> dict[str, Any]:
    rewritten = copy.deepcopy(row)
    messages = rewritten.get("messages", [])
    if len(messages) < 3:
        return rewritten

    prompt = str(messages[1].get("content", ""))
    old_answer = str(messages[2].get("content", ""))
    query_type = str(rewritten.get("metadata", {}).get("query_type", ""))
    hops = parse_prompt_hops(prompt)
    answer = build_strategy_answer(query_type=query_type, hops=hops, old_answer=old_answer)
    if not answer:
        return rewritten

    messages[2]["content"] = answer
    metadata = rewritten.setdefault("metadata", {})
    metadata["answer_style"] = "strategy_v2"
    return rewritten


def parse_prompt_hops(prompt: str) -> list[PromptHop]:
    hops: list[PromptHop] = []
    current: PromptHop | None = None
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and "--" in stripped:
            relation_match = HOP_RELATION_RE.search(stripped)
            if relation_match is None:
                current = None
                continue
            label = ""
            if LABEL_MARKER in stripped:
                label = stripped.split(LABEL_MARKER, 1)[1].strip()
            current = PromptHop(relation=relation_match.group("relation"), label=label)
            hops.append(current)
            continue
        if current is not None and stripped.startswith("evidence:"):
            current.evidence = stripped.split("evidence:", 1)[1].strip()
    return hops


def build_strategy_answer(*, query_type: str, hops: list[PromptHop], old_answer: str = "") -> str:
    if query_type == "indirect_speech_act":
        hop = find_hop(hops, "indirectness") or first_content_hop(hops)
        return answer_indirect(hop)
    if query_type == "relation_aware_response":
        primary = find_hop(hops, "speechAct", "stance", "politeness") or first_content_hop(hops)
        relation = find_hop(hops, "conditioned_by_relation")
        return answer_relation(primary, relation)
    if query_type == "stance_grounded_response":
        hop = find_hop(hops, "stance", "speechAct", "emotionCue") or first_content_hop(hops)
        return answer_stance(hop)
    if query_type == "emotion_cascade":
        hop = find_hop(hops, "emotionCue", "certainty", "stance") or first_content_hop(hops)
        return answer_emotion(hop)
    if query_type == "ellipsis_recovery":
        hop = find_hop(hops, "ellipsisSubject", "ellipsisTarget") or first_content_hop(hops)
        return answer_ellipsis(hop)
    if query_type == "intent_from_context":
        primary = find_hop(hops, "stance", "speechAct", "facework", "listenerPressure") or first_content_hop(hops)
        event = find_hop(hops, "grounds_event")
        return answer_intent(primary, event)
    return old_answer


def find_hop(hops: list[PromptHop], *relations: str) -> PromptHop | None:
    relation_set = set(relations)
    for hop in hops:
        if hop.relation in relation_set:
            return hop
    return None


def first_content_hop(hops: list[PromptHop]) -> PromptHop:
    return hops[0] if hops else PromptHop(relation="unknown", label="unknown", evidence="")


def answer_indirect(hop: PromptHop) -> str:
    return (
        f"{quote_evidence(hop)}는 {label_ko(hop.label)} 신호가 강해서 겉말보다 숨은 부담, 거절, 망설임을 먼저 읽어야 한다. "
        "응답은 이유를 캐묻기보다 상대의 선택권을 인정하고, 가능하면 다음 기회나 낮은 부담의 대안을 열어 주는 방향이 자연스럽다."
    )


def answer_relation(primary: PromptHop, relation: PromptHop | None) -> str:
    relation_text = "화자 관계"
    if relation is not None and relation.label:
        relation_text = f"{label_ko(relation.label)} 관계"
    return (
        f"{quote_evidence(primary)}에는 {signal_ko(primary.relation)}={label_ko(primary.label)} 단서가 있고, "
        f"이 단서는 {relation_text} 안에서 해석해야 한다. "
        f"따라서 답변은 {strategy_for(primary.relation, primary.label)} "
        "친밀하더라도 단정하거나 압박하지 않고, 관계를 보존하는 말투로 이어 가는 것이 좋다."
    )


def answer_stance(hop: PromptHop) -> str:
    return (
        f"{quote_evidence(hop)}의 핵심은 {signal_ko(hop.relation)}={label_ko(hop.label)}이다. "
        f"다음 반응은 {strategy_for(hop.relation, hop.label)} "
        "즉, KG 단서를 그대로 나열하기보다 현재 태도에 맞는 응답 강도와 방향을 조절해야 한다."
    )


def answer_emotion(hop: PromptHop) -> str:
    return (
        f"{quote_evidence(hop)}에서 드러나는 감정 흐름은 {label_ko(hop.label)} 쪽이다. "
        f"응답은 {strategy_for(hop.relation, hop.label)} "
        "감정 확인을 먼저 하고 해결책이나 조언은 뒤에 붙이는 순서가 안전하다."
    )


def answer_ellipsis(hop: PromptHop) -> str:
    return (
        f"{quote_evidence(hop)}에서는 생략된 정보가 {label_ko(hop.label)} 축으로 복원된다. "
        "응답은 빠진 주체나 대상을 노골적으로 캐묻기보다, 앞뒤 맥락에서 공유된 대상으로 자연스럽게 보충해 이어 가는 편이 좋다."
    )


def answer_intent(primary: PromptHop, event: PromptHop | None) -> str:
    event_text = f" 사건 맥락({short_text(event.label or event.evidence)})까지 고려하면" if event is not None else ""
    return (
        f"{quote_evidence(primary)}의 {signal_ko(primary.relation)}={label_ko(primary.label)} 단서와{event_text} "
        f"화자의 의도는 단순 정보 전달보다 관계 조율에 가깝다. 응답은 {strategy_for(primary.relation, primary.label)}"
    )


def quote_evidence(hop: PromptHop) -> str:
    text = short_text(hop.evidence or hop.label or "해당 발화")
    return f"'{text}'"


def short_text(text: str, *, max_chars: int = 44) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def signal_ko(signal_type: str) -> str:
    names = {
        "speechAct": "화행",
        "stance": "태도",
        "politeness": "말투",
        "indirectness": "간접화행",
        "emotionCue": "감정 단서",
        "certainty": "확신성",
        "ellipsisSubject": "생략 주체",
        "ellipsisTarget": "생략 대상",
        "facework": "체면 조율",
        "listenerPressure": "청자 압박",
    }
    return names.get(signal_type, signal_type)


def label_ko(label: str) -> str:
    names = {
        "answer": "응답",
        "question": "질문",
        "request": "요청",
        "advice": "조언",
        "complaint": "불만",
        "agreement": "동의",
        "disagreement": "반대",
        "empathy": "공감",
        "self_disclosure": "자기노출",
        "indirect_refusal": "완곡한 거절",
        "softened": "완충된 표현",
        "implicit": "암시적 표현",
        "ambiguous": "모호한 표현",
        "sarcastic": "비꼼",
        "supportive": "지지적 태도",
        "frustrated": "답답함/불만",
        "hesitant_negative": "망설이는 부정",
        "hesitant_positive": "조심스러운 긍정",
        "playful": "장난스러운 태도",
        "neutral": "중립",
        "polite": "공손한 말투",
        "deferential": "격식 있는 높임",
        "casual": "친근한 말투",
        "plain": "평서체",
        "mixed": "혼합된 말투",
        "sadness": "슬픔",
        "frustration": "답답함",
        "playfulness": "장난기",
        "relief": "안도감",
        "awkwardness": "어색함",
        "friend": "친구",
        "family": "가족",
        "service": "서비스/상담",
        "acquaintance": "지인",
        "stranger": "낯선 관계",
    }
    return names.get(label, label)


def strategy_for(signal_type: str, label: str) -> str:
    if signal_type == "speechAct":
        return {
            "question": "질문 의도를 먼저 받아 주고 필요한 정보를 짧게 확인하는 방향이 좋다.",
            "request": "요청을 수락하거나 조율하되 가능한 조건을 분명히 말하는 편이 좋다.",
            "advice": "조언의 압박감을 낮추고 제안형으로 돌려주는 편이 좋다.",
            "complaint": "불만의 원인을 인정한 뒤 방어적으로 맞받아치지 않는 편이 좋다.",
            "agreement": "동의를 이어 받되 대화를 확장할 근거를 하나 덧붙이는 편이 좋다.",
            "disagreement": "반대 의사를 직접 부딪히기보다 이유를 완충해서 설명하는 편이 좋다.",
            "empathy": "감정 확인을 먼저 하고 해결책은 뒤에 붙이는 편이 좋다.",
            "self_disclosure": "자기노출을 평가하지 말고 공감과 후속 질문으로 이어 가는 편이 좋다.",
            "indirect_refusal": "거절 신호를 존중해 부담을 낮추고 대안을 열어 두는 편이 좋다.",
        }.get(label, "발화 의도를 먼저 확인하고 그 의도에 맞춰 응답 방향을 좁히는 편이 좋다.")
    if signal_type == "stance":
        return {
            "supportive": "지지적 태도를 유지하면서 상대가 더 말할 수 있게 받아 주는 편이 좋다.",
            "frustrated": "짜증이나 피로를 반박하지 말고 원인을 인정한 뒤 완충하는 편이 좋다.",
            "hesitant_negative": "부정적 신호를 직접 압박하지 말고 거절 여지를 안전하게 만들어 주는 편이 좋다.",
            "hesitant_positive": "조심스러운 긍정을 확정으로 몰지 말고 확인 질문으로 이어 가는 편이 좋다.",
            "playful": "장난스러운 결을 살리되 상대를 깎아내리지 않는 선에서 받아치는 편이 좋다.",
        }.get(label, "태도를 과장하지 말고 현재 정서의 강도에 맞춰 응답하는 편이 좋다.")
    if signal_type == "politeness":
        return {
            "polite": "존댓말의 거리를 유지하면서 너무 딱딱하지 않게 답하는 편이 좋다.",
            "deferential": "상대의 체면과 위계를 더 강하게 존중하는 말투가 필요하다.",
            "casual": "친근한 말투를 유지하되 무례하게 단정하지 않는 편이 좋다.",
            "plain": "담백하게 답하되 관계가 애매하면 표현을 한 단계 부드럽게 만드는 편이 좋다.",
        }.get(label, "말투의 격식을 현재 관계에 맞게 조절하는 편이 좋다.")
    if signal_type == "emotionCue":
        return {
            "sadness": "슬픔을 바로 해결하려 들기보다 먼저 인정하고 곁에 있다는 신호를 주는 편이 좋다.",
            "frustration": "분노나 답답함을 낮추기 위해 원인을 요약하고 다음 행동을 작게 제안하는 편이 좋다.",
            "playfulness": "가벼운 농담을 받아 주면서 안전한 분위기를 유지하는 편이 좋다.",
            "relief": "안도감을 확인하고 긍정적 흐름을 이어 주는 편이 좋다.",
            "awkwardness": "어색함을 키우지 않도록 짧고 부드럽게 전환하는 편이 좋다.",
        }.get(label, "감정 단서를 먼저 반영하고 해결책은 뒤에 제안하는 편이 좋다.")
    if signal_type in {"indirectness", "ellipsisSubject", "ellipsisTarget"}:
        return "숨은 의미를 직접 단정하기보다 상대가 편하게 확인할 수 있게 완충하는 편이 좋다."
    return "단서를 그대로 나열하기보다 대화 맥락에 맞는 응답 전략으로 바꾸는 편이 좋다."

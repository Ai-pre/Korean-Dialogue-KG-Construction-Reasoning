from __future__ import annotations

from collections import defaultdict

from kg_reasoning.korean_pragmatics import MULTIHOP_QUERY_TEMPLATES
from kg_reasoning.schema import (
    DialogueRecord,
    EventRecord,
    MultiHopQueryRecord,
    PragmaticSignalRecord,
    ReasoningHopRecord,
    SpeakerRelationRecord,
)


def build_reasoning_hops(
    *,
    dialogue_id: str,
    events: list[EventRecord],
    pragmatic_signals: list[PragmaticSignalRecord],
    speaker_relations: list[SpeakerRelationRecord],
) -> list[ReasoningHopRecord]:
    event_map = {event.event_id: event for event in events if event.dialogue_id == dialogue_id}
    relation_map: dict[frozenset[str], SpeakerRelationRecord] = {}
    for relation in speaker_relations:
        if relation.dialogue_id != dialogue_id:
            continue
        relation_map[frozenset((relation.speaker_a, relation.speaker_b))] = relation

    hops: list[ReasoningHopRecord] = []
    for signal in pragmatic_signals:
        if signal.dialogue_id != dialogue_id:
            continue
        signal_id = f"{signal.utterance_id}:{signal.signal_type}"
        hops.append(
            ReasoningHopRecord(
                dialogue_id=dialogue_id,
                hop_id=f"{signal_id}:self",
                source_kind="utterance",
                source_id=signal.utterance_id,
                relation=signal.signal_type,
                target_kind="pragmatic",
                target_id=signal_id,
                evidence_text=signal.evidence_text,
                score=signal.confidence,
                attributes={"label": signal.label},
            )
        )
        if signal.linked_event_id and signal.linked_event_id in event_map:
            event = event_map[signal.linked_event_id]
            hops.append(
                ReasoningHopRecord(
                    dialogue_id=dialogue_id,
                    hop_id=f"{signal_id}:{event.event_id}",
                    source_kind="pragmatic",
                    source_id=signal_id,
                    relation="grounds_event",
                    target_kind="event",
                    target_id=event.event_id,
                    evidence_text=event.event_text,
                    score=min(1.0, max(signal.confidence, 0.5)),
                    attributes={"event_text": event.event_text, "event_cause": event.event_cause},
                )
            )
        if signal.target_speaker:
            relation = relation_map.get(frozenset((signal.speaker, signal.target_speaker)))
            if relation is not None and relation.relation_label != "unknown":
                relation_id = f"{relation.speaker_a}:{relation.speaker_b}"
                hops.append(
                    ReasoningHopRecord(
                        dialogue_id=dialogue_id,
                        hop_id=f"{signal_id}:{relation_id}",
                        source_kind="pragmatic",
                        source_id=signal_id,
                        relation="conditioned_by_relation",
                        target_kind="speaker_relation",
                        target_id=relation_id,
                        evidence_text=relation.relation_label,
                        score=min(1.0, max(signal.confidence, relation.confidence)),
                        attributes={
                            "relation_label": relation.relation_label,
                            "social_distance": relation.social_distance,
                            "power_relation": relation.power_relation,
                        },
                    )
                )
    return hops


def _question_for_signal(signal: PragmaticSignalRecord, event: EventRecord | None, relation: SpeakerRelationRecord | None) -> tuple[str, str, list[str]]:
    if signal.signal_type == "indirectness":
        return (
            "indirect_speech_act",
            "이 발화는 직접 의미보다 어떤 간접 화행으로 읽혀야 하는가?",
            ["indirectness", "speechAct"],
        )
    if signal.signal_type in {"ellipsisSubject", "ellipsisTarget"}:
        return (
            "ellipsis_recovery",
            "이 발화에서 생략된 주체나 대상은 무엇인가?",
            ["ellipsisSubject", "ellipsisTarget"],
        )
    if relation is not None and signal.signal_type in {"speechAct", "politeness", "stance"}:
        return (
            "relation_aware_response",
            "이 관계와 말투를 유지하면서 어떤 응답이 가장 자연스러운가?",
            ["politeness", "speechAct", "social_relation"],
        )
    if event is not None and signal.signal_type == "stance":
        return (
            "intent_from_context",
            "이 사건 맥락까지 포함하면 화자의 숨은 의도는 무엇인가?",
            ["stance", "event"],
        )
    if signal.signal_type in {"emotionCue", "certainty"}:
        return (
            "emotion_cascade",
            "이 대화에서 감정은 어떤 방향으로 전이되고 있는가?",
            ["emotionCue", "certainty"],
        )
    return (
        "stance_grounded_response",
        "이 태도에 맞는 다음 반응은 무엇인가?",
        ["stance", "emotionCue"],
    )


def _answer_for_signal(signal: PragmaticSignalRecord, event: EventRecord | None, relation: SpeakerRelationRecord | None) -> str:
    if signal.signal_type == "indirectness":
        return f"{signal.speaker}의 발화는 '{signal.label}'한 간접 화행으로 해석된다."
    if signal.signal_type in {"ellipsisSubject", "ellipsisTarget"}:
        return f"생략된 정보는 '{signal.label}' 축으로 복원하는 것이 자연스럽다."
    if relation is not None:
        return (
            f"{relation.relation_label} 관계와 {relation.power_relation} 위계를 유지하면서 "
            f"'{signal.label}' 태도를 반영한 응답이 적절하다."
        )
    if event is not None:
        return f"사건 '{event.event_text}' 맥락에서 화자의 숨은 의도는 '{signal.label}' 태도로 정리된다."
    return f"이 발화는 '{signal.label}' 태도를 유지하는 응답이 필요하다."


def build_multihop_queries(
    *,
    dialogue: DialogueRecord,
    events: list[EventRecord],
    pragmatic_signals: list[PragmaticSignalRecord],
    speaker_relations: list[SpeakerRelationRecord],
    max_queries: int = 8,
) -> tuple[list[ReasoningHopRecord], list[MultiHopQueryRecord]]:
    dialogue_events = [event for event in events if event.dialogue_id == dialogue.dialogue_id]
    dialogue_signals = [signal for signal in pragmatic_signals if signal.dialogue_id == dialogue.dialogue_id]
    dialogue_relations = [relation for relation in speaker_relations if relation.dialogue_id == dialogue.dialogue_id]

    event_map = {event.event_id: event for event in dialogue_events}
    relation_map: dict[frozenset[str], SpeakerRelationRecord] = {
        frozenset((relation.speaker_a, relation.speaker_b)): relation for relation in dialogue_relations
    }
    hops = build_reasoning_hops(
        dialogue_id=dialogue.dialogue_id,
        events=dialogue_events,
        pragmatic_signals=dialogue_signals,
        speaker_relations=dialogue_relations,
    )
    hop_ids_by_source: dict[str, list[str]] = defaultdict(list)
    for hop in hops:
        hop_ids_by_source[hop.source_id].append(hop.hop_id)

    query_candidates = [
        signal
        for signal in sorted(
            dialogue_signals,
            key=lambda item: _signal_priority(item, relation_map),
            reverse=True,
        )
        if _signal_priority(signal=signal, relation_map=relation_map) > 0
    ]

    queries: list[MultiHopQueryRecord] = []
    for index, signal in enumerate(query_candidates[:max_queries], start=1):
        event = event_map.get(signal.linked_event_id) if signal.linked_event_id else None
        relation = None
        if signal.target_speaker:
            relation = relation_map.get(frozenset((signal.speaker, signal.target_speaker)))
            if relation is not None and relation.relation_label == "unknown":
                relation = None
        query_type, question, reasoning_focus = _localized_question_for_signal(signal, event, relation)
        template = MULTIHOP_QUERY_TEMPLATES[query_type]
        signal_id = f"{signal.utterance_id}:{signal.signal_type}"
        supporting_hop_ids = []
        self_hop_id = f"{signal_id}:self"
        if self_hop_id in {hop.hop_id for hop in hops}:
            supporting_hop_ids.append(self_hop_id)
        supporting_hop_ids.extend(hop_ids_by_source.get(signal_id, []))
        for hop_id in hop_ids_by_source.get(signal.utterance_id, []):
            if hop_id not in supporting_hop_ids:
                supporting_hop_ids.append(hop_id)
        supporting_hop_ids = supporting_hop_ids[:4]
        queries.append(
            MultiHopQueryRecord(
                dialogue_id=dialogue.dialogue_id,
                query_id=f"{dialogue.dialogue_id}:Q{index}",
                query_type=query_type,
                question=question,
                answer=_localized_answer_for_signal(signal, event, relation),
                supporting_hop_ids=supporting_hop_ids,
                target_utterance_id=signal.utterance_id,
                difficulty="2-hop" if len(supporting_hop_ids) <= 2 else "3-hop",
                reasoning_focus=reasoning_focus,
                metadata={
                    "template_description": template["description"],
                    "signal_label": signal.label,
                    "signal_type": signal.signal_type,
                },
            )
        )
    return hops, queries


def _signal_priority(
    signal: PragmaticSignalRecord,
    relation_map: dict[frozenset[str], SpeakerRelationRecord],
) -> float:
    label = signal.label
    relation = _relation_for_signal(signal, relation_map)
    has_informative_relation = relation is not None and relation.relation_label != "unknown"

    if signal.signal_type == "indirectness":
        base = 100.0
    elif signal.signal_type == "speechAct":
        if label == "answer":
            return 0.0
        base = 92.0
    elif signal.signal_type == "emotionCue":
        base = 86.0
    elif signal.signal_type == "facework":
        base = 82.0
    elif signal.signal_type == "listenerPressure":
        base = 76.0
    elif signal.signal_type == "stance":
        if label == "neutral":
            return 0.0
        base = 72.0
    elif signal.signal_type in {"ellipsisSubject", "ellipsisTarget"}:
        base = 64.0
    elif signal.signal_type == "certainty":
        base = 58.0
    elif signal.signal_type == "humor":
        base = 54.0
    elif signal.signal_type == "politeness":
        if label in {"plain", "mixed"} or not has_informative_relation:
            return 0.0
        base = 48.0
    else:
        base = 25.0

    if signal.linked_event_id:
        base += 8.0
    if has_informative_relation:
        base += 5.0
    return base + signal.confidence


def _relation_for_signal(
    signal: PragmaticSignalRecord,
    relation_map: dict[frozenset[str], SpeakerRelationRecord],
) -> SpeakerRelationRecord | None:
    if not signal.target_speaker:
        return None
    return relation_map.get(frozenset((signal.speaker, signal.target_speaker)))


def _localized_question_for_signal(
    signal: PragmaticSignalRecord,
    event: EventRecord | None,
    relation: SpeakerRelationRecord | None,
) -> tuple[str, str, list[str]]:
    if signal.signal_type == "indirectness":
        return (
            "indirect_speech_act",
            "이 발화는 겉으로 한 말과 달리 어떤 간접 화행으로 해석해야 하는가?",
            ["indirectness", "speechAct"],
        )
    if signal.signal_type in {"ellipsisSubject", "ellipsisTarget"}:
        return (
            "ellipsis_recovery",
            "이 발화에서 생략된 주체나 대상은 무엇으로 복원하는 것이 자연스러운가?",
            ["ellipsisSubject", "ellipsisTarget"],
        )
    if relation is not None and signal.signal_type in {"speechAct", "politeness", "stance"}:
        return (
            "relation_aware_response",
            "화자 관계와 말투를 함께 고려하면 어떤 응답 방향이 가장 자연스러운가?",
            ["politeness", "speechAct", "social_relation"],
        )
    if event is not None and signal.signal_type in {"speechAct", "stance", "facework", "listenerPressure"}:
        return (
            "intent_from_context",
            "사건 맥락과 발화 태도를 함께 보면 화자의 숨은 의도는 무엇인가?",
            ["stance", "event"],
        )
    if signal.signal_type in {"emotionCue", "certainty"}:
        return (
            "emotion_cascade",
            "대화의 감정 단서는 어떤 방향으로 이어지고 있는가?",
            ["emotionCue", "certainty"],
        )
    return (
        "stance_grounded_response",
        "이 발화의 태도에 맞는 다음 반응은 무엇인가?",
        ["stance", "emotionCue"],
    )


def _localized_answer_for_signal(
    signal: PragmaticSignalRecord,
    event: EventRecord | None,
    relation: SpeakerRelationRecord | None,
) -> str:
    evidence = _short_evidence(signal.evidence_text)
    signal_name = _signal_name_for_answer(signal.signal_type)
    label_name = _label_name_for_answer(signal.label)
    if signal.signal_type == "indirectness":
        return (
            f"'{evidence}'는 {label_name} 단서가 강해서 문자 그대로의 정보 전달보다 숨은 부담이나 망설임을 읽어야 한다. "
            "따라서 답변은 상대를 몰아붙이지 말고, 거절 가능성을 인정하면서 선택지를 열어 주는 방향이 자연스럽다."
        )
    if signal.signal_type in {"ellipsisSubject", "ellipsisTarget"}:
        return (
            f"'{evidence}'에서 빠진 대상은 {label_name} 쪽으로 복원하는 것이 앞뒤 흐름과 맞다. "
            "응답할 때는 생략된 주체나 대상을 직접 보충해 주되, 대화자가 이미 공유한 맥락처럼 자연스럽게 이어 가는 편이 좋다."
        )
    if relation is not None:
        relation_context = _relation_context_for_answer(relation)
        strategy = _strategy_for_answer(signal.signal_type, signal.label)
        return (
            f"'{evidence}'에는 {signal_name}={label_name} 단서가 있고, 두 화자는 {relation_context}. "
            f"그래서 응답은 {strategy} 특히 관계가 깨지지 않도록 말투의 강도를 조절하는 것이 핵심이다."
        )
    if event is not None:
        strategy = _strategy_for_answer(signal.signal_type, signal.label)
        return (
            f"'{evidence}'는 사건 '{event.event_text}'와 이어지며, 이 발화의 {signal_name}은 {label_name}로 해석된다. "
            f"응답은 사건 맥락을 먼저 짚고 {strategy}"
        )
    strategy = _strategy_for_answer(signal.signal_type, signal.label)
    return f"'{evidence}'에는 {signal_name}={label_name} 단서가 있으므로, 응답은 {strategy}"


def _short_evidence(text: str, *, max_chars: int = 42) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def _relation_context_for_answer(relation: SpeakerRelationRecord) -> str:
    relation_label = _label_name_for_answer(relation.relation_label)
    distance = {
        "close": "가까운 거리",
        "neutral": "중립적인 거리",
        "distant": "조금 거리가 있는 관계",
    }.get(relation.social_distance, f"{relation.social_distance} 거리")
    power = {
        "equal": "대등한 위계",
        "speaker_a_higher": "한쪽이 더 우위인 위계",
        "speaker_b_higher": "상대가 더 우위인 위계",
        "unknown": "위계가 뚜렷하지 않은 관계",
    }.get(relation.power_relation, f"{relation.power_relation} 위계")
    return f"{relation_label}에 가까운 관계이고 {distance}, {power}에 놓여 있다"


def _strategy_for_answer(signal_type: str, label: str) -> str:
    if signal_type == "speechAct":
        return {
            "question": "질문 의도를 먼저 받아 주고 필요한 정보를 짧게 확인하는 방향이 좋다.",
            "request": "요청을 수락하거나 조심스럽게 조율하되, 가능한 조건을 분명히 말하는 편이 좋다.",
            "advice": "조언의 압박감을 낮추고 상대가 선택할 수 있게 제안형으로 돌려주는 편이 좋다.",
            "complaint": "불만의 원인을 인정한 뒤 방어적으로 맞받아치지 않는 편이 좋다.",
            "agreement": "동의를 이어 받되 대화를 확장할 수 있는 근거를 하나 덧붙이는 편이 좋다.",
            "disagreement": "반대 의사를 직접 부딪히기보다 이유를 완충해서 설명하는 편이 좋다.",
            "empathy": "감정 확인을 먼저 하고 해결책은 뒤에 붙이는 편이 좋다.",
            "self_disclosure": "자기노출을 평가하지 말고 공감과 후속 질문으로 이어 가는 편이 좋다.",
            "indirect_refusal": "완곡한 거절 신호를 존중해 부담을 낮추고 대안을 열어 두는 편이 좋다.",
        }.get(label, "발화 의도를 먼저 확인하고 그 의도에 맞춰 응답 방향을 좁히는 편이 좋다.")
    if signal_type == "stance":
        return {
            "supportive": "지지적 태도를 유지하면서 상대의 감정을 더 말할 수 있게 받아 주는 편이 좋다.",
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
            "playfulness": "가벼운 농담을 받아 주면서 대화의 안전한 분위기를 유지하는 편이 좋다.",
            "relief": "안도감을 확인하고 긍정적 흐름을 이어 주는 편이 좋다.",
            "awkwardness": "어색함을 키우지 않도록 짧고 부드럽게 전환하는 편이 좋다.",
        }.get(label, "감정 단서를 먼저 반영하고 해결책은 뒤에 제안하는 편이 좋다.")
    if signal_type in {"ellipsisSubject", "ellipsisTarget"}:
        return "생략된 대상을 복원한 뒤 그 맥락을 기준으로 짧게 이어 주는 편이 좋다."
    if signal_type == "certainty":
        return "확신의 정도를 그대로 반영해 단정하거나 유보하는 강도를 맞추는 편이 좋다."
    return "단서를 그대로 나열하기보다 대화 맥락에 맞는 응답 전략으로 바꾸는 편이 좋다."


def _label_name_for_answer(label: str) -> str:
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


def _signal_name_for_answer(signal_type: str) -> str:
    names = {
        "speechAct": "화행",
        "stance": "태도",
        "politeness": "말투",
        "indirectness": "간접화행 단서",
        "emotionCue": "감정 단서",
        "listenerPressure": "응답 압력",
        "facework": "체면 조절 단서",
        "humor": "유머 단서",
        "ellipsisSubject": "생략 주체 단서",
        "ellipsisTarget": "생략 대상 단서",
        "agreementShift": "동의 변화 단서",
        "certainty": "확신도 단서",
    }
    return names.get(signal_type, "화용론 단서")

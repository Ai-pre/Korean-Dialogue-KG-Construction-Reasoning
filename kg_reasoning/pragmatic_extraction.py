from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations

from kg_reasoning.schema import (
    DialogueRecord,
    EventRecord,
    PragmaticSignalRecord,
    SpeakerRelationRecord,
    Utterance,
)


TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
POLITE_ENDINGS = ("요", "죠", "네요", "습니다", "습니까", "합니다", "입니다", "드립니다")
QUESTION_MARKERS = ("왜", "어때", "어떻게", "언제", "누구", "괜찮", "할까", "일까", "나요")
AGREEMENT_MARKERS = ("맞아", "맞아요", "그러게", "그니까", "인정", "좋아", "오케이", "ㅇㅋ")
DISAGREEMENT_MARKERS = ("아니", "싫", "별로", "안 돼", "안되", "글쎄", "무리")
EMPATHY_MARKERS = ("힘들", "고생", "속상", "괜찮아", "위로", "걱정", "ㅠ")
ADVICE_MARKERS = ("해봐", "해보자", "하면 돼", "추천", "가자", "말해", "쉬어")
REQUEST_MARKERS = ("해줘", "부탁", "도와", "같이", "줄래", "하자", "해주", "싶어", "하고 싶", "해도 돼")
REFUSAL_MARKERS = ("좀 그렇", "어렵", "어려울", "어려워", "힘들 것", "안 될", "부담", "곤란", "글쎄")
COMPLAINT_MARKERS = ("짜증", "화나", "싫어", "너무해", "불편", "빡", "답답", "왜 그래", "아파", "아프")
SELF_DISCLOSURE_MARKERS = ("나는", "난 ", "내가", "나도", "저는", "제가")
SOFTENING_MARKERS = ("좀", "...", "그렇긴", "괜찮긴", "아마", "일단", "굳이")
IMPLICIT_MARKERS = ("그거", "그건", "그게", "그렇게", "그 당시", "그 말")
UNCERTAIN_MARKERS = ("아마", "같아", "듯", "글쎄", "모르", "확실", "일 수도")
FAMILY_MARKERS = ("엄마", "아빠", "부모", "동생", "언니", "오빠", "형", "누나", "할머니", "할아버지")
SERVICE_MARKERS = ("주문", "예약", "문의", "상담", "고객", "배송", "결제")


def extract_pragmatics(
    dialogues: list[DialogueRecord],
    *,
    provider: str = "rule",
) -> tuple[list[PragmaticSignalRecord], list[SpeakerRelationRecord]]:
    if provider != "rule":
        raise ValueError(f"Unsupported pragmatic extraction provider: {provider}")

    signals: list[PragmaticSignalRecord] = []
    relations: list[SpeakerRelationRecord] = []
    for dialogue in dialogues:
        signals.extend(extract_pragmatic_signals(dialogue=dialogue, provider=provider))
        relations.extend(extract_speaker_relations(dialogue=dialogue, provider=provider))
    return signals, relations


def extract_pragmatic_signals(
    *,
    dialogue: DialogueRecord,
    provider: str = "rule",
) -> list[PragmaticSignalRecord]:
    records: list[PragmaticSignalRecord] = []
    utterances = dialogue.utterances
    for index, utterance in enumerate(utterances):
        utterance_id = stable_utterance_id(dialogue.dialogue_id, utterance, index)
        target_speaker = infer_target_speaker(utterances, index)
        text = utterance.text.strip()
        if not text:
            continue

        speech_act, speech_confidence = infer_speech_act(text)
        records.append(
            make_signal(
                dialogue=dialogue,
                utterance=utterance,
                utterance_id=utterance_id,
                target_speaker=target_speaker,
                signal_type="speechAct",
                label=speech_act,
                evidence_text=text,
                confidence=speech_confidence,
                provider=provider,
            )
        )

        stance, stance_confidence = infer_stance(text, speech_act)
        records.append(
            make_signal(
                dialogue=dialogue,
                utterance=utterance,
                utterance_id=utterance_id,
                target_speaker=target_speaker,
                signal_type="stance",
                label=stance,
                evidence_text=text,
                confidence=stance_confidence,
                provider=provider,
            )
        )

        politeness, politeness_confidence = infer_politeness(text)
        records.append(
            make_signal(
                dialogue=dialogue,
                utterance=utterance,
                utterance_id=utterance_id,
                target_speaker=target_speaker,
                signal_type="politeness",
                label=politeness,
                evidence_text=text[-24:],
                confidence=politeness_confidence,
                provider=provider,
            )
        )

        indirectness, indirectness_confidence = infer_indirectness(text, speech_act)
        if indirectness != "direct":
            records.append(
                make_signal(
                    dialogue=dialogue,
                    utterance=utterance,
                    utterance_id=utterance_id,
                    target_speaker=target_speaker,
                    signal_type="indirectness",
                    label=indirectness,
                    evidence_text=text,
                    confidence=indirectness_confidence,
                    provider=provider,
                )
            )

        emotion = None if speech_act == "advice" and "아프면" in text else infer_emotion_cue(text)
        if emotion is not None:
            label, confidence = emotion
            records.append(
                make_signal(
                    dialogue=dialogue,
                    utterance=utterance,
                    utterance_id=utterance_id,
                    target_speaker=target_speaker,
                    signal_type="emotionCue",
                    label=label,
                    evidence_text=text,
                    confidence=confidence,
                    provider=provider,
                )
            )

        certainty = infer_certainty(text)
        if certainty is not None:
            label, confidence = certainty
            records.append(
                make_signal(
                    dialogue=dialogue,
                    utterance=utterance,
                    utterance_id=utterance_id,
                    target_speaker=target_speaker,
                    signal_type="certainty",
                    label=label,
                    evidence_text=text,
                    confidence=confidence,
                    provider=provider,
                )
            )

        ellipsis = infer_ellipsis(text, speech_act)
        for signal_type, label, confidence in ellipsis:
            records.append(
                make_signal(
                    dialogue=dialogue,
                    utterance=utterance,
                    utterance_id=utterance_id,
                    target_speaker=target_speaker,
                    signal_type=signal_type,
                    label=label,
                    evidence_text=text,
                    confidence=confidence,
                    provider=provider,
                )
            )

        if has_any(text, ("ㅋㅋ", "ㅎㅎ", "키키", "농담")):
            records.append(
                make_signal(
                    dialogue=dialogue,
                    utterance=utterance,
                    utterance_id=utterance_id,
                    target_speaker=target_speaker,
                    signal_type="humor",
                    label="light_banter",
                    evidence_text=text,
                    confidence=0.72,
                    provider=provider,
                )
            )

        if speech_act in {"indirect_refusal", "complaint"} or indirectness in {"softened", "implicit"}:
            records.append(
                make_signal(
                    dialogue=dialogue,
                    utterance=utterance,
                    utterance_id=utterance_id,
                    target_speaker=target_speaker,
                    signal_type="facework",
                    label="face_saving" if speech_act == "indirect_refusal" else "self_protective",
                    evidence_text=text,
                    confidence=0.67,
                    provider=provider,
                )
            )

        if speech_act == "request":
            records.append(
                make_signal(
                    dialogue=dialogue,
                    utterance=utterance,
                    utterance_id=utterance_id,
                    target_speaker=target_speaker,
                    signal_type="listenerPressure",
                    label="medium",
                    evidence_text=text,
                    confidence=0.62,
                    provider=provider,
                )
            )
    return records


def extract_speaker_relations(
    *,
    dialogue: DialogueRecord,
    provider: str = "rule",
) -> list[SpeakerRelationRecord]:
    speakers = sorted({utterance.speaker for utterance in dialogue.utterances if utterance.speaker})
    if len(speakers) < 2:
        return []

    utterance_ids_by_speaker: dict[str, list[str]] = defaultdict(list)
    text_by_speaker: dict[str, list[str]] = defaultdict(list)
    for index, utterance in enumerate(dialogue.utterances):
        utterance_id = stable_utterance_id(dialogue.dialogue_id, utterance, index)
        utterance_ids_by_speaker[utterance.speaker].append(utterance_id)
        text_by_speaker[utterance.speaker].append(utterance.text)

    full_text = "\n".join(utterance.text for utterance in dialogue.utterances)
    relation_label = infer_relation_label(dialogue.topic, full_text)
    social_distance = infer_social_distance(full_text)
    power_relation = infer_power_relation(dialogue.utterances)
    confidence = relation_confidence(relation_label, social_distance, power_relation)

    records: list[SpeakerRelationRecord] = []
    for speaker_a, speaker_b in combinations(speakers, 2):
        evidence_ids = (utterance_ids_by_speaker[speaker_a] + utterance_ids_by_speaker[speaker_b])[:6]
        records.append(
            SpeakerRelationRecord(
                dialogue_id=dialogue.dialogue_id,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                relation_label=relation_label,
                social_distance=social_distance,
                power_relation=power_relation,
                extractor=provider,
                confidence=confidence,
                evidence_utterance_ids=evidence_ids,
                attributes={
                    "topic": dialogue.topic,
                    "speaker_a_turns": len(text_by_speaker[speaker_a]),
                    "speaker_b_turns": len(text_by_speaker[speaker_b]),
                },
            )
        )
    return records


def attach_event_links_to_pragmatics(
    pragmatic_signals: list[PragmaticSignalRecord],
    events: list[EventRecord],
    *,
    min_score: float = 0.08,
) -> list[PragmaticSignalRecord]:
    events_by_dialogue: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        events_by_dialogue[event.dialogue_id].append(event)

    linked: list[PragmaticSignalRecord] = []
    for signal in pragmatic_signals:
        if signal.linked_event_id:
            linked.append(signal)
            continue
        best_event, best_score = find_best_event_link(signal, events_by_dialogue.get(signal.dialogue_id, []))
        if best_event is None or best_score < min_score:
            linked.append(signal)
            continue
        attributes = dict(signal.attributes)
        attributes["event_link_score"] = round(best_score, 4)
        linked.append(
            PragmaticSignalRecord(
                dialogue_id=signal.dialogue_id,
                utterance_id=signal.utterance_id,
                speaker=signal.speaker,
                signal_type=signal.signal_type,
                label=signal.label,
                evidence_text=signal.evidence_text,
                extractor=signal.extractor,
                confidence=signal.confidence,
                target_speaker=signal.target_speaker,
                linked_event_id=best_event.event_id,
                attributes=attributes,
            )
        )
    return linked


def find_best_event_link(
    signal: PragmaticSignalRecord,
    candidate_events: list[EventRecord],
) -> tuple[EventRecord | None, float]:
    signal_tokens = set(tokenize(signal.evidence_text))
    if not signal_tokens:
        return None, 0.0

    best_event: EventRecord | None = None
    best_score = 0.0
    for event in candidate_events:
        event_text = " ".join([event.event_text, event.event_cause, *event.anchors, *event.evidence])
        event_tokens = set(tokenize(event_text))
        if not event_tokens:
            continue
        overlap = len(signal_tokens & event_tokens)
        score = overlap / max(len(signal_tokens), 1)
        if any(anchor and anchor in signal.evidence_text for anchor in event.anchors):
            score += 0.2
        if any(piece and piece in signal.evidence_text for piece in event.evidence):
            score += 0.3
        if score > best_score:
            best_event = event
            best_score = score
    return best_event, best_score


def stable_utterance_id(dialogue_id: str, utterance: Utterance, index: int) -> str:
    if utterance.utterance_id:
        return utterance.utterance_id
    return f"{dialogue_id}:U{index + 1:04d}"


def infer_target_speaker(utterances: list[Utterance], index: int) -> str | None:
    current = utterances[index]
    if current.addressee:
        return current.addressee
    for next_utterance in utterances[index + 1 :]:
        if next_utterance.speaker != current.speaker:
            return next_utterance.speaker
    for previous in reversed(utterances[:index]):
        if previous.speaker != current.speaker:
            return previous.speaker
    return None


def make_signal(
    *,
    dialogue: DialogueRecord,
    utterance: Utterance,
    utterance_id: str,
    target_speaker: str | None,
    signal_type: str,
    label: str,
    evidence_text: str,
    confidence: float,
    provider: str,
) -> PragmaticSignalRecord:
    return PragmaticSignalRecord(
        dialogue_id=dialogue.dialogue_id,
        utterance_id=utterance_id,
        speaker=utterance.speaker,
        signal_type=signal_type,
        label=label,
        evidence_text=evidence_text,
        extractor=provider,
        confidence=confidence,
        target_speaker=target_speaker,
        attributes={"topic": dialogue.topic},
    )


def infer_speech_act(text: str) -> tuple[str, float]:
    if has_any(text, REFUSAL_MARKERS):
        return "indirect_refusal", 0.82
    if has_any(text, REQUEST_MARKERS):
        return "request", 0.78
    if has_any(text, ADVICE_MARKERS):
        return "advice", 0.68
    if has_any(text, COMPLAINT_MARKERS):
        return "complaint", 0.8
    if has_any(text, AGREEMENT_MARKERS):
        return "agreement", 0.74
    if has_any(text, DISAGREEMENT_MARKERS):
        return "disagreement", 0.72
    if is_question_like(text):
        return "question", 0.76
    if has_any(text, EMPATHY_MARKERS):
        return "empathy", 0.7
    if has_any(text, SELF_DISCLOSURE_MARKERS):
        return "self_disclosure", 0.66
    if has_any(text, ("ㅋㅋ", "ㅎㅎ", "키키")):
        return "teasing", 0.64
    return "answer", 0.52


def infer_stance(text: str, speech_act: str) -> tuple[str, float]:
    if has_any(text, ("ㅋㅋ", "ㅎㅎ", "키키")):
        return "playful", 0.74
    if speech_act in {"agreement", "empathy", "reassurance", "advice"}:
        return "supportive", 0.72
    if speech_act in {"complaint"} or has_any(text, ("ㅠ", "짜증", "힘들", "답답")):
        return "frustrated", 0.72
    if speech_act in {"indirect_refusal", "disagreement"}:
        return "hesitant_negative", 0.7
    if has_any(text, ("좋긴", "괜찮긴", "나쁘진")):
        return "hesitant_positive", 0.66
    if has_any(text, ("내가 문제", "내 탓", "못하", "미안")):
        return "self_deprecating", 0.66
    return "neutral", 0.46


def infer_politeness(text: str) -> tuple[str, float]:
    stripped = text.rstrip(".!?~ ")
    if stripped.endswith(("습니다", "합니다", "드립니다", "습니까")):
        return "deferential", 0.88
    if stripped.endswith(POLITE_ENDINGS) or has_any(stripped, ("주세요", "해요", "거예요")):
        return "polite", 0.82
    if has_any(stripped, ("ㅋㅋ", "ㅎㅎ", "야", "ㅇㅋ", "키키")):
        return "casual", 0.78
    if stripped.endswith(("다", "네", "어")):
        return "plain", 0.58
    return "mixed", 0.5


def infer_indirectness(text: str, speech_act: str) -> tuple[str, float]:
    if speech_act == "indirect_refusal":
        return "softened", 0.82
    if has_any(text, ("비꼬", "참 잘", "대단하네")):
        return "sarcastic", 0.72
    if has_any(text, IMPLICIT_MARKERS):
        return "implicit", 0.68
    if has_any(text, SOFTENING_MARKERS):
        return "softened", 0.64
    if has_any(text, ("뭔가", "애매", "모르겠")):
        return "ambiguous", 0.6
    return "direct", 0.55


def infer_emotion_cue(text: str) -> tuple[str, float] | None:
    if has_any(text, ("ㅠ", "ㅜ", "슬프", "속상")):
        return "sadness", 0.78
    if has_any(text, ("ㅋㅋ", "ㅎㅎ", "키키", "웃기")):
        return "playfulness", 0.76
    if has_any(text, ("하...", "짜증", "답답", "화나", "빡", "아파", "아프")):
        return "frustration", 0.78
    if has_any(text, ("휴", "다행", "살았다")):
        return "relief", 0.66
    if has_any(text, ("머쓱", "민망", "어색")):
        return "awkwardness", 0.66
    return None


def infer_certainty(text: str) -> tuple[str, float] | None:
    if has_any(text, ("확실", "무조건", "진짜", "분명")):
        return "certain", 0.7
    if has_any(text, UNCERTAIN_MARKERS):
        return "uncertain", 0.66
    if has_any(text, ("아마", "일 수도", "듯")):
        return "speculative", 0.62
    return None


def infer_ellipsis(text: str, speech_act: str) -> list[tuple[str, str, float]]:
    records: list[tuple[str, str, float]] = []
    if has_any(text, IMPLICIT_MARKERS):
        records.append(("ellipsisTarget", "event", 0.63))
    if speech_act in {"answer", "agreement", "disagreement"} and not has_any(text, SELF_DISCLOSURE_MARKERS):
        if len(tokenize(text)) <= 5 or text.startswith(("그", "아", "음", "하")):
            records.append(("ellipsisSubject", "speaker", 0.52))
    return records


def infer_relation_label(topic: str, full_text: str) -> str:
    merged = f"{topic}\n{full_text}"
    if has_any(merged, FAMILY_MARKERS):
        return "family"
    if has_any(merged, SERVICE_MARKERS):
        return "service"
    if has_any(merged, ("회사", "상사", "동료", "업무", "퇴근", "출근")):
        return "coworker"
    if has_any(merged, ("커뮤니티", "댓글", "게시판", "카페", "밴드")):
        return "online_community"
    if has_any(merged, ("자기", "연애", "남친", "여친", "사랑")):
        return "romantic"
    if has_any(merged, ("ㅋㅋ", "ㅎㅎ", "키키", "야", "친구")):
        return "friend"
    return "unknown"


def infer_social_distance(full_text: str) -> str:
    casual_count = sum(full_text.count(marker) for marker in ("ㅋㅋ", "ㅎㅎ", "키키", "야", "ㅇㅋ"))
    polite_count = sum(full_text.count(marker) for marker in POLITE_ENDINGS)
    if casual_count >= 2:
        return "close"
    if casual_count >= 1:
        return "neutral"
    if polite_count >= 3:
        return "distant"
    return "unknown"


def infer_power_relation(utterances: list[Utterance]) -> str:
    polite_by_speaker: Counter[str] = Counter()
    turns_by_speaker: Counter[str] = Counter()
    for utterance in utterances:
        turns_by_speaker[utterance.speaker] += 1
        politeness, _ = infer_politeness(utterance.text)
        if politeness in {"polite", "deferential"}:
            polite_by_speaker[utterance.speaker] += 1
    if len(turns_by_speaker) != 2:
        return "unknown"
    speakers = list(turns_by_speaker)
    rates = {
        speaker: polite_by_speaker[speaker] / max(turns_by_speaker[speaker], 1)
        for speaker in speakers
    }
    first, second = speakers
    diff = rates[first] - rates[second]
    if abs(diff) < 0.35:
        return "equal"
    return "listener_higher" if diff > 0 else "speaker_higher"


def relation_confidence(relation_label: str, social_distance: str, power_relation: str) -> float:
    confidence = 0.45
    if relation_label != "unknown":
        confidence += 0.18
    if social_distance != "unknown":
        confidence += 0.14
    if power_relation != "unknown":
        confidence += 0.08
    return min(confidence, 0.86)


def has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def is_question_like(text: str) -> bool:
    stripped = text.strip()
    if "?" in stripped:
        return True
    if has_any(stripped, QUESTION_MARKERS):
        return True
    return re.search(r"(?:^|\s)(뭐|무엇)(?:\s|$)", stripped) is not None


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]

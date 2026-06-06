from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RELATION_TYPES = (
    "xIntent",
    "xNeed",
    "xEffect",
    "xReact",
    "xWant",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
)

PRAGMATIC_SIGNAL_TYPES = (
    "speechAct",
    "stance",
    "politeness",
    "indirectness",
    "emotionCue",
    "listenerPressure",
    "facework",
    "humor",
    "ellipsisSubject",
    "ellipsisTarget",
    "agreementShift",
    "certainty",
)

SPEECH_ACT_LABELS = (
    "question",
    "answer",
    "agreement",
    "disagreement",
    "empathy",
    "advice",
    "request",
    "indirect_refusal",
    "complaint",
    "self_disclosure",
    "teasing",
    "reassurance",
)

STANCE_LABELS = (
    "supportive",
    "hesitant_positive",
    "hesitant_negative",
    "disagreeing",
    "neutral",
    "self_deprecating",
    "playful",
    "frustrated",
)

POLITENESS_LABELS = (
    "plain",
    "casual",
    "polite",
    "deferential",
    "mixed",
)

INDIRECTNESS_LABELS = (
    "direct",
    "softened",
    "implicit",
    "sarcastic",
    "ambiguous",
)

SOCIAL_RELATION_LABELS = (
    "family",
    "friend",
    "romantic",
    "senior_junior",
    "coworker",
    "service",
    "stranger",
    "online_community",
    "unknown",
)

SOCIAL_DISTANCE_LABELS = (
    "intimate",
    "close",
    "neutral",
    "distant",
    "unknown",
)

POWER_RELATION_LABELS = (
    "speaker_higher",
    "equal",
    "listener_higher",
    "unknown",
)

MULTIHOP_QUERY_TYPES = (
    "intent_from_context",
    "indirect_speech_act",
    "ellipsis_recovery",
    "stance_grounded_response",
    "relation_aware_response",
    "emotion_cascade",
)

MULTIHOP_NODE_KINDS = (
    "utterance",
    "event",
    "pragmatic",
    "speaker_relation",
)


@dataclass(slots=True)
class Utterance:
    speaker: str
    text: str
    utterance_id: str | None = None
    addressee: str | None = None
    register: str | None = None
    style_tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Utterance":
        utterance_id = raw.get("utterance_id")
        addressee = raw.get("addressee")
        register = raw.get("register")
        return cls(
            speaker=str(raw["speaker"]),
            text=str(raw["text"]),
            utterance_id=str(utterance_id) if utterance_id is not None else None,
            addressee=str(addressee) if addressee is not None else None,
            register=str(register) if register is not None else None,
            style_tags=[str(item) for item in raw.get("style_tags", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"speaker": self.speaker, "text": self.text}
        if self.utterance_id is not None:
            payload["utterance_id"] = self.utterance_id
        if self.addressee is not None:
            payload["addressee"] = self.addressee
        if self.register is not None:
            payload["register"] = self.register
        if self.style_tags:
            payload["style_tags"] = list(self.style_tags)
        return payload


@dataclass(slots=True)
class DialogueRecord:
    dialogue_id: str
    topic: str
    utterances: list[Utterance]
    seed_events: list[dict[str, Any]] = field(default_factory=list)
    seed_triples: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DialogueRecord":
        return cls(
            dialogue_id=str(raw["dialogue_id"]),
            topic=str(raw.get("topic", "unknown")),
            utterances=[Utterance.from_dict(item) for item in raw.get("utterances", [])],
            seed_events=list(raw.get("seed_events", [])),
            seed_triples=list(raw.get("seed_triples", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "topic": self.topic,
            "utterances": [item.to_dict() for item in self.utterances],
            "seed_events": self.seed_events,
            "seed_triples": self.seed_triples,
        }


@dataclass(slots=True)
class EventRecord:
    dialogue_id: str
    event_id: str
    event_text: str
    event_cause: str
    anchors: list[str]
    evidence: list[str]
    extractor: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EventRecord":
        return cls(
            dialogue_id=str(raw["dialogue_id"]),
            event_id=str(raw["event_id"]),
            event_text=str(raw["event_text"]),
            event_cause=str(raw.get("event_cause", "")),
            anchors=[str(item) for item in raw.get("anchors", [])],
            evidence=[str(item) for item in raw.get("evidence", [])],
            extractor=str(raw.get("extractor", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "event_id": self.event_id,
            "event_text": self.event_text,
            "event_cause": self.event_cause,
            "anchors": self.anchors,
            "evidence": self.evidence,
            "extractor": self.extractor,
        }


@dataclass(slots=True)
class TripleRecord:
    dialogue_id: str
    event_id: str
    head: str
    relation: str
    tail: str
    extractor: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TripleRecord":
        relation = str(raw["relation"])
        if relation not in RELATION_TYPES:
            raise ValueError(f"Unsupported relation: {relation}")
        return cls(
            dialogue_id=str(raw["dialogue_id"]),
            event_id=str(raw["event_id"]),
            head=str(raw["head"]),
            relation=relation,
            tail=str(raw["tail"]),
            extractor=str(raw.get("extractor", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "event_id": self.event_id,
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "extractor": self.extractor,
        }


@dataclass(slots=True)
class PragmaticSignalRecord:
    dialogue_id: str
    utterance_id: str
    speaker: str
    signal_type: str
    label: str
    evidence_text: str
    extractor: str
    confidence: float = 0.0
    target_speaker: str | None = None
    linked_event_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PragmaticSignalRecord":
        signal_type = str(raw["signal_type"])
        if signal_type not in PRAGMATIC_SIGNAL_TYPES:
            raise ValueError(f"Unsupported pragmatic signal type: {signal_type}")
        target_speaker = raw.get("target_speaker")
        linked_event_id = raw.get("linked_event_id")
        return cls(
            dialogue_id=str(raw["dialogue_id"]),
            utterance_id=str(raw["utterance_id"]),
            speaker=str(raw["speaker"]),
            signal_type=signal_type,
            label=str(raw["label"]),
            evidence_text=str(raw.get("evidence_text", "")),
            extractor=str(raw.get("extractor", "unknown")),
            confidence=float(raw.get("confidence", 0.0)),
            target_speaker=str(target_speaker) if target_speaker is not None else None,
            linked_event_id=str(linked_event_id) if linked_event_id is not None else None,
            attributes=dict(raw.get("attributes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dialogue_id": self.dialogue_id,
            "utterance_id": self.utterance_id,
            "speaker": self.speaker,
            "signal_type": self.signal_type,
            "label": self.label,
            "evidence_text": self.evidence_text,
            "extractor": self.extractor,
            "confidence": self.confidence,
            "attributes": dict(self.attributes),
        }
        if self.target_speaker is not None:
            payload["target_speaker"] = self.target_speaker
        if self.linked_event_id is not None:
            payload["linked_event_id"] = self.linked_event_id
        return payload


@dataclass(slots=True)
class SpeakerRelationRecord:
    dialogue_id: str
    speaker_a: str
    speaker_b: str
    relation_label: str
    social_distance: str
    power_relation: str
    extractor: str
    confidence: float = 0.0
    evidence_utterance_ids: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SpeakerRelationRecord":
        relation_label = str(raw["relation_label"])
        social_distance = str(raw["social_distance"])
        power_relation = str(raw["power_relation"])
        if relation_label not in SOCIAL_RELATION_LABELS:
            raise ValueError(f"Unsupported social relation label: {relation_label}")
        if social_distance not in SOCIAL_DISTANCE_LABELS:
            raise ValueError(f"Unsupported social distance label: {social_distance}")
        if power_relation not in POWER_RELATION_LABELS:
            raise ValueError(f"Unsupported power relation label: {power_relation}")
        return cls(
            dialogue_id=str(raw["dialogue_id"]),
            speaker_a=str(raw["speaker_a"]),
            speaker_b=str(raw["speaker_b"]),
            relation_label=relation_label,
            social_distance=social_distance,
            power_relation=power_relation,
            extractor=str(raw.get("extractor", "unknown")),
            confidence=float(raw.get("confidence", 0.0)),
            evidence_utterance_ids=[str(item) for item in raw.get("evidence_utterance_ids", [])],
            attributes=dict(raw.get("attributes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "speaker_a": self.speaker_a,
            "speaker_b": self.speaker_b,
            "relation_label": self.relation_label,
            "social_distance": self.social_distance,
            "power_relation": self.power_relation,
            "extractor": self.extractor,
            "confidence": self.confidence,
            "evidence_utterance_ids": list(self.evidence_utterance_ids),
            "attributes": dict(self.attributes),
        }


@dataclass(slots=True)
class ReasoningHopRecord:
    dialogue_id: str
    hop_id: str
    source_kind: str
    source_id: str
    relation: str
    target_kind: str
    target_id: str
    evidence_text: str
    score: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReasoningHopRecord":
        source_kind = str(raw["source_kind"])
        target_kind = str(raw["target_kind"])
        if source_kind not in MULTIHOP_NODE_KINDS:
            raise ValueError(f"Unsupported source kind: {source_kind}")
        if target_kind not in MULTIHOP_NODE_KINDS:
            raise ValueError(f"Unsupported target kind: {target_kind}")
        return cls(
            dialogue_id=str(raw["dialogue_id"]),
            hop_id=str(raw["hop_id"]),
            source_kind=source_kind,
            source_id=str(raw["source_id"]),
            relation=str(raw["relation"]),
            target_kind=target_kind,
            target_id=str(raw["target_id"]),
            evidence_text=str(raw.get("evidence_text", "")),
            score=float(raw.get("score", 0.0)),
            attributes=dict(raw.get("attributes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "hop_id": self.hop_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "relation": self.relation,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "evidence_text": self.evidence_text,
            "score": self.score,
            "attributes": dict(self.attributes),
        }


@dataclass(slots=True)
class MultiHopQueryRecord:
    dialogue_id: str
    query_id: str
    query_type: str
    question: str
    answer: str
    supporting_hop_ids: list[str]
    target_utterance_id: str | None = None
    difficulty: str = "2-hop"
    reasoning_focus: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MultiHopQueryRecord":
        query_type = str(raw["query_type"])
        if query_type not in MULTIHOP_QUERY_TYPES:
            raise ValueError(f"Unsupported multi-hop query type: {query_type}")
        target_utterance_id = raw.get("target_utterance_id")
        return cls(
            dialogue_id=str(raw["dialogue_id"]),
            query_id=str(raw["query_id"]),
            query_type=query_type,
            question=str(raw["question"]),
            answer=str(raw.get("answer", "")),
            supporting_hop_ids=[str(item) for item in raw.get("supporting_hop_ids", [])],
            target_utterance_id=str(target_utterance_id) if target_utterance_id is not None else None,
            difficulty=str(raw.get("difficulty", "2-hop")),
            reasoning_focus=[str(item) for item in raw.get("reasoning_focus", [])],
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dialogue_id": self.dialogue_id,
            "query_id": self.query_id,
            "query_type": self.query_type,
            "question": self.question,
            "answer": self.answer,
            "supporting_hop_ids": list(self.supporting_hop_ids),
            "difficulty": self.difficulty,
            "reasoning_focus": list(self.reasoning_focus),
            "metadata": dict(self.metadata),
        }
        if self.target_utterance_id is not None:
            payload["target_utterance_id"] = self.target_utterance_id
        return payload

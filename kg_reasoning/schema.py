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


@dataclass(slots=True)
class Utterance:
    speaker: str
    text: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Utterance":
        return cls(speaker=str(raw["speaker"]), text=str(raw["text"]))

    def to_dict(self) -> dict[str, str]:
        return {"speaker": self.speaker, "text": self.text}


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
    anchors: list[str]
    evidence: list[str]
    extractor: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EventRecord":
        return cls(
            dialogue_id=str(raw["dialogue_id"]),
            event_id=str(raw["event_id"]),
            event_text=str(raw["event_text"]),
            anchors=[str(item) for item in raw.get("anchors", [])],
            evidence=[str(item) for item in raw.get("evidence", [])],
            extractor=str(raw.get("extractor", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "event_id": self.event_id,
            "event_text": self.event_text,
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

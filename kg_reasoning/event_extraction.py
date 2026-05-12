from __future__ import annotations

from kg_reasoning.heuristics import choose_anchors, choose_event_text
from kg_reasoning.llm import OpenAICompatibleClient
from kg_reasoning.schema import DialogueRecord, EventRecord
from kg_reasoning.text import normalize_text, unique_preserving_order


EVENT_SYSTEM_PROMPT = """You extract Korean dialogue events.
Return JSON with an `events` array.
Each item must include:
- event_text: one concise Korean event phrase
- anchors: Korean verb or predicate lemmas
- evidence: one or more original utterances
Only include events that matter for causal commonsense reasoning."""


def extract_events(
    dialogues: list[DialogueRecord],
    provider: str = "hybrid",
    client: OpenAICompatibleClient | None = None,
) -> list[EventRecord]:
    events: list[EventRecord] = []
    for dialogue in dialogues:
        if provider == "seed":
            events.extend(seed_events_from_dialogue(dialogue))
            continue
        if provider in {"llm", "hybrid"} and client:
            llm_events = llm_extract_events(dialogue=dialogue, client=client)
            if llm_events:
                events.extend(llm_events)
                continue
        events.extend(rule_extract_events(dialogue))
    return events


def seed_events_from_dialogue(dialogue: DialogueRecord) -> list[EventRecord]:
    rows: list[EventRecord] = []
    for index, seed in enumerate(dialogue.seed_events, start=1):
        rows.append(
            EventRecord(
                dialogue_id=dialogue.dialogue_id,
                event_id=f"{dialogue.dialogue_id}-E{index}",
                event_text=str(seed["event_text"]),
                anchors=[str(item) for item in seed.get("anchors", [])],
                evidence=[str(item) for item in seed.get("evidence", [])],
                extractor="seed",
            )
        )
    return rows


def llm_extract_events(dialogue: DialogueRecord, client: OpenAICompatibleClient) -> list[EventRecord]:
    user_prompt = "\n".join(
        f"{utterance.speaker}: {utterance.text}" for utterance in dialogue.utterances
    )
    try:
        payload = client.chat_json(
            system_prompt=EVENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )
    except Exception:
        return []
    rows: list[EventRecord] = []
    for index, item in enumerate(payload.get("events", []), start=1):
        event_text = normalize_text(str(item.get("event_text", "")))
        if not event_text:
            continue
        anchors = unique_preserving_order([str(anchor) for anchor in item.get("anchors", [])])
        evidence = unique_preserving_order([str(entry) for entry in item.get("evidence", [])])
        rows.append(
            EventRecord(
                dialogue_id=dialogue.dialogue_id,
                event_id=f"{dialogue.dialogue_id}-E{index}",
                event_text=event_text,
                anchors=anchors,
                evidence=evidence,
                extractor="llm",
            )
        )
    return rows


def rule_extract_events(dialogue: DialogueRecord) -> list[EventRecord]:
    candidates: list[EventRecord] = []
    seen_texts: set[str] = set()
    for utterance in dialogue.utterances:
        text = normalize_text(utterance.text)
        if len(text) < 6:
            continue
        event_text = choose_event_text(text)
        if event_text in seen_texts:
            continue
        seen_texts.add(event_text)
        candidates.append(
            EventRecord(
                dialogue_id=dialogue.dialogue_id,
                event_id=f"{dialogue.dialogue_id}-E{len(candidates) + 1}",
                event_text=event_text,
                anchors=choose_anchors(text),
                evidence=[text],
                extractor="rule",
            )
        )
    if candidates:
        return candidates
    fallback = " ".join(normalize_text(item.text) for item in dialogue.utterances[:2]).strip()
    return [
        EventRecord(
            dialogue_id=dialogue.dialogue_id,
            event_id=f"{dialogue.dialogue_id}-E1",
            event_text=choose_event_text(fallback),
            anchors=choose_anchors(fallback),
            evidence=[fallback],
            extractor="rule",
        )
    ]

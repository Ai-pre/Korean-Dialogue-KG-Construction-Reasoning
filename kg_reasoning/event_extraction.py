from __future__ import annotations

from kg_reasoning.heuristics import choose_anchors, choose_event_text
from kg_reasoning.legacy_prompts import build_legacy_event_prompt, estimate_event_count
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
        if provider in {"legacy-llm", "legacy-hybrid"} and client:
            legacy_events = legacy_llm_extract_events(dialogue=dialogue, client=client)
            if legacy_events:
                events.extend(legacy_events)
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
                event_cause=str(seed.get("event_cause") or " ".join(str(item) for item in seed.get("evidence", [])[:2])),
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
                event_cause=normalize_text(str(item.get("event_cause", ""))),
                anchors=anchors,
                evidence=evidence,
                extractor="llm",
            )
        )
    return rows


def legacy_llm_extract_events(dialogue: DialogueRecord, client: OpenAICompatibleClient) -> list[EventRecord]:
    conversation_text = "\n".join(f"{utterance.speaker}: {utterance.text}" for utterance in dialogue.utterances)
    n_events = estimate_event_count(len(dialogue.utterances))
    prompt = build_legacy_event_prompt(conversation_text=conversation_text, n_events=n_events)
    try:
        payload = client.chat_json_messages(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
    except Exception:
        return []
    rows: list[EventRecord] = []
    for index, item in enumerate(payload.get("events", []), start=1):
        event_text = normalize_text(str(item.get("event_sentence") or item.get("event_text") or ""))
        if not event_text:
            continue
        event_cause = normalize_text(str(item.get("event_cause", "")))
        evidence = collect_evidence(dialogue=dialogue, event_text=event_text)
        rows.append(
            EventRecord(
                dialogue_id=dialogue.dialogue_id,
                event_id=f"{dialogue.dialogue_id}-E{index}",
                event_text=event_text,
                event_cause=event_cause,
                anchors=choose_anchors(event_text),
                evidence=evidence,
                extractor="legacy-llm",
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
                event_cause=build_rule_event_cause(dialogue=dialogue),
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
            event_cause=build_rule_event_cause(dialogue=dialogue),
            anchors=choose_anchors(fallback),
            evidence=[fallback],
            extractor="rule",
        )
    ]


def build_rule_event_cause(dialogue: DialogueRecord) -> str:
    context = [normalize_text(item.text) for item in dialogue.utterances if normalize_text(item.text)]
    if not context:
        return ""
    if len(context) == 1:
        return context[0]
    return " / ".join(context[:2])


def collect_evidence(dialogue: DialogueRecord, event_text: str) -> list[str]:
    event_tokens = set(event_text.split())
    matched = []
    for utterance in dialogue.utterances:
        text = normalize_text(utterance.text)
        if not text:
            continue
        if any(token in text for token in event_tokens if len(token) > 1):
            matched.append(text)
    return unique_preserving_order(matched)[:3] or [normalize_text(dialogue.utterances[0].text)]

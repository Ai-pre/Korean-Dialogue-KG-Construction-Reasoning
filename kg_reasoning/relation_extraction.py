from __future__ import annotations

from collections import defaultdict

from kg_reasoning.heuristics import build_relation_bundle
from kg_reasoning.llm import OpenAICompatibleClient
from kg_reasoning.schema import DialogueRecord, EventRecord, RELATION_TYPES, TripleRecord


RELATION_SYSTEM_PROMPT = """You expand Korean dialogue events into ATOMIC-style relations.
Return JSON with a `triples` array.
Each item must include:
- relation
- tail
Generate exactly one sentence per tail."""


def extract_relations(
    dialogues: list[DialogueRecord],
    events: list[EventRecord],
    provider: str = "hybrid",
    client: OpenAICompatibleClient | None = None,
) -> list[TripleRecord]:
    dialogue_index = {dialogue.dialogue_id: dialogue for dialogue in dialogues}
    grouped_events: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        grouped_events[event.dialogue_id].append(event)

    rows: list[TripleRecord] = []
    for dialogue_id, event_rows in grouped_events.items():
        dialogue = dialogue_index.get(dialogue_id)
        if provider == "seed" and dialogue:
            rows.extend(seed_relations_from_dialogue(dialogue=dialogue, events=event_rows))
            continue
        if provider in {"llm", "hybrid"} and client:
            llm_rows = llm_extract_relations(event_rows=event_rows, client=client)
            if llm_rows:
                rows.extend(llm_rows)
                continue
        rows.extend(rule_extract_relations(event_rows))
    return rows


def seed_relations_from_dialogue(dialogue: DialogueRecord, events: list[EventRecord]) -> list[TripleRecord]:
    event_lookup = {event.event_text: event for event in events}
    rows: list[TripleRecord] = []
    for triple in dialogue.seed_triples:
        event = event_lookup.get(str(triple["event_text"]))
        if not event:
            continue
        rows.append(
            TripleRecord(
                dialogue_id=dialogue.dialogue_id,
                event_id=event.event_id,
                head=event.event_text,
                relation=str(triple["relation"]),
                tail=str(triple["tail"]),
                extractor="seed",
            )
        )
    return rows


def llm_extract_relations(
    event_rows: list[EventRecord],
    client: OpenAICompatibleClient,
) -> list[TripleRecord]:
    rows: list[TripleRecord] = []
    for event in event_rows:
        user_prompt = (
            f"event_text: {event.event_text}\n"
            f"anchors: {', '.join(event.anchors)}\n"
            f"evidence: {' | '.join(event.evidence)}\n"
            "Return JSON like {\"triples\": [{\"relation\": \"xIntent\", \"tail\": \"...\"}]}"
        )
        try:
            payload = client.chat_json(
                system_prompt=RELATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
            )
        except Exception:
            return []
        relation_map = {str(item.get("relation")): str(item.get("tail", "")).strip() for item in payload.get("triples", [])}
        if any(relation not in relation_map for relation in RELATION_TYPES):
            return []
        for relation in RELATION_TYPES:
            rows.append(
                TripleRecord(
                    dialogue_id=event.dialogue_id,
                    event_id=event.event_id,
                    head=event.event_text,
                    relation=relation,
                    tail=relation_map[relation],
                    extractor="llm",
                )
            )
    return rows


def rule_extract_relations(event_rows: list[EventRecord]) -> list[TripleRecord]:
    rows: list[TripleRecord] = []
    for event in event_rows:
        bundle = build_relation_bundle(event.event_text)
        for relation in RELATION_TYPES:
            rows.append(
                TripleRecord(
                    dialogue_id=event.dialogue_id,
                    event_id=event.event_id,
                    head=event.event_text,
                    relation=relation,
                    tail=bundle[relation],
                    extractor="rule",
                )
            )
    return rows

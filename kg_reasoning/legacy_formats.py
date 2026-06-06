from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_reasoning.io import read_json, write_jsonl
from kg_reasoning.schema import EventRecord, TripleRecord


def load_legacy_event_blocks(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("Legacy event payload must be a JSON array.")
    return [item for item in payload if isinstance(item, dict)]


def load_legacy_triple_blocks(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("Legacy triple payload must be a JSON array.")
    return [item for item in payload if isinstance(item, dict)]


def convert_legacy_events(blocks: list[dict[str, Any]], *, extractor: str = "legacy-llm") -> list[EventRecord]:
    rows: list[EventRecord] = []
    for block in blocks:
        dialogue_id = Path(str(block.get("filename", "unknown"))).stem
        for index, event in enumerate(block.get("events", []), start=1):
            if not isinstance(event, dict):
                continue
            raw_event_id = str(event.get("id") or f"E{index}")
            rows.append(
                EventRecord(
                    dialogue_id=dialogue_id,
                    event_id=normalize_legacy_event_id(dialogue_id, raw_event_id),
                    event_text=str(event.get("event_sentence", "")),
                    event_cause=str(event.get("event_cause", "")),
                    anchors=[],
                    evidence=[],
                    extractor=extractor,
                )
            )
    return rows


def convert_legacy_triples(blocks: list[dict[str, Any]], *, extractor: str = "legacy-llm") -> list[TripleRecord]:
    rows: list[TripleRecord] = []
    for block in blocks:
        dialogue_id = Path(str(block.get("filename", "unknown"))).stem
        for triple in block.get("triples", []):
            if not isinstance(triple, dict):
                continue
            raw_event_id = str(triple.get("event_id") or "E1")
            rows.append(
                TripleRecord(
                    dialogue_id=dialogue_id,
                    event_id=normalize_legacy_event_id(dialogue_id, raw_event_id),
                    head=str(triple.get("head", "")),
                    relation=str(triple.get("relation", "")),
                    tail=str(triple.get("tail", "")),
                    extractor=extractor,
                )
            )
    return rows


def write_legacy_events_jsonl(
    blocks: list[dict[str, Any]],
    output_path: str | Path,
    *,
    extractor: str = "legacy-llm",
) -> None:
    rows = convert_legacy_events(blocks, extractor=extractor)
    write_jsonl(output_path, [row.to_dict() for row in rows])


def write_legacy_triples_jsonl(
    blocks: list[dict[str, Any]],
    output_path: str | Path,
    *,
    extractor: str = "legacy-llm",
) -> None:
    rows = convert_legacy_triples(blocks, extractor=extractor)
    write_jsonl(output_path, [row.to_dict() for row in rows])


def normalize_legacy_event_id(dialogue_id: str, raw_event_id: str) -> str:
    if raw_event_id.startswith(f"{dialogue_id}-"):
        return raw_event_id
    return f"{dialogue_id}-{raw_event_id}"

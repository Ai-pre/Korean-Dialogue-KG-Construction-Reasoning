from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from kg_reasoning.io import ensure_parent
from kg_reasoning.legacy_prompts import (
    build_legacy_event_prompt,
    build_legacy_triple_prompt,
    estimate_event_count,
)
from kg_reasoning.llm import OpenAICompatibleClient
from kg_reasoning.schema import DialogueRecord, EventRecord, TripleRecord


@dataclass(slots=True)
class TextExportStats:
    dialogues_written: int = 0
    output_dir: str = ""

    def to_dict(self) -> dict[str, int | str]:
        return {
            "dialogues_written": self.dialogues_written,
            "output_dir": self.output_dir,
        }


@dataclass(slots=True)
class LegacyProcessStats:
    files_seen: int = 0
    events_written: int = 0
    triples_written: int = 0
    event_output: str = ""
    triple_output: str = ""

    def to_dict(self) -> dict[str, int | str]:
        return {
            "files_seen": self.files_seen,
            "events_written": self.events_written,
            "triples_written": self.triples_written,
            "event_output": self.event_output,
            "triple_output": self.triple_output,
        }


def format_legacy_conversation(dialogue: DialogueRecord) -> str:
    return "\n".join(f"{utterance.speaker} : {utterance.text}" for utterance in dialogue.utterances)


def export_dialogues_to_txt(
    dialogues: list[DialogueRecord],
    output_dir: str | Path,
    *,
    limit: int | None = None,
) -> TextExportStats:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(dialogues, key=lambda item: item.dialogue_id)
    if limit is not None:
        selected = selected[:limit]

    written = 0
    for dialogue in selected:
        text = format_legacy_conversation(dialogue)
        if not text.strip():
            continue
        target_path = target_dir / f"{dialogue.dialogue_id}.txt"
        target_path.write_text(text, encoding="utf-8")
        written += 1

    return TextExportStats(dialogues_written=written, output_dir=str(target_dir))


def process_legacy_dialogue_files(
    *,
    src_dir: str | Path,
    out_event_json: str | Path,
    out_triple_json: str | Path,
    client: OpenAICompatibleClient,
    limit: int | None = None,
    out_event_jsonl: str | Path | None = None,
    out_triple_jsonl: str | Path | None = None,
) -> LegacyProcessStats:
    src_path = Path(src_dir)
    files = sorted(path for path in src_path.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
    if limit is not None:
        files = files[:limit]

    _start_json_array(out_event_json)
    _start_json_array(out_triple_json)
    if out_event_jsonl is not None:
        ensure_parent(out_event_jsonl).write_text("", encoding="utf-8")
    if out_triple_jsonl is not None:
        ensure_parent(out_triple_jsonl).write_text("", encoding="utf-8")

    first_event = True
    first_triple = True
    stats = LegacyProcessStats(
        files_seen=len(files),
        event_output=str(out_event_json),
        triple_output=str(out_triple_json),
    )

    for file_path in files:
        conversation = file_path.read_text(encoding="utf-8-sig")
        n_lines = len([line for line in conversation.splitlines() if line.strip()])
        n_events = estimate_event_count(n_lines)

        events_data = extract_legacy_events(client=client, conversation_text=conversation, n_events=n_events)
        events_data["filename"] = file_path.name
        first_event = _append_json_array_item(out_event_json, events_data, first=first_event)
        stats.events_written += len(events_data.get("events", []))
        if out_event_jsonl is not None:
            _append_event_jsonl_rows(out_event_jsonl, file_path=file_path, events_data=events_data)

        triple_list: list[dict[str, str]] = []
        for event in events_data.get("events", []):
            triples = generate_legacy_triples(client=client, event=event)
            for triple in triples:
                triple["event_id"] = str(event.get("id", ""))
                triple["head"] = str(event.get("event_sentence", ""))
                triple["filename"] = file_path.name
                triple_list.append(triple)

        triple_block = {"filename": file_path.name, "triples": triple_list}
        first_triple = _append_json_array_item(out_triple_json, triple_block, first=first_triple)
        stats.triples_written += len(triple_list)
        if out_triple_jsonl is not None:
            _append_triple_jsonl_rows(out_triple_jsonl, file_path=file_path, triple_list=triple_list)

    _close_json_array(out_event_json)
    _close_json_array(out_triple_json)
    return stats


def extract_legacy_events(
    *,
    client: OpenAICompatibleClient,
    conversation_text: str,
    n_events: int,
) -> dict:
    prompt = build_legacy_event_prompt(conversation_text=conversation_text, n_events=n_events)

    for _ in range(3):
        try:
            payload = client.chat_json_messages(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            if isinstance(payload, dict) and "events" in payload:
                return payload
        except Exception:
            time.sleep(1)

    return {"events": []}


def generate_legacy_triples(
    *,
    client: OpenAICompatibleClient,
    event: dict,
) -> list[dict[str, str]]:
    head = str(event.get("event_sentence", ""))
    cause = str(event.get("event_cause", ""))
    prompt = build_legacy_triple_prompt(event_text=head, event_cause=cause)

    for _ in range(3):
        try:
            payload = client.chat_json_messages(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            triples = payload.get("triples", []) if isinstance(payload, dict) else []
            if len(triples) == 9:
                return [dict(item) for item in triples if isinstance(item, dict)]
        except Exception:
            time.sleep(1)

    return []


def _start_json_array(path: str | Path) -> None:
    target = ensure_parent(path)
    target.write_text("[\n", encoding="utf-8")


def _append_json_array_item(path: str | Path, payload: dict, *, first: bool) -> bool:
    target = Path(path)
    with target.open("a", encoding="utf-8") as handle:
        if not first:
            handle.write(",\n")
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        handle.flush()
    return False


def _close_json_array(path: str | Path) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write("\n]\n")


def _append_event_jsonl_rows(path: str | Path, *, file_path: Path, events_data: dict) -> None:
    dialogue_id = file_path.stem
    rows: list[EventRecord] = []
    for index, event in enumerate(events_data.get("events", []), start=1):
        if not isinstance(event, dict):
            continue
        raw_event_id = str(event.get("id") or f"E{index}")
        rows.append(
            EventRecord(
                dialogue_id=dialogue_id,
                event_id=_normalize_event_id(dialogue_id, raw_event_id),
                event_text=str(event.get("event_sentence", "")),
                event_cause=str(event.get("event_cause", "")),
                anchors=[],
                evidence=[],
                extractor="legacy-llm",
            )
        )
    if not rows:
        return
    with ensure_parent(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()


def _append_triple_jsonl_rows(path: str | Path, *, file_path: Path, triple_list: list[dict[str, str]]) -> None:
    dialogue_id = file_path.stem
    rows: list[TripleRecord] = []
    for triple in triple_list:
        raw_event_id = str(triple.get("event_id") or "E1")
        rows.append(
            TripleRecord(
                dialogue_id=dialogue_id,
                event_id=_normalize_event_id(dialogue_id, raw_event_id),
                head=str(triple.get("head", "")),
                relation=str(triple.get("relation", "")),
                tail=str(triple.get("tail", "")),
                extractor="legacy-llm",
            )
        )
    if not rows:
        return
    with ensure_parent(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()


def _normalize_event_id(dialogue_id: str, raw_event_id: str) -> str:
    if raw_event_id.startswith(f"{dialogue_id}-"):
        return raw_event_id
    return f"{dialogue_id}-{raw_event_id}"

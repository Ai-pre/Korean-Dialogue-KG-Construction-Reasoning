from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from kg_reasoning.schema import DialogueRecord, EventRecord, TripleRecord


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def write_json(path: str | Path, payload: dict) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dialogues(path: str | Path) -> list[DialogueRecord]:
    return [DialogueRecord.from_dict(row) for row in read_jsonl(path)]


def load_events(path: str | Path) -> list[EventRecord]:
    return [EventRecord.from_dict(row) for row in read_jsonl(path)]


def load_triples(path: str | Path) -> list[TripleRecord]:
    return [TripleRecord.from_dict(row) for row in read_jsonl(path)]

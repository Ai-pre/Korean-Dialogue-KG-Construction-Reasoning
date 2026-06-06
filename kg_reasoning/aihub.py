from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from kg_reasoning.io import write_jsonl
from kg_reasoning.schema import DialogueRecord, Utterance
from kg_reasoning.text import normalize_text


SYSTEM_MARKER_PREFIXES = (
    "#@SYSTEM#",
    "#@system#",
    "#@시스템#",
    "#@사진#",
    "#@음성#",
    "#@동영상#",
    "#@기타#",
)
SPEAKER_PREFIX_RE = re.compile(r"^\s*\d+\s*:\s*")
SPEAKER_DIGIT_RE = re.compile(r"(\d+)")


@dataclass(slots=True)
class ImportStats:
    files_scanned: int = 0
    dialogues_seen: int = 0
    dialogues_written: int = 0
    utterances_written: int = 0
    skipped_empty: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "files_scanned": self.files_scanned,
            "dialogues_seen": self.dialogues_seen,
            "dialogues_written": self.dialogues_written,
            "utterances_written": self.utterances_written,
            "skipped_empty": self.skipped_empty,
        }


def import_aihub_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    include_system_utterances: bool = False,
    min_utterances: int = 2,
) -> ImportStats:
    stats = ImportStats()
    imported: list[DialogueRecord] = []

    for source_name, payload in iter_payloads(input_path):
        stats.files_scanned += 1
        for raw_dialogue in extract_dialogue_objects(payload):
            stats.dialogues_seen += 1
            dialogue = parse_aihub_dialogue(
                raw_dialogue=raw_dialogue,
                file_hint=Path(source_name).stem,
                include_system_utterances=include_system_utterances,
                min_utterances=min_utterances,
            )
            if dialogue is None:
                stats.skipped_empty += 1
                continue
            imported.append(dialogue)
            stats.dialogues_written += 1
            stats.utterances_written += len(dialogue.utterances)
            if limit is not None and stats.dialogues_written >= limit:
                write_jsonl(output_path, [item.to_dict() for item in imported])
                return stats

    write_jsonl(output_path, [item.to_dict() for item in imported])
    return stats


def iter_payloads(input_path: str | Path) -> Iterator[tuple[str, Any]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        yield from load_payload_file(path)
        return

    for file_path in sorted(iter_payload_files(path)):
        yield from load_payload_file(file_path)


def iter_payload_files(root: Path) -> Iterable[Path]:
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix in {".json", ".zip"}:
            yield file_path


def load_payload_file(path: Path) -> Iterator[tuple[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        yield str(path), read_json_payload(path.read_text(encoding="utf-8-sig"))
        return
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member_name in sorted(archive.namelist()):
                if not member_name.lower().endswith(".json"):
                    continue
                payload = read_json_payload(archive.read(member_name).decode("utf-8-sig"))
                yield f"{path}!{member_name}", payload
        return
    raise ValueError(f"Unsupported AIHub file type: {path}")


def read_json_payload(text: str) -> Any:
    return json.loads(text)


def extract_dialogue_objects(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("info"), list):
            return [item for item in payload["info"] if isinstance(item, dict)]
        if isinstance(payload.get("data"), list):
            return [item for item in payload["data"] if isinstance(item, dict)]
        if "header" in payload and "body" in payload:
            return [payload]
        if isinstance(payload.get("dialogues"), list):
            return [item for item in payload["dialogues"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Unsupported AIHub payload structure.")


def parse_aihub_dialogue(
    *,
    raw_dialogue: dict[str, Any],
    file_hint: str,
    include_system_utterances: bool,
    min_utterances: int,
) -> DialogueRecord | None:
    if "annotations" in raw_dialogue:
        return parse_info_dialogue(
            raw_dialogue=raw_dialogue,
            file_hint=file_hint,
            include_system_utterances=include_system_utterances,
            min_utterances=min_utterances,
        )
    return parse_header_body_dialogue(
        raw_dialogue=raw_dialogue,
        file_hint=file_hint,
        include_system_utterances=include_system_utterances,
        min_utterances=min_utterances,
    )


def parse_info_dialogue(
    *,
    raw_dialogue: dict[str, Any],
    file_hint: str,
    include_system_utterances: bool,
    min_utterances: int,
) -> DialogueRecord | None:
    annotations = raw_dialogue.get("annotations", {})
    dialogue_id = str(
        raw_dialogue.get("title")
        or raw_dialogue.get("filename")
        or raw_dialogue.get("id")
        or file_hint
    )
    topic = str(
        annotations.get("subject")
        or raw_dialogue.get("subject")
        or raw_dialogue.get("category")
        or raw_dialogue.get("medianame")
        or "unknown"
    )

    utterances: list[Utterance] = []
    lines = annotations.get("lines", [])
    for index, item in enumerate(lines, start=1):
        if not isinstance(item, dict):
            continue
        text = extract_line_text(item)
        if not text:
            continue
        if not include_system_utterances and is_system_utterance(text):
            continue
        speaker = normalize_speaker_label(item.get("speaker"), fallback=index)
        utterances.append(Utterance(speaker=speaker, text=text))

    if len(utterances) < min_utterances:
        return None
    return DialogueRecord(dialogue_id=dialogue_id, topic=topic, utterances=utterances)


def parse_header_body_dialogue(
    *,
    raw_dialogue: dict[str, Any],
    file_hint: str,
    include_system_utterances: bool,
    min_utterances: int,
) -> DialogueRecord | None:
    header = raw_dialogue.get("header", {})
    dialogue_info = header.get("dialogueInfo", raw_dialogue.get("dialogueInfo", {}))
    participants_info = header.get("participantsInfo", raw_dialogue.get("participantsInfo", []))
    participant_labels = build_participant_label_map(participants_info)

    dialogue_id = str(
        dialogue_info.get("dialogueID") or raw_dialogue.get("dialogueID") or raw_dialogue.get("id") or file_hint
    )
    topic = str(
        dialogue_info.get("single_topic")
        or dialogue_info.get("topic")
        or dialogue_info.get("type")
        or raw_dialogue.get("topic")
        or "unknown"
    )

    body = raw_dialogue.get("body") or raw_dialogue.get("utterances") or []
    utterances: list[Utterance] = []
    for index, item in enumerate(body, start=1):
        if not isinstance(item, dict):
            continue
        text = normalize_text(str(item.get("utterance") or item.get("text") or ""))
        if not text:
            continue
        if not include_system_utterances and is_system_utterance(text):
            continue
        participant_id = str(
            item.get("participantID")
            or item.get("speaker")
            or item.get("participant")
            or item.get("turnID")
            or index
        )
        speaker = participant_labels.get(participant_id, normalize_speaker_label(participant_id, fallback=index))
        utterances.append(Utterance(speaker=speaker, text=text))

    if len(utterances) < min_utterances:
        return None
    return DialogueRecord(dialogue_id=dialogue_id, topic=topic, utterances=utterances)


def extract_line_text(item: dict[str, Any]) -> str:
    text = normalize_text(str(item.get("norm_text") or item.get("utterance") or item.get("text") or ""))
    return SPEAKER_PREFIX_RE.sub("", text)


def normalize_speaker_label(raw_speaker: Any, *, fallback: int) -> str:
    if isinstance(raw_speaker, dict):
        candidate = str(raw_speaker.get("id") or raw_speaker.get("speakerID") or fallback)
    else:
        candidate = str(raw_speaker or fallback)
    match = SPEAKER_DIGIT_RE.search(candidate)
    if match:
        return match.group(1)
    return candidate.strip() or str(fallback)


def build_participant_label_map(participants_info: list[Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for index, item in enumerate(participants_info, start=1):
        if not isinstance(item, dict):
            continue
        participant_id = str(item.get("participantID") or item.get("id") or f"P{index:02d}")
        labels[participant_id] = str(index)
    return labels


def is_system_utterance(text: str) -> bool:
    stripped = text.strip()
    if any(stripped.startswith(prefix) for prefix in SYSTEM_MARKER_PREFIXES):
        return True
    return stripped.startswith("#@")

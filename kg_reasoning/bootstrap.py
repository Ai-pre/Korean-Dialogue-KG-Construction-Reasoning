from __future__ import annotations

from pathlib import Path

from kg_reasoning.event_extraction import seed_events_from_dialogue
from kg_reasoning.io import load_dialogues, write_jsonl
from kg_reasoning.relation_extraction import seed_relations_from_dialogue


def prepare_demo_assets(dialogues_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dialogues = load_dialogues(dialogues_path)
    events = []
    triples = []
    for dialogue in dialogues:
        event_rows = seed_events_from_dialogue(dialogue)
        events.extend(event_rows)
        triples.extend(seed_relations_from_dialogue(dialogue, event_rows))

    dialogues_file = write_jsonl(output / "dialogues.jsonl", [dialogue.to_dict() for dialogue in dialogues])
    events_file = write_jsonl(output / "events.jsonl", [event.to_dict() for event in events])
    triples_file = write_jsonl(output / "triples.jsonl", [triple.to_dict() for triple in triples])
    return {"dialogues": dialogues_file, "events": events_file, "triples": triples_file}

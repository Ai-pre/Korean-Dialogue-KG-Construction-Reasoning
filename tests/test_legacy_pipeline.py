from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kg_reasoning.io import load_events, load_triples
from kg_reasoning.legacy_formats import write_legacy_events_jsonl, write_legacy_triples_jsonl
from kg_reasoning.legacy_pipeline import export_dialogues_to_txt, process_legacy_dialogue_files
from kg_reasoning.schema import DialogueRecord, RELATION_TYPES, Utterance


class FakeClient:
    def __init__(self) -> None:
        self.event_attempts = 0
        self.triple_attempts = 0

    def chat_json_messages(self, messages, temperature: float = 0.1, max_tokens: int | None = None):
        prompt = messages[0]["content"]
        if "===DIALOG===" in prompt:
            self.event_attempts += 1
            if self.event_attempts == 1:
                raise RuntimeError("temporary event failure")
            return {
                "events": [
                    {
                        "id": "E1",
                        "event_sentence": "한 참여자는 추천을 부탁했다.",
                        "event_cause": "상대방에게 정보를 얻고 싶었기 때문이다.",
                    }
                ]
            }

        self.triple_attempts += 1
        if self.triple_attempts == 1:
            raise RuntimeError("temporary triple failure")
        return {
            "triples": [
                {"relation": relation, "tail": f"{relation} 결과 문장이다."}
                for relation in RELATION_TYPES
            ]
        }


class LegacyPipelineTest(unittest.TestCase):
    def test_export_and_process_legacy_pipeline(self) -> None:
        dialogues = [
            DialogueRecord(
                dialogue_id="dlg-b",
                topic="테스트",
                utterances=[
                    Utterance(speaker="1", text="앱 추천해 줄래?"),
                    Utterance(speaker="2", text="에이블리 한번 써 봐."),
                ],
            ),
            DialogueRecord(
                dialogue_id="dlg-a",
                topic="테스트",
                utterances=[
                    Utterance(speaker="1", text="TV 안 봐."),
                    Utterance(speaker="2", text="왜 안 봐?"),
                ],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            txt_dir = tmp_path / "txt"
            events_output = tmp_path / "dialog_events.json"
            triples_output = tmp_path / "dialog_triples.json"
            events_jsonl = tmp_path / "events.jsonl"
            triples_jsonl = tmp_path / "triples.jsonl"

            export_stats = export_dialogues_to_txt(dialogues, txt_dir)
            process_stats = process_legacy_dialogue_files(
                src_dir=txt_dir,
                out_event_json=events_output,
                out_triple_json=triples_output,
                client=FakeClient(),
                out_event_jsonl=events_jsonl,
                out_triple_jsonl=triples_jsonl,
            )

            exported_names = sorted(path.name for path in txt_dir.glob("*.txt"))
            event_blocks = json.loads(events_output.read_text(encoding="utf-8"))
            triple_blocks = json.loads(triples_output.read_text(encoding="utf-8"))
            event_rows = load_events(events_jsonl)
            triple_rows = load_triples(triples_jsonl)

        self.assertEqual(export_stats.dialogues_written, 2)
        self.assertEqual(exported_names, ["dlg-a.txt", "dlg-b.txt"])
        self.assertEqual(process_stats.files_seen, 2)
        self.assertEqual(process_stats.events_written, 2)
        self.assertEqual(process_stats.triples_written, 18)
        self.assertEqual(len(event_blocks), 2)
        self.assertEqual(len(triple_blocks), 2)
        self.assertEqual(len(triple_blocks[0]["triples"]), 9)
        self.assertEqual(len(event_rows), 2)
        self.assertEqual(len(triple_rows), 18)

    def test_convert_legacy_json_arrays_to_jsonl(self) -> None:
        event_blocks = [
            {
                "filename": "dlg-001.txt",
                "events": [
                    {
                        "id": "E1",
                        "event_sentence": "한 참여자는 추천을 부탁했다.",
                        "event_cause": "상대방에게 정보를 얻고 싶었다.",
                    }
                ],
            }
        ]
        triple_blocks = [
            {
                "filename": "dlg-001.txt",
                "triples": [
                    {
                        "event_id": "E1",
                        "head": "한 참여자는 추천을 부탁했다.",
                        "relation": "xIntent",
                        "tail": "유용한 정보를 얻고 싶어 했다.",
                    }
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            events_jsonl = tmp_path / "events.jsonl"
            triples_jsonl = tmp_path / "triples.jsonl"

            write_legacy_events_jsonl(event_blocks, events_jsonl)
            write_legacy_triples_jsonl(triple_blocks, triples_jsonl)

            events = load_events(events_jsonl)
            triples = load_triples(triples_jsonl)

        self.assertEqual(events[0].dialogue_id, "dlg-001")
        self.assertEqual(events[0].event_id, "dlg-001-E1")
        self.assertEqual(events[0].extractor, "legacy-llm")
        self.assertEqual(triples[0].dialogue_id, "dlg-001")
        self.assertEqual(triples[0].event_id, "dlg-001-E1")
        self.assertEqual(triples[0].relation, "xIntent")


if __name__ == "__main__":
    unittest.main()

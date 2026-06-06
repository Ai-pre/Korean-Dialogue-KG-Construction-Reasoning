from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from kg_reasoning.aihub import import_aihub_dataset
from kg_reasoning.io import load_dialogues


class AIHubImportTest(unittest.TestCase):
    def test_import_realistic_aihub_zip_payload(self) -> None:
        payload = {
            "dataset": {"identifier": "020"},
            "info": [
                {
                    "id": 10103,
                    "filename": "FACEBOOK_101_03.txt",
                    "title": "FACEBOOK_101_03",
                    "mediatype": "SNS",
                    "medianame": "페이스북",
                    "category": "일상대화",
                    "annotations": {
                        "subject": "교통",
                        "speaker_type": "1:1",
                        "text": "1 : #@시스템#사진#\n1 : 언니는 차 멀미 안 해?\n2 : 나는 멀미약 먹으면 괜찮아.",
                        "lines": [
                            {
                                "id": 1,
                                "text": "1 : #@시스템#사진#",
                                "norm_text": "#@시스템#사진#",
                                "speaker": {"id": "1번", "sex": "여성", "age": "20대"},
                                "speechAct": "(표현) 보여주기",
                            },
                            {
                                "id": 2,
                                "text": "1 : 언니는 차 멀미 안 해?",
                                "norm_text": "언니는 차 멀미 안 해?",
                                "speaker": {"id": "1번", "sex": "여성", "age": "20대"},
                                "speechAct": "(지시) 질문하기",
                            },
                            {
                                "id": 3,
                                "text": "2 : 나는 멀미약 먹으면 괜찮아.",
                                "norm_text": "나는 멀미약 먹으면 괜찮아.",
                                "speaker": {"id": "2번", "sex": "여성", "age": "20대"},
                                "speechAct": "(표현) 진술하기",
                            },
                        ],
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "sample_aihub.zip"
            output_path = tmp_path / "dialogues.jsonl"

            with zipfile.ZipFile(input_path, "w") as archive:
                archive.writestr("FACEBOOK_101_03.json", json.dumps(payload, ensure_ascii=False))

            stats = import_aihub_dataset(input_path=input_path, output_path=output_path)
            dialogues = load_dialogues(output_path)

        self.assertEqual(stats.files_scanned, 1)
        self.assertEqual(stats.dialogues_written, 1)
        self.assertEqual(len(dialogues), 1)
        self.assertEqual(dialogues[0].dialogue_id, "FACEBOOK_101_03")
        self.assertEqual(dialogues[0].topic, "교통")
        self.assertEqual(len(dialogues[0].utterances), 2)
        self.assertEqual(dialogues[0].utterances[0].speaker, "1")
        self.assertEqual(dialogues[0].utterances[0].text, "언니는 차 멀미 안 해?")
        self.assertEqual(dialogues[0].utterances[1].speaker, "2")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kg_reasoning.sft_prepare_ablation import main


class SftPrepareAblationTest(unittest.TestCase):
    def test_main_writes_blank_kg_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            output_path = Path(temp_dir) / "output.jsonl"
            rows = [
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {
                            "role": "user",
                            "content": "[Retrieved Events]\n1. 이벤트\n\n[Grounded Triples]\n- triple\n\n[Instruction]\n응답",
                        },
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            ]
            input_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--input-jsonl",
                        str(input_path),
                        "--output-jsonl",
                        str(output_path),
                        "--kg-ablation",
                        "blank-evidence",
                    ]
                )

            written_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(written_rows), 1)
        self.assertNotIn("이벤트", written_rows[0]["messages"][1]["content"])
        self.assertNotIn("triple", written_rows[0]["messages"][1]["content"])
        self.assertEqual(written_rows[0]["messages"][-1]["content"], "answer")


if __name__ == "__main__":
    unittest.main()

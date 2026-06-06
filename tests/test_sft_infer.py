from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kg_reasoning.sft_infer import load_inference_row, parse_args


class SftInferTest(unittest.TestCase):
    def test_load_inference_row_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            rows = [
                {"messages": [{"role": "user", "content": "첫째"}]},
                {"messages": [{"role": "user", "content": "둘째"}]},
            ]
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            selected = load_inference_row(input_jsonl=str(path), row_index=1, messages_file=None)

        self.assertEqual(selected, rows[1])

    def test_load_inference_row_from_messages_file_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "messages.json"
            payload = {"messages": [{"role": "user", "content": "안녕"}]}
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            selected = load_inference_row(input_jsonl=None, row_index=0, messages_file=str(path))

        self.assertEqual(selected, payload)

    def test_parse_args_defaults(self) -> None:
        args = parse_args(["--input-jsonl", "rows.jsonl"])
        self.assertEqual(args.row_index, 0)
        self.assertEqual(args.kg_ablation, "none")
        self.assertIsNone(args.adapter_path)


if __name__ == "__main__":
    unittest.main()

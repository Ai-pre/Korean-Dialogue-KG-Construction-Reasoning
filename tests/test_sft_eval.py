from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from kg_reasoning.sft_eval import (
    apply_kg_ablation,
    blank_kg_sections,
    build_generation_kwargs,
    load_message_rows,
    normalize_response,
    normalize_generation_config,
    parse_args,
    split_rows,
    token_f1,
)
from kg_reasoning.sft_train import split_rows as split_train_rows


class SftEvalTest(unittest.TestCase):
    def test_load_message_rows_handles_bom_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "messages.jsonl"
            rows = [
                {"messages": [{"role": "user", "content": "안녕"}]},
                {"messages": [{"role": "assistant", "content": "반가워"}]},
            ]
            with path.open("w", encoding="utf-8-sig") as handle:
                handle.write(json.dumps(rows[0], ensure_ascii=False) + "\n\n")
                handle.write("  \n")
                handle.write(json.dumps(rows[1], ensure_ascii=False) + "\n")

            loaded = load_message_rows(path)

        self.assertEqual(loaded, rows)

    def test_split_rows_matches_training_split(self) -> None:
        rows = [{"id": index} for index in range(25)]

        train_rows, eval_rows = split_rows(rows, eval_ratio=0.08, seed=7)
        expected_train_rows, expected_eval_rows = split_train_rows(rows, eval_ratio=0.08, seed=7)

        self.assertEqual(train_rows, expected_train_rows)
        self.assertEqual(eval_rows, expected_eval_rows)

    def test_normalize_response_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_response("  안녕   하세요 \n 반갑습니다  "), "안녕 하세요 반갑습니다")

    def test_token_f1_handles_overlap_and_empty_cases(self) -> None:
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("", "안녕"), 0.0)
        self.assertEqual(token_f1("안녕 세상", "다른 말"), 0.0)
        self.assertAlmostEqual(token_f1("안녕 세상", "안녕 친구"), 0.5)

    def test_blank_kg_sections_removes_retrieved_events_and_grounded_triples(self) -> None:
        text = (
            "[Topic]\n여행\n\n"
            "[Retrieved Events]\n1. 첫 이벤트\n2. 둘째 이벤트\n\n"
            "[Grounded Triples]\n- triple one\n- triple two\n\n"
            "[Instruction]\n다음 발화를 생성하라."
        )

        ablated = blank_kg_sections(text)

        self.assertNotIn("첫 이벤트", ablated)
        self.assertNotIn("triple one", ablated)
        self.assertIn("[Retrieved Events]", ablated)
        self.assertIn("[Grounded Triples]", ablated)
        self.assertIn("[Instruction]", ablated)

    def test_apply_kg_ablation_only_changes_user_messages(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": "[Retrieved Events]\n1. 이벤트\n\n[Grounded Triples]\n- triple\n\n[Instruction]\n응답",
            },
            {"role": "assistant", "content": "answer"},
        ]

        ablated = apply_kg_ablation(messages, "blank-evidence")

        self.assertEqual(ablated[0]["content"], "system")
        self.assertNotIn("이벤트", ablated[1]["content"])
        self.assertNotIn("triple", ablated[1]["content"])
        self.assertEqual(ablated[2]["content"], "answer")

    def test_build_generation_kwargs_omits_sampling_args_for_greedy(self) -> None:
        kwargs = build_generation_kwargs(
            max_new_tokens=64,
            temperature=0.0,
            top_p=0.9,
            pad_token_id=1,
            eos_token_id=2,
        )
        self.assertEqual(kwargs["max_new_tokens"], 64)
        self.assertEqual(kwargs["pad_token_id"], 1)
        self.assertEqual(kwargs["eos_token_id"], 2)
        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)

    def test_build_generation_kwargs_includes_sampling_args_when_enabled(self) -> None:
        kwargs = build_generation_kwargs(
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=1,
            eos_token_id=2,
        )
        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["top_p"], 0.9)

    def test_normalize_generation_config_clears_sampling_only_values_for_greedy(self) -> None:
        config = SimpleNamespace(do_sample=True, temperature=0.7, top_p=0.9, top_k=50)

        normalized = normalize_generation_config(config, temperature=0.0, top_p=0.9)

        self.assertFalse(normalized.do_sample)
        self.assertIsNone(normalized.temperature)
        self.assertIsNone(normalized.top_p)
        self.assertIsNone(normalized.top_k)

    def test_normalize_generation_config_sets_sampling_values_when_enabled(self) -> None:
        config = SimpleNamespace(do_sample=False, temperature=None, top_p=None, top_k=None)

        normalized = normalize_generation_config(config, temperature=0.8, top_p=0.92)

        self.assertTrue(normalized.do_sample)
        self.assertEqual(normalized.temperature, 0.8)
        self.assertEqual(normalized.top_p, 0.92)

    def test_parse_args_allows_base_model_only_eval(self) -> None:
        args = parse_args(
            [
                "--train-file",
                "train.jsonl",
                "--output-jsonl",
                "eval.jsonl",
            ]
        )
        self.assertIsNone(args.adapter_path)
        self.assertEqual(args.train_file, "train.jsonl")
        self.assertEqual(args.output_jsonl, "eval.jsonl")
        self.assertEqual(args.kg_ablation, "none")


if __name__ == "__main__":
    unittest.main()

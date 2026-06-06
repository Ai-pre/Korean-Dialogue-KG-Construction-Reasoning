from __future__ import annotations

import unittest

from kg_reasoning.sft_train import build_supervised_example, split_rows


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False):
        parts = [f"{item['role']}:{item['content']}" for item in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def __call__(self, text, add_special_tokens: bool = False, truncation: bool = False, max_length: int | None = None):
        tokens = [ord(ch) for ch in text]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": tokens}


class SftTrainTest(unittest.TestCase):
    def test_build_supervised_example_masks_prompt(self) -> None:
        tokenizer = FakeTokenizer()
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "ans"},
            ]
        }
        encoded = build_supervised_example(tokenizer=tokenizer, row=row, max_length=256)
        self.assertIsNotNone(encoded)
        assert encoded is not None
        masked_count = sum(1 for label in encoded.labels if label == -100)
        self.assertGreater(masked_count, 0)
        self.assertEqual(len(encoded.input_ids), len(encoded.labels))
        self.assertEqual(len(encoded.input_ids), len(encoded.attention_mask))
        self.assertEqual(encoded.labels[-1], ord("s"))

    def test_split_rows_creates_eval_partition(self) -> None:
        rows = [{"id": index} for index in range(20)]
        train_rows, eval_rows = split_rows(rows, eval_ratio=0.1, seed=7)
        self.assertEqual(len(train_rows) + len(eval_rows), 20)
        self.assertGreaterEqual(len(eval_rows), 1)


if __name__ == "__main__":
    unittest.main()

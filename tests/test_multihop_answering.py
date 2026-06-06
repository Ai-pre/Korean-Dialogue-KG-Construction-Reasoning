from __future__ import annotations

import unittest

from kg_reasoning.multihop_answering import parse_prompt_hops, rewrite_multihop_training_answer


def make_training_row(query_type: str = "indirect_speech_act") -> dict:
    prompt = "\n".join(
        [
            "[Reasoning Hops]",
            "- 1. utterance:U1 --indirectness--> pragmatic:U1:indirectness | label=softened",
            "  evidence: that is a little hard for me.",
            "- 2. utterance:U1 --speechAct--> pragmatic:U1:speechAct | label=indirect_refusal",
            "  evidence: that is a little hard for me.",
            "- 3. pragmatic:U1:speechAct --conditioned_by_relation--> speaker_relation:1:2 | label=friend",
            "  evidence: friend",
        ]
    )
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "old template answer"},
        ],
        "metadata": {"query_type": query_type, "query_id": "Q1"},
    }


class MultiHopAnsweringTest(unittest.TestCase):
    def test_parse_prompt_hops_extracts_relation_label_and_evidence(self) -> None:
        row = make_training_row()
        hops = parse_prompt_hops(row["messages"][1]["content"])

        self.assertEqual(hops[0].relation, "indirectness")
        self.assertEqual(hops[0].label, "softened")
        self.assertEqual(hops[0].evidence, "that is a little hard for me.")

    def test_rewrite_multihop_training_answer_adds_strategy_answer(self) -> None:
        row = make_training_row()
        rewritten = rewrite_multihop_training_answer(row)

        answer = rewritten["messages"][2]["content"]
        self.assertNotEqual(answer, "old template answer")
        self.assertIn("that is a little hard for me.", answer)
        self.assertIn("\uc120\ud0dd\uad8c", answer)
        self.assertEqual(rewritten["metadata"]["answer_style"], "strategy_v2")

    def test_rewrite_multihop_training_answer_uses_relation_context(self) -> None:
        row = make_training_row(query_type="relation_aware_response")
        rewritten = rewrite_multihop_training_answer(row)

        answer = rewritten["messages"][2]["content"]
        self.assertIn("\uce5c\uad6c \uad00\uacc4", answer)
        self.assertIn("\ud654\ud589", answer)


if __name__ == "__main__":
    unittest.main()

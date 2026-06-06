from __future__ import annotations

import unittest

from kg_reasoning.augmentation import build_dialogue_query
from kg_reasoning.schema import DialogueRecord, Utterance
from kg_reasoning.training_data import build_training_examples


class TrainingDataTest(unittest.TestCase):
    def test_history_query_uses_only_previous_turns(self) -> None:
        dialogue = DialogueRecord(
            dialogue_id="dlg-001",
            topic="meal",
            utterances=[
                Utterance(speaker="1", text="뭐 먹지"),
                Utterance(speaker="2", text="라면 먹을래"),
                Utterance(speaker="1", text="좋아"),
            ],
        )
        query = build_dialogue_query(dialogue, query_source="history", history_window=2)
        self.assertIn("뭐 먹지", query)
        self.assertIn("라면 먹을래", query)
        self.assertNotIn("좋아", query)
        self.assertNotIn("1:", query)

    def test_history_query_respects_window(self) -> None:
        dialogue = DialogueRecord(
            dialogue_id="dlg-002",
            topic="meal",
            utterances=[
                Utterance(speaker="1", text="첫 번째"),
                Utterance(speaker="2", text="두 번째"),
                Utterance(speaker="1", text="세 번째"),
                Utterance(speaker="2", text="네 번째"),
                Utterance(speaker="1", text="다섯 번째"),
            ],
        )
        query = build_dialogue_query(dialogue, query_source="history", history_window=2)
        self.assertNotIn("두 번째", query)
        self.assertIn("세 번째", query)
        self.assertIn("네 번째", query)
        self.assertNotIn("다섯 번째", query)

    def test_build_next_turn_examples(self) -> None:
        rows = [
            {
                "dialogue_id": "dlg-001",
                "topic": "meal",
                "query": "1: 뭐 먹지 2: 라면 먹을래",
                "query_source": "history",
                "evidence_source": "retrieval",
                "utterances": [
                    {"speaker": "1", "text": "뭐 먹지"},
                    {"speaker": "2", "text": "라면 먹을래"},
                    {"speaker": "1", "text": "좋아"},
                ],
                "retrieved_nodes": [{"node_text": "한 참여자는 메뉴를 고민했다.", "final_score": 0.9}],
                "grounded_triples": [
                    {
                        "head": "한 참여자는 메뉴를 고민했다.",
                        "relation": "xIntent",
                        "tail": "배를 채우기 위해 메뉴를 정하려고 했다.",
                    }
                ],
                "baseline": "baseline",
                "kg_aware": "kg aware",
            }
        ]
        examples = build_training_examples(rows, task="next-turn")
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["messages"][-1]["content"], "좋아")
        self.assertIn("[Dialogue History]", examples[0]["messages"][1]["content"])
        self.assertIn("[KG Evidence]", examples[0]["messages"][1]["content"])
        self.assertEqual(examples[0]["metadata"]["task"], "next-turn")

    def test_build_kg_response_examples(self) -> None:
        rows = [
            {
                "dialogue_id": "dlg-001",
                "topic": "meal",
                "query": "좋아",
                "query_source": "last-utterance",
                "evidence_source": "dialogue-linked",
                "utterances": [
                    {"speaker": "1", "text": "뭐 먹지"},
                    {"speaker": "2", "text": "라면 먹을래"},
                    {"speaker": "1", "text": "좋아"},
                ],
                "retrieved_nodes": [{"node_text": "한 참여자는 메뉴를 고민했다.", "final_score": 1.0}],
                "grounded_triples": [
                    {
                        "head": "한 참여자는 메뉴를 고민했다.",
                        "relation": "xIntent",
                        "tail": "배를 채우기 위해 메뉴를 정하려고 했다.",
                    }
                ],
                "baseline": "baseline",
                "kg_aware": "kg aware",
            }
        ]
        examples = build_training_examples(rows, task="kg-response")
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["messages"][-1]["content"], "kg aware")
        self.assertEqual(examples[0]["metadata"]["task"], "kg-response")


if __name__ == "__main__":
    unittest.main()

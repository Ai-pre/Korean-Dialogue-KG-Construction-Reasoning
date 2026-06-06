from __future__ import annotations

import unittest

from kg_reasoning.augmentation import augment_dialogues, build_dialogue_query, rank_dialogue_events
from kg_reasoning.graph import build_graph
from kg_reasoning.schema import DialogueRecord, TripleRecord, Utterance


class AugmentationTest(unittest.TestCase):
    def test_build_dialogue_query(self) -> None:
        dialogue = DialogueRecord(
            dialogue_id="dlg-001",
            topic="meal",
            utterances=[
                Utterance(speaker="1", text="배고파"),
                Utterance(speaker="2", text="라면 먹을래"),
            ],
        )
        self.assertEqual(build_dialogue_query(dialogue, query_source="last-utterance"), "라면 먹을래")
        self.assertEqual(build_dialogue_query(dialogue, query_source="full-dialogue"), "배고파 라면 먹을래")

    def test_augment_dialogues_outputs_context(self) -> None:
        triples = [
            TripleRecord(
                dialogue_id="dlg-001",
                event_id="dlg-001-E1",
                head="라면을 먹을지 고민한다",
                relation="xIntent",
                tail="배를 채우기 위해 메뉴를 정하려고 한다.",
                extractor="legacy-llm",
            )
        ]
        graph = build_graph(triples)
        encoder = {
            "node_text_vectors": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            "node_embeddings": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            "metrics": {},
        }
        dialogues = [
            DialogueRecord(
                dialogue_id="dlg-001",
                topic="meal",
                utterances=[
                    Utterance(speaker="1", text="뭐 먹지"),
                    Utterance(speaker="2", text="라면 먹을래"),
                ],
            )
        ]

        rows = augment_dialogues(
            dialogues=dialogues,
            graph=graph,
            encoder=encoder,
            triples=triples,
            top_k=1,
            mode="template",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dialogue_id"], "dlg-001")
        self.assertEqual(rows[0]["query"], "라면 먹을래")
        self.assertEqual(rows[0]["evidence_source"], "dialogue-linked")
        self.assertEqual(len(rows[0]["retrieved_nodes"]), 1)
        self.assertEqual(rows[0]["grounded_triples"][0]["relation"], "xIntent")
        self.assertIn("[Retrieved Events]", rows[0]["augmentation_context"])
        self.assertIn("[Grounded Triples]", rows[0]["augmentation_context"])

    def test_rank_dialogue_events_prefers_matching_history(self) -> None:
        triples = [
            TripleRecord(
                dialogue_id="dlg-001",
                event_id="dlg-001-E1",
                head="한 참여자는 코인 가격이 급등했다고 말했다.",
                relation="xIntent",
                tail="코인 시장 변동을 공유하고 싶었다.",
                extractor="legacy-llm",
            ),
            TripleRecord(
                dialogue_id="dlg-001",
                event_id="dlg-001-E2",
                head="한 참여자는 다이어트를 내일부터 시작하겠다고 말했다.",
                relation="xIntent",
                tail="건강을 챙기고 싶었다.",
                extractor="legacy-llm",
            ),
        ]
        nodes, grounded = rank_dialogue_events(
            query="트럼프 말 한마디에 코인이 또 올랐대",
            dialogue_triples=triples,
            top_events=1,
        )
        self.assertEqual(len(nodes), 1)
        self.assertIn("코인", nodes[0]["node_text"])
        self.assertEqual(len(grounded), 1)
        self.assertIn("코인", grounded[0].head)


if __name__ == "__main__":
    unittest.main()

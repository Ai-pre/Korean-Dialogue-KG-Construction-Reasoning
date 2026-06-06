from __future__ import annotations

import unittest

from kg_reasoning.schema import MultiHopQueryRecord, ReasoningHopRecord
from kg_reasoning.training_data import build_multihop_training_rows, build_training_examples


class MultiHopTrainingDataTest(unittest.TestCase):
    def test_build_multihop_kg_response_example_contains_reasoning_hops(self) -> None:
        hops = [
            ReasoningHopRecord(
                dialogue_id="D1",
                hop_id="H1",
                source_kind="utterance",
                source_id="U1",
                relation="speechAct",
                target_kind="pragmatic",
                target_id="U1:speechAct",
                evidence_text="오늘 가는 건 좀 어려울 것 같아...",
                score=0.82,
                attributes={"label": "indirect_refusal"},
            ),
            ReasoningHopRecord(
                dialogue_id="D1",
                hop_id="H2",
                source_kind="pragmatic",
                source_id="U1:speechAct",
                relation="conditioned_by_relation",
                target_kind="speaker_relation",
                target_id="1:2",
                evidence_text="friend",
                score=0.7,
                attributes={"relation_label": "friend"},
            ),
        ]
        queries = [
            MultiHopQueryRecord(
                dialogue_id="D1",
                query_id="Q1",
                query_type="relation_aware_response",
                question="화자 관계와 말투를 함께 고려하면 어떤 응답 방향이 자연스러운가?",
                answer="친구 관계이므로 완곡한 거절을 압박하지 않고 다음 기회를 열어 주는 응답이 자연스럽다.",
                supporting_hop_ids=["H1", "H2"],
                target_utterance_id="U1",
                reasoning_focus=["speechAct", "social_relation"],
            )
        ]

        rows = build_multihop_training_rows(queries=queries, hops=hops)
        examples = build_training_examples(rows, task="multi-hop-kg-response")

        self.assertEqual(len(examples), 1)
        user_prompt = examples[0]["messages"][1]["content"]
        self.assertIn("[Reasoning Hops]", user_prompt)
        self.assertIn("conditioned_by_relation", user_prompt)
        self.assertEqual(examples[0]["messages"][2]["content"], queries[0].answer)
        self.assertEqual(examples[0]["metadata"]["task"], "multi-hop-kg-response")


if __name__ == "__main__":
    unittest.main()

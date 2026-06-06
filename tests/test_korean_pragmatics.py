from __future__ import annotations

import unittest

from kg_reasoning.korean_pragmatics import build_pragmatic_schema_snapshot
from kg_reasoning.schema import (
    MultiHopQueryRecord,
    PragmaticSignalRecord,
    SpeakerRelationRecord,
    Utterance,
)


class KoreanPragmaticsTest(unittest.TestCase):
    def test_utterance_roundtrip_preserves_optional_fields(self) -> None:
        utterance = Utterance(
            speaker="1",
            text="그건 좀...",
            utterance_id="U3",
            addressee="2",
            register="casual",
            style_tags=["ellipsis", "hesitation"],
        )

        raw = utterance.to_dict()
        restored = Utterance.from_dict(raw)

        self.assertEqual(restored.utterance_id, "U3")
        self.assertEqual(restored.addressee, "2")
        self.assertEqual(restored.register, "casual")
        self.assertEqual(restored.style_tags, ["ellipsis", "hesitation"])

    def test_pragmatic_signal_roundtrip(self) -> None:
        record = PragmaticSignalRecord(
            dialogue_id="D1",
            utterance_id="U1",
            speaker="1",
            signal_type="speechAct",
            label="indirect_refusal",
            evidence_text="그건 좀...",
            extractor="rule",
            confidence=0.8,
            target_speaker="2",
            linked_event_id="E1",
            attributes={"marker": "좀"},
        )

        restored = PragmaticSignalRecord.from_dict(record.to_dict())

        self.assertEqual(restored.signal_type, "speechAct")
        self.assertEqual(restored.label, "indirect_refusal")
        self.assertEqual(restored.linked_event_id, "E1")
        self.assertEqual(restored.attributes["marker"], "좀")

    def test_speaker_relation_roundtrip(self) -> None:
        record = SpeakerRelationRecord(
            dialogue_id="D1",
            speaker_a="1",
            speaker_b="2",
            relation_label="friend",
            social_distance="close",
            power_relation="equal",
            extractor="rule",
            confidence=0.6,
            evidence_utterance_ids=["U1", "U2"],
        )

        restored = SpeakerRelationRecord.from_dict(record.to_dict())

        self.assertEqual(restored.relation_label, "friend")
        self.assertEqual(restored.social_distance, "close")
        self.assertEqual(restored.power_relation, "equal")

    def test_multihop_query_roundtrip(self) -> None:
        record = MultiHopQueryRecord(
            dialogue_id="D1",
            query_id="D1:Q1",
            query_type="relation_aware_response",
            question="이 관계와 말투를 유지하면서 어떤 응답이 자연스러운가?",
            answer="친한 친구 말투를 유지한 응답이 적절하다.",
            supporting_hop_ids=["H1", "H2"],
            target_utterance_id="U3",
            reasoning_focus=["politeness", "social_relation"],
        )

        restored = MultiHopQueryRecord.from_dict(record.to_dict())

        self.assertEqual(restored.query_type, "relation_aware_response")
        self.assertEqual(restored.supporting_hop_ids, ["H1", "H2"])
        self.assertEqual(restored.target_utterance_id, "U3")

    def test_pragmatic_schema_snapshot_exposes_new_layers(self) -> None:
        snapshot = build_pragmatic_schema_snapshot()

        self.assertIn("signal_taxonomy", snapshot)
        self.assertIn("social_taxonomy", snapshot)
        self.assertIn("multihop_query_templates", snapshot)
        self.assertIn("speechAct", snapshot["signal_taxonomy"])


if __name__ == "__main__":
    unittest.main()

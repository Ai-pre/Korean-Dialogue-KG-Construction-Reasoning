from __future__ import annotations

import unittest

from kg_reasoning.pragmatic_extraction import attach_event_links_to_pragmatics, extract_pragmatics
from kg_reasoning.schema import DialogueRecord, EventRecord, Utterance


class PragmaticExtractionTest(unittest.TestCase):
    def test_rule_extractor_finds_indirect_refusal_and_close_relation(self) -> None:
        dialogue = DialogueRecord(
            dialogue_id="D-prag-1",
            topic="친구 약속",
            utterances=[
                Utterance(speaker="1", text="오늘 가는 건 좀 어려울 것 같아...", utterance_id="U1"),
                Utterance(speaker="2", text="ㅋㅋ 알겠어 다음에 보자", utterance_id="U2"),
            ],
        )

        signals, relations = extract_pragmatics([dialogue])

        signal_pairs = {(signal.signal_type, signal.label) for signal in signals}
        self.assertIn(("speechAct", "indirect_refusal"), signal_pairs)
        self.assertIn(("indirectness", "softened"), signal_pairs)
        self.assertIn(("politeness", "casual"), signal_pairs)
        self.assertEqual(relations[0].relation_label, "friend")
        self.assertEqual(relations[0].social_distance, "neutral")

    def test_attach_event_links_uses_evidence_overlap(self) -> None:
        dialogue = DialogueRecord(
            dialogue_id="D-prag-2",
            topic="친구 약속",
            utterances=[
                Utterance(speaker="1", text="오늘 가는 건 좀 어려울 것 같아...", utterance_id="U1"),
                Utterance(speaker="2", text="다음에 보자", utterance_id="U2"),
            ],
        )
        events = [
            EventRecord(
                dialogue_id="D-prag-2",
                event_id="E1",
                event_text="화자 1은 오늘 약속에 가기 어렵다고 말한다.",
                event_cause="개인 사정이 있어 약속 참여가 어렵다.",
                anchors=["어려울"],
                evidence=["오늘 가는 건 좀 어려울 것 같아..."],
                extractor="rule",
            )
        ]

        signals, _ = extract_pragmatics([dialogue])
        linked = attach_event_links_to_pragmatics(signals, events)

        linked_event_ids = {signal.linked_event_id for signal in linked if signal.linked_event_id}
        self.assertEqual(linked_event_ids, {"E1"})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from kg_reasoning.multihop import build_multihop_queries, build_reasoning_hops
from kg_reasoning.schema import (
    DialogueRecord,
    EventRecord,
    PragmaticSignalRecord,
    SpeakerRelationRecord,
    Utterance,
)


class MultiHopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dialogue = DialogueRecord(
            dialogue_id="D1",
            topic="연애/결혼",
            utterances=[
                Utterance(speaker="1", text="그건 좀...", utterance_id="U1", addressee="2"),
                Utterance(speaker="2", text="왜?", utterance_id="U2", addressee="1"),
            ],
        )
        self.events = [
            EventRecord(
                dialogue_id="D1",
                event_id="E1",
                event_text="한 참여자는 제안을 완곡하게 거절했다.",
                event_cause="부담을 느꼈기 때문이다.",
                anchors=["거절"],
                evidence=["그건 좀..."],
                extractor="rule",
            )
        ]
        self.pragmatics = [
            PragmaticSignalRecord(
                dialogue_id="D1",
                utterance_id="U1",
                speaker="1",
                signal_type="speechAct",
                label="indirect_refusal",
                evidence_text="그건 좀...",
                extractor="rule",
                confidence=0.9,
                target_speaker="2",
                linked_event_id="E1",
            ),
            PragmaticSignalRecord(
                dialogue_id="D1",
                utterance_id="U1",
                speaker="1",
                signal_type="indirectness",
                label="softened",
                evidence_text="좀",
                extractor="rule",
                confidence=0.8,
                target_speaker="2",
                linked_event_id="E1",
            ),
        ]
        self.relations = [
            SpeakerRelationRecord(
                dialogue_id="D1",
                speaker_a="1",
                speaker_b="2",
                relation_label="friend",
                social_distance="close",
                power_relation="equal",
                extractor="rule",
                confidence=0.7,
                evidence_utterance_ids=["U1", "U2"],
            )
        ]

    def test_build_reasoning_hops_links_utterance_to_pragmatics_and_event(self) -> None:
        hops = build_reasoning_hops(
            dialogue_id="D1",
            events=self.events,
            pragmatic_signals=self.pragmatics,
            speaker_relations=self.relations,
        )

        relations = {hop.relation for hop in hops}
        self.assertIn("speechAct", relations)
        self.assertIn("grounds_event", relations)
        self.assertIn("conditioned_by_relation", relations)

    def test_build_multihop_queries_creates_korean_reasoning_tasks(self) -> None:
        hops, queries = build_multihop_queries(
            dialogue=self.dialogue,
            events=self.events,
            pragmatic_signals=self.pragmatics,
            speaker_relations=self.relations,
        )

        self.assertGreaterEqual(len(hops), 3)
        self.assertGreaterEqual(len(queries), 2)
        query_types = {query.query_type for query in queries}
        self.assertIn("indirect_speech_act", query_types)
        self.assertIn("relation_aware_response", query_types)


if __name__ == "__main__":
    unittest.main()

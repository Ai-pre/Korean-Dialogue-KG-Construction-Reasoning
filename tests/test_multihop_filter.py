from __future__ import annotations

import unittest

from kg_reasoning.multihop_filter import select_balanced_examples


GENERIC_EVENT = "\ub300\ud654 \uc18d \ud575\uc2ec \uc0c1\ud669\uc744 \uc815\ub9ac\ud558\uace0 \ub2e4\uc74c \ud589\ub3d9\uc744 \uace0\ubbfc\ud55c\ub2e4"
SPECIFIC_EVENT = "\uc0c1\ub300\uac00 \uc644\uace1\ud558\uac8c \uac70\uc808\ud55c\ub2e4"


def make_row(
    query_type: str,
    query_id: str,
    *,
    generic: bool = False,
    event_hop: bool = True,
) -> dict:
    event_text = GENERIC_EVENT if generic else SPECIFIC_EVENT
    prompt_lines = [
        "[Reasoning Hops]",
        "- 1. utterance:U1 --speechAct--> pragmatic:U1:speechAct | label=indirect_refusal",
        "  evidence: today is a little hard for me.",
    ]
    if event_hop:
        prompt_lines.extend(
            [
                f"- 2. pragmatic:U1:speechAct --grounds_event--> event:E1 | label={event_text}",
                f"  evidence: {event_text}",
            ]
        )
    else:
        prompt_lines.extend(
            [
                "- 2. utterance:U2 --stanceToward--> pragmatic:U1:stance | label=soft_support",
                "  evidence: soft_support",
            ]
        )
    prompt_lines.extend(
        [
            "- 3. pragmatic:U1:speechAct --conditioned_by_relation--> speaker_relation:1:2 | label=friend",
            "  evidence: friend",
        ]
    )
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "\n".join(prompt_lines)},
            {"role": "assistant", "content": "A relation-aware response should soften the refusal."},
        ],
        "metadata": {"query_type": query_type, "query_id": query_id},
    }


def make_indirect_row(
    query_id: str,
    *,
    evidence: str,
    label: str = "softened",
    speech_act: str = "answer",
) -> dict:
    prompt = "\n".join(
        [
            "[Reasoning Hops]",
            f"- 1. utterance:U1 --indirectness--> pragmatic:U1:indirectness | label={label}",
            f"  evidence: {evidence}",
            f"- 2. utterance:U1 --speechAct--> pragmatic:U1:speechAct | label={speech_act}",
            f"  evidence: {evidence}",
            "- 3. pragmatic:U1:speechAct --conditioned_by_relation--> speaker_relation:1:2 | label=friend",
            "  evidence: friend",
        ]
    )
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "old"},
        ],
        "metadata": {"query_type": "indirect_speech_act", "query_id": query_id},
    }


class MultiHopFilterTest(unittest.TestCase):
    def test_select_balanced_examples_respects_caps(self) -> None:
        rows = [
            make_row("relation_aware_response", f"R{i}")
            for i in range(5)
        ]
        rows.extend(make_row("indirect_speech_act", f"I{i}") for i in range(3))

        selected, stats = select_balanced_examples(
            rows,
            type_caps={"relation_aware_response": 2, "indirect_speech_act": 2},
            seed=7,
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(stats.kept_by_type["relation_aware_response"], 2)
        self.assertEqual(stats.kept_by_type["indirect_speech_act"], 2)

    def test_select_balanced_examples_drops_generic_event(self) -> None:
        rows = [
            make_row("relation_aware_response", "good"),
            make_row("relation_aware_response", "generic", generic=True),
        ]

        selected, stats = select_balanced_examples(
            rows,
            type_caps={"relation_aware_response": 5},
            seed=7,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["metadata"]["query_id"], "good")
        self.assertEqual(stats.rejected_by_reason["generic_event"], 1)

    def test_select_balanced_examples_can_drop_event_hops(self) -> None:
        rows = [
            make_row("relation_aware_response", "event"),
            make_row("relation_aware_response", "no-event", event_hop=False),
        ]

        selected, stats = select_balanced_examples(
            rows,
            type_caps={"relation_aware_response": 5},
            drop_event_hops=True,
            seed=7,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["metadata"]["query_id"], "no-event")
        self.assertEqual(stats.rejected_by_reason["event_hop"], 1)

    def test_select_balanced_examples_can_drop_weak_softened_indirectness(self) -> None:
        rows = [
            make_indirect_row("weak", evidence="I just felt sad...", speech_act="indirect_refusal"),
            make_indirect_row("strong", evidence="\uc774\uac74 \uc5b4\ub835\ub2e4...", label="softened", speech_act="indirect_refusal"),
            make_indirect_row("implicit", evidence="\uc544\ub2c8 \uadf8\uac74 \ubb34\ub9ac\uc57c", label="implicit", speech_act="disagreement"),
        ]

        selected, stats = select_balanced_examples(
            rows,
            type_caps={"indirect_speech_act": 5},
            drop_weak_indirectness=True,
            seed=7,
        )

        selected_ids = {row["metadata"]["query_id"] for row in selected}
        self.assertEqual(selected_ids, {"strong", "implicit"})
        self.assertEqual(stats.rejected_by_reason["weak_softened_indirectness"], 1)

    def test_select_balanced_examples_can_drop_weak_indirect_speech_act(self) -> None:
        rows = [
            make_indirect_row("weak", evidence="That thing again?", label="implicit", speech_act="question"),
            make_indirect_row("strong", evidence="\uc544\ub2c8 \uadf8\uac74 \ubb34\ub9ac\uc57c", label="implicit", speech_act="disagreement"),
        ]

        selected, stats = select_balanced_examples(
            rows,
            type_caps={"indirect_speech_act": 5},
            drop_weak_indirectness=True,
            seed=7,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["metadata"]["query_id"], "strong")
        self.assertEqual(stats.rejected_by_reason["weak_indirect_speech_act"], 1)


if __name__ == "__main__":
    unittest.main()

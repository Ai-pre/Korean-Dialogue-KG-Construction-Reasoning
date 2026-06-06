from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


DEFAULT_QUERY_TYPE_CAPS = {
    "relation_aware_response": 6000,
    "indirect_speech_act": 6000,
    "intent_from_context": 6000,
    "stance_grounded_response": 4000,
    "emotion_cascade": 3500,
    "ellipsis_recovery": 1500,
}

GENERIC_EVENT_MARKERS = (
    "\ub300\ud654 \uc18d \ud575\uc2ec \uc0c1\ud669",
    "\ub2e4\uc74c \ud589\ub3d9\uc744 \uace0\ubbfc\ud55c\ub2e4",
)
INDIRECTNESS_HOP_RE = re.compile(
    r"--indirectness-->[^\n]*\| label=(?P<label>[^\n]+)\n\s*evidence:\s*(?P<evidence>[^\n]+)"
)
SPEECH_ACT_HOP_RE = re.compile(
    r"--speechAct-->[^\n]*\| label=(?P<label>[^\n]+)\n\s*evidence:\s*(?P<evidence>[^\n]+)"
)
WEAK_SOFTENING_SURFACE_MARKERS = ("...", "\u2026", ",,", "\u3160", "\u315c")
STRONG_INDIRECT_SPEECH_ACTS = ("indirect_refusal", "disagreement", "complaint", "request")
STRONG_INDIRECTNESS_MARKERS = (
    "\uc5b4\ub835",
    "\ud798\ub4e4",
    "\ubd80\ub2f4",
    "\uace4\ub780",
    "\uc548 \ub420",
    "\uc548\ub418",
    "\uc2eb",
    "\uae00\uc384",
    "\ubb34\ub9ac",
    "\uac70\uc808",
    "\ubabb\ud558",
    "\uc880 \uadf8\ub807",
)


@dataclass(slots=True)
class FilterStats:
    seen: int = 0
    eligible: int = 0
    selected: int = 0
    kept_by_type: Counter[str] = field(default_factory=Counter)
    eligible_by_type: Counter[str] = field(default_factory=Counter)
    rejected_by_reason: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "eligible": self.eligible,
            "selected": self.selected,
            "kept_by_type": dict(self.kept_by_type),
            "eligible_by_type": dict(self.eligible_by_type),
            "rejected_by_reason": dict(self.rejected_by_reason),
        }


def parse_type_caps(raw_caps: str | None) -> dict[str, int]:
    if not raw_caps:
        return dict(DEFAULT_QUERY_TYPE_CAPS)
    caps = dict(DEFAULT_QUERY_TYPE_CAPS)
    for part in raw_caps.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid type cap: {part}")
        key, value = part.split("=", 1)
        caps[key.strip()] = int(value)
    return caps


def select_balanced_examples(
    rows: Iterable[dict[str, Any]],
    *,
    type_caps: dict[str, int] | None = None,
    min_hops: int = 3,
    drop_generic_event: bool = True,
    drop_event_hops: bool = False,
    drop_weak_indirectness: bool = False,
    drop_unknown_relation: bool = False,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], FilterStats]:
    rng = random.Random(seed)
    caps = type_caps or DEFAULT_QUERY_TYPE_CAPS
    reservoirs: dict[str, list[dict[str, Any]]] = {key: [] for key in caps}
    seen_by_type: Counter[str] = Counter()
    stats = FilterStats()

    for row in rows:
        add_row_to_reservoirs(
            row,
            reservoirs=reservoirs,
            seen_by_type=seen_by_type,
            stats=stats,
            rng=rng,
            caps=caps,
            min_hops=min_hops,
            drop_generic_event=drop_generic_event,
            drop_event_hops=drop_event_hops,
            drop_weak_indirectness=drop_weak_indirectness,
            drop_unknown_relation=drop_unknown_relation,
        )

    selected: list[dict[str, Any]] = []
    for query_type in sorted(reservoirs):
        selected.extend(reservoirs[query_type])
    rng.shuffle(selected)
    stats.selected = len(selected)
    stats.kept_by_type = Counter(get_query_type(row) for row in selected)
    return selected, stats


def add_row_to_reservoirs(
    row: dict[str, Any],
    *,
    reservoirs: dict[str, list[dict[str, Any]]],
    seen_by_type: Counter[str],
    stats: FilterStats,
    rng: random.Random,
    caps: dict[str, int],
    min_hops: int,
    drop_generic_event: bool,
    drop_event_hops: bool,
    drop_weak_indirectness: bool,
    drop_unknown_relation: bool,
) -> None:
    stats.seen += 1
    query_type = get_query_type(row)
    if query_type not in caps:
        stats.rejected_by_reason["unsupported_query_type"] += 1
        return

    reason = rejection_reason(
        row,
        min_hops=min_hops,
        drop_generic_event=drop_generic_event,
        drop_event_hops=drop_event_hops,
        drop_weak_indirectness=drop_weak_indirectness,
        drop_unknown_relation=drop_unknown_relation,
    )
    if reason is not None:
        stats.rejected_by_reason[reason] += 1
        return

    stats.eligible += 1
    stats.eligible_by_type[query_type] += 1
    seen_by_type[query_type] += 1
    reservoir = reservoirs[query_type]
    cap = caps[query_type]
    if len(reservoir) < cap:
        reservoir.append(row)
        return

    replacement_index = rng.randrange(seen_by_type[query_type])
    if replacement_index < cap:
        reservoir[replacement_index] = row


def rejection_reason(
    row: dict[str, Any],
    *,
    min_hops: int,
    drop_generic_event: bool,
    drop_event_hops: bool,
    drop_weak_indirectness: bool,
    drop_unknown_relation: bool,
) -> str | None:
    messages = row.get("messages", [])
    if len(messages) < 3:
        return "missing_messages"
    prompt = str(messages[1].get("content", ""))
    answer = str(messages[2].get("content", "")).strip()
    merged_text = f"{prompt}\n{answer}"
    if not answer:
        return "empty_answer"
    if count_reasoning_hops(prompt) < min_hops:
        return "too_few_hops"
    if drop_generic_event and has_generic_event(merged_text):
        return "generic_event"
    if drop_event_hops and has_event_grounding_hop(prompt):
        return "event_hop"
    if drop_weak_indirectness:
        weak_indirectness_reason = weak_indirectness_rejection_reason(row, prompt)
        if weak_indirectness_reason is not None:
            return weak_indirectness_reason
    if drop_unknown_relation and ("\uc5c6\ub294 \uad00\uacc4" in merged_text or "unknown \uad00\uacc4" in merged_text):
        return "unknown_relation"
    return None


def get_query_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    return str(metadata.get("query_type", ""))


def count_reasoning_hops(prompt: str) -> int:
    return prompt.count(" --")


def has_generic_event(text: str) -> bool:
    return any(marker in text for marker in GENERIC_EVENT_MARKERS)


def has_event_grounding_hop(prompt: str) -> bool:
    return "--grounds_event-->" in prompt


def has_weak_softened_indirectness(row: dict[str, Any], prompt: str) -> bool:
    if get_query_type(row) != "indirect_speech_act":
        return False
    match = INDIRECTNESS_HOP_RE.search(prompt)
    if match is None:
        return False
    label = match.group("label").strip()
    evidence = match.group("evidence").strip()
    if label != "softened":
        return False
    has_weak_surface = any(marker in evidence for marker in WEAK_SOFTENING_SURFACE_MARKERS)
    has_strong_marker = any(marker in evidence for marker in STRONG_INDIRECTNESS_MARKERS)
    return has_weak_surface and not has_strong_marker


def weak_indirectness_rejection_reason(row: dict[str, Any], prompt: str) -> str | None:
    if get_query_type(row) != "indirect_speech_act":
        return None
    indirectness_match = INDIRECTNESS_HOP_RE.search(prompt)
    if indirectness_match is None:
        return "weak_indirect_missing_hop"
    label = indirectness_match.group("label").strip()
    speech_act_match = SPEECH_ACT_HOP_RE.search(prompt)
    speech_act = speech_act_match.group("label").strip() if speech_act_match else ""
    if label != "sarcastic" and speech_act not in STRONG_INDIRECT_SPEECH_ACTS:
        return "weak_indirect_speech_act"
    if has_weak_softened_indirectness(row, prompt):
        return "weak_softened_indirectness"
    if has_weak_implicit_indirectness(row, prompt):
        return "weak_implicit_indirectness"
    return None


def has_weak_implicit_indirectness(row: dict[str, Any], prompt: str) -> bool:
    if get_query_type(row) != "indirect_speech_act":
        return False
    indirectness_match = INDIRECTNESS_HOP_RE.search(prompt)
    if indirectness_match is None:
        return False
    label = indirectness_match.group("label").strip()
    evidence = indirectness_match.group("evidence").strip()
    if label != "implicit":
        return False
    speech_act_match = SPEECH_ACT_HOP_RE.search(prompt)
    speech_act = speech_act_match.group("label").strip() if speech_act_match else ""
    has_strong_speech_act = speech_act in STRONG_INDIRECT_SPEECH_ACTS
    has_strong_marker = any(marker in evidence for marker in STRONG_INDIRECTNESS_MARKERS)
    return not has_strong_speech_act and not has_strong_marker

from __future__ import annotations

from typing import Any

from kg_reasoning.schema import MultiHopQueryRecord, ReasoningHopRecord


NEXT_TURN_SYSTEM_PROMPT = (
    "너는 한국어 대화 응답 모델이다. 대화 이력과 참고 KG 근거를 바탕으로 "
    "다음 화자의 자연스럽고 일관된 한 발화를 생성하라. "
    "KG를 그대로 나열하지 말고 필요한 내용만 자연스럽게 반영하라."
)

KG_RESPONSE_SYSTEM_PROMPT = (
    "너는 한국어 상식 추론 도우미다. 질의와 KG 근거를 보고 "
    "상황 요약, 의도와 감정, 주변 반응을 근거 중심으로 설명하라."
)


def render_utterance_history(utterances: list[dict[str, Any]]) -> str:
    return "\n".join(f"{item['speaker']}: {item['text']}" for item in utterances)


MULTIHOP_KG_RESPONSE_SYSTEM_PROMPT = (
    "너는 한국어 대화 KG 추론 보조자다. 대화의 사건, 화용론 신호, 화자 관계를 연결한 "
    "multi-hop reasoning path를 근거로 사용자의 의도와 적절한 응답 방향을 설명하라. "
    "근거에 없는 내용을 지어내지 말고, 간접화행/말투/관계 단서를 자연스러운 한국어로 풀어라."
)


def render_kg_evidence(row: dict[str, Any], *, max_nodes: int = 3, max_triples: int = 6) -> str:
    lines = ["[Retrieved Events]"]
    for index, node in enumerate(row.get("retrieved_nodes", [])[:max_nodes], start=1):
        score = node.get("final_score")
        if isinstance(score, (int, float)):
            lines.append(f"{index}. {node['node_text']} (score={score:.4f})")
        else:
            lines.append(f"{index}. {node['node_text']}")
    lines.append("")
    lines.append("[Grounded Triples]")
    for triple in row.get("grounded_triples", [])[:max_triples]:
        lines.append(f"- {triple['head']} | {triple['relation']} | {triple['tail']}")
    return "\n".join(lines)


def build_next_turn_example(row: dict[str, Any]) -> dict[str, Any] | None:
    utterances = list(row.get("utterances", []))
    if len(utterances) < 2:
        return None
    history = utterances[:-1]
    target = utterances[-1]
    user_prompt = "\n".join(
        [
            f"[Topic]\n{row.get('topic', 'unknown')}",
            "",
            f"[Dialogue History]\n{render_utterance_history(history)}",
            "",
            f"[Next Speaker]\n{target['speaker']}",
            "",
            f"[KG Evidence]\n{render_kg_evidence(row)}",
            "",
            "[Instruction]\n다음 화자의 한 발화를 한국어로 자연스럽게 생성하라.",
        ]
    )
    return {
        "messages": [
            {"role": "system", "content": NEXT_TURN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": str(target["text"])},
        ],
        "metadata": {
            "task": "next-turn",
            "dialogue_id": row.get("dialogue_id"),
            "topic": row.get("topic"),
            "target_speaker": target.get("speaker"),
            "query_source": row.get("query_source"),
            "evidence_source": row.get("evidence_source"),
        },
    }


def build_kg_response_example(row: dict[str, Any]) -> dict[str, Any]:
    user_prompt = "\n".join(
        [
            f"[Topic]\n{row.get('topic', 'unknown')}",
            "",
            f"[User Query]\n{row.get('query', '')}",
            "",
            f"[KG Evidence]\n{render_kg_evidence(row)}",
            "",
            "[Instruction]\n상황 요약, 의도와 감정, 주변 반응을 한국어로 설명하라.",
        ]
    )
    return {
        "messages": [
            {"role": "system", "content": KG_RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": str(row.get("kg_aware", ""))},
        ],
        "metadata": {
            "task": "kg-response",
            "dialogue_id": row.get("dialogue_id"),
            "topic": row.get("topic"),
            "query_source": row.get("query_source"),
            "evidence_source": row.get("evidence_source"),
            "baseline": row.get("baseline", ""),
        },
    }


def build_multihop_training_rows(
    *,
    queries: list[MultiHopQueryRecord],
    hops: list[ReasoningHopRecord],
) -> list[dict[str, Any]]:
    hops_by_id = {hop.hop_id: hop for hop in hops}
    rows: list[dict[str, Any]] = []
    for query in queries:
        supporting_hops = [
            hops_by_id[hop_id].to_dict()
            for hop_id in query.supporting_hop_ids
            if hop_id in hops_by_id
        ]
        rows.append(
            {
                "dialogue_id": query.dialogue_id,
                "query_id": query.query_id,
                "query_type": query.query_type,
                "question": query.question,
                "answer": query.answer,
                "difficulty": query.difficulty,
                "reasoning_focus": list(query.reasoning_focus),
                "supporting_hop_ids": list(query.supporting_hop_ids),
                "supporting_hops": supporting_hops,
                "target_utterance_id": query.target_utterance_id,
                "metadata": dict(query.metadata),
            }
        )
    return rows


def render_reasoning_hops(row: dict[str, Any], *, max_hops: int = 6) -> str:
    lines = ["[Reasoning Hops]"]
    for index, hop in enumerate(row.get("supporting_hops", [])[:max_hops], start=1):
        attributes = hop.get("attributes", {})
        label = attributes.get("label") or attributes.get("relation_label") or attributes.get("event_text")
        label_suffix = f" | label={label}" if label else ""
        lines.append(
            "- "
            f"{index}. {hop.get('source_kind')}:{hop.get('source_id')} "
            f"--{hop.get('relation')}--> "
            f"{hop.get('target_kind')}:{hop.get('target_id')}"
            f"{label_suffix}"
        )
        evidence = str(hop.get("evidence_text", "")).strip()
        if evidence:
            lines.append(f"  evidence: {evidence}")
    if len(lines) == 1:
        lines.append("- no supporting hops")
    return "\n".join(lines)


def build_multihop_kg_response_example(row: dict[str, Any]) -> dict[str, Any]:
    focus = ", ".join(str(item) for item in row.get("reasoning_focus", [])) or "unknown"
    user_prompt = "\n".join(
        [
            f"[Question]\n{row.get('question', '')}",
            "",
            f"[Query Type]\n{row.get('query_type', '')}",
            "",
            f"[Reasoning Focus]\n{focus}",
            "",
            render_reasoning_hops(row),
            "",
            "[Instruction]\n위 hop들을 연결해서 한국어 대화 맥락의 의도, 화용론 단서, 응답 방향을 설명하라.",
        ]
    )
    return {
        "messages": [
            {"role": "system", "content": MULTIHOP_KG_RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": str(row.get("answer", ""))},
        ],
        "metadata": {
            "task": "multi-hop-kg-response",
            "dialogue_id": row.get("dialogue_id"),
            "query_id": row.get("query_id"),
            "query_type": row.get("query_type"),
            "difficulty": row.get("difficulty"),
            "reasoning_focus": row.get("reasoning_focus", []),
            "supporting_hop_ids": row.get("supporting_hop_ids", []),
        },
    }


def build_training_examples(rows: list[dict[str, Any]], *, task: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if task == "next-turn":
            example = build_next_turn_example(row)
            if example is None:
                continue
            examples.append(example)
            continue
        if task == "kg-response":
            examples.append(build_kg_response_example(row))
            continue
        if task == "multi-hop-kg-response":
            examples.append(build_multihop_kg_response_example(row))
            continue
        raise ValueError(f"Unsupported task: {task}")
    return examples

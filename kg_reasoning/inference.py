from __future__ import annotations

from collections import defaultdict
from typing import Any

from kg_reasoning.heuristics import build_relation_bundle, choose_event_text
from kg_reasoning.llm import OpenAICompatibleClient
from kg_reasoning.retrieval import RetrievalIndex, SearchResult
from kg_reasoning.schema import TripleRecord
from kg_reasoning.text import unique_preserving_order


RELATION_LABELS = {
    "xIntent": "의도",
    "xNeed": "전제 조건",
    "xEffect": "직접 결과",
    "xReact": "감정",
    "xWant": "다음 바람",
    "oEffect": "주변 영향",
    "oReact": "주변 반응",
    "oWant": "주변 바람",
    "xAttr": "화자 특성",
}


def compare_reasoning(
    query: str,
    graph: dict[str, Any],
    encoder: dict[str, Any],
    triples: list[TripleRecord],
    top_k: int = 5,
    client: OpenAICompatibleClient | None = None,
    mode: str = "template",
) -> dict[str, Any]:
    index = RetrievalIndex(graph=graph, encoder=encoder)
    query_event_text = choose_event_text(query)
    search_results = index.search(query=query, top_k=top_k)
    grounded_triples = collect_grounded_triples(
        search_results=search_results,
        triples=triples,
        query_event_text=query_event_text,
        cap=12,
    )
    baseline = build_baseline_response(query=query)
    if mode == "llm" and client:
        kg_aware = build_llm_response(query=query, grounded_triples=grounded_triples, client=client)
    else:
        kg_aware = build_template_response(query=query, search_results=search_results, grounded_triples=grounded_triples)
    return {
        "baseline": baseline,
        "kg_aware": kg_aware,
        "retrieved_nodes": [result.to_dict() for result in search_results],
        "grounded_triples": [triple.to_dict() for triple in grounded_triples],
    }


def collect_grounded_triples(
    search_results: list[SearchResult],
    triples: list[TripleRecord],
    query_event_text: str,
    cap: int = 12,
) -> list[TripleRecord]:
    ordered_heads = [query_event_text]
    ordered_heads.extend(result.node_text for result in search_results)
    grounded: list[TripleRecord] = []
    for head in ordered_heads:
        for triple in triples:
            if triple.head != head:
                continue
            if triple in grounded:
                continue
            grounded.append(triple)
            if len(grounded) >= cap:
                return grounded
    return grounded


def build_baseline_response(query: str) -> str:
    event_text = choose_event_text(query)
    generic = build_relation_bundle(event_text)
    return (
        f"질문만 보면 화자는 '{event_text}'에 가까운 상황으로 보입니다. "
        f"감정적으로는 {generic['xReact'].rstrip('.')} 정도를 추정할 수 있지만, "
        "원인-결과 사슬이나 주변 반응까지는 근거가 부족합니다."
    )


def build_template_response(
    query: str,
    search_results: list[SearchResult],
    grounded_triples: list[TripleRecord],
) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for triple in grounded_triples:
        grouped[triple.relation].append(triple.tail)

    grounded_heads = unique_preserving_order([triple.head for triple in grounded_triples])
    visible_events = grounded_heads[:3] or [result.node_text for result in search_results[:3]]
    top_events = ", ".join(visible_events) or "관련 이벤트 없음"
    summary_lines = [
        f"입력 해석: {query}",
        f"가장 가까운 이벤트: {top_events}",
    ]
    for relation in ("xIntent", "xReact", "xEffect", "oReact", "oWant", "xAttr"):
        tails = grouped.get(relation)
        if not tails:
            continue
        summary_lines.append(f"{RELATION_LABELS[relation]}: {tails[0]}")
    summary_lines.append("근거:")
    for triple in grounded_triples[:6]:
        summary_lines.append(f"- {triple.head} -> {triple.relation}: {triple.tail}")
    return "\n".join(summary_lines)


def build_llm_response(
    query: str,
    grounded_triples: list[TripleRecord],
    client: OpenAICompatibleClient,
) -> str:
    context = "\n".join(
        f"- {triple.head} / {triple.relation} / {triple.tail}" for triple in grounded_triples[:10]
    )
    system_prompt = (
        "You are a Korean commonsense reasoning assistant. "
        "Use the supplied KG evidence to build a grounded answer."
    )
    user_prompt = (
        f"[사용자 발화]\n{query}\n\n"
        f"[KG 근거]\n{context}\n\n"
        "한국어로 답하되, 상황 요약, 의도/감정, 주변 반응을 구분해 간결하게 설명하세요."
    )
    return client.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)

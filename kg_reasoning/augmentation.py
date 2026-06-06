from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

import numpy as np

from kg_reasoning.heuristics import choose_event_text
from kg_reasoning.inference import build_baseline_response, build_llm_response, build_template_response
from kg_reasoning.llm import OpenAICompatibleClient
from kg_reasoning.retrieval import RetrievalIndex, SearchResult
from kg_reasoning.schema import DialogueRecord, TripleRecord
from kg_reasoning.text import char_ngrams, cosine_similarity, hash_vector, tokenize


EVENT_PREFIX_RE = re.compile(r"^(한 참여자는|다른 친구는|상대방은|또 다른 친구는|한 친구는)\s+")
EVENT_SUFFIX_RE = re.compile(
    r"\s*(라고|다고)?\s*(말했다|언급했다|지적했다|강조했다|설명했다|공유했다|회상했다|주장했다|다짐했다|질문했다|표현했다)\.?$"
)
COMMON_SUFFIXES = (
    "으로는",
    "에서는",
    "에게는",
    "한테는",
    "이라고",
    "라고",
    "으로",
    "에서",
    "에게",
    "한테",
    "까지",
    "부터",
    "보다",
    "처럼",
    "이라",
    "이다",
    "이고",
    "인데",
    "이라도",
    "도",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "와",
    "과",
    "에",
    "의",
    "로",
    "만",
    "나",
    "요",
    "다",
)


def build_dialogue_query(
    dialogue: DialogueRecord,
    query_source: str = "last-utterance",
    *,
    history_window: int | None = 6,
) -> str:
    if not dialogue.utterances:
        return ""
    if query_source == "history":
        history = dialogue.utterances[:-1]
        if not history:
            return ""
        if history_window is not None and history_window > 0:
            history = history[-history_window:]
        return " ".join(utterance.text for utterance in history)
    if query_source == "full-dialogue":
        return " ".join(utterance.text for utterance in dialogue.utterances)
    return dialogue.utterances[-1].text


def build_augmentation_context(
    *,
    query: str,
    retrieved_nodes: list[dict[str, Any]],
    grounded_triples: list[dict[str, Any]],
    max_nodes: int = 3,
    max_triples: int = 6,
) -> str:
    lines = [
        "[Query]",
        query,
        "",
        "[Retrieved Events]",
    ]
    for index, node in enumerate(retrieved_nodes[:max_nodes], start=1):
        lines.append(f"{index}. {node['node_text']} (score={node['final_score']:.4f})")
    lines.append("")
    lines.append("[Grounded Triples]")
    for triple in grounded_triples[:max_triples]:
        lines.append(f"- {triple['head']} | {triple['relation']} | {triple['tail']}")
    return "\n".join(lines)


def collect_grounded_triples_fast(
    *,
    search_results: list[SearchResult],
    triple_lookup: dict[str, list[TripleRecord]],
    query_event_text: str,
    cap: int = 12,
) -> list[TripleRecord]:
    ordered_heads = [query_event_text]
    ordered_heads.extend(result.node_text for result in search_results)
    grounded: list[TripleRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for head in ordered_heads:
        for triple in triple_lookup.get(head, []):
            triple_key = (triple.head, triple.relation, triple.tail)
            if triple_key in seen:
                continue
            seen.add(triple_key)
            grounded.append(triple)
            if len(grounded) >= cap:
                return grounded
    return grounded


def build_dialogue_linked_nodes(grounded_triples: list[TripleRecord], max_nodes: int = 3) -> list[dict[str, Any]]:
    heads: list[str] = []
    for triple in grounded_triples:
        if triple.head not in heads:
            heads.append(triple.head)
        if len(heads) >= max_nodes:
            break
    return [
        {
            "node_id": -1,
            "node_text": head,
            "lexical_score": 1.0,
            "structural_score": 1.0,
            "final_score": 1.0,
        }
        for head in heads
    ]


def normalize_event_surface(text: str) -> str:
    stripped = EVENT_PREFIX_RE.sub("", text.strip())
    stripped = EVENT_SUFFIX_RE.sub("", stripped)
    return stripped.strip() or text.strip()


def normalize_keyword_token(token: str) -> str:
    candidate = token.strip()
    for suffix in COMMON_SUFFIXES:
        if len(candidate) > len(suffix) + 1 and candidate.endswith(suffix):
            return candidate[: -len(suffix)]
    return candidate


def rank_dialogue_events(
    *,
    query: str,
    dialogue_triples: list[TripleRecord],
    top_events: int = 3,
) -> tuple[list[dict[str, Any]], list[TripleRecord]]:
    grouped: dict[str, list[TripleRecord]] = defaultdict(list)
    for triple in dialogue_triples:
        grouped[triple.event_id].append(triple)
    if not grouped:
        return [], []

    query_vector = hash_vector(query, dim=96)
    query_tokens = {normalize_keyword_token(token) for token in tokenize(query)}
    query_ngrams = set(char_ngrams(query, n=2))
    ranked: list[tuple[float, str, list[TripleRecord]]] = []
    for event_id, rows in grouped.items():
        head = rows[0].head
        normalized_head = normalize_event_surface(head)
        head_vector = hash_vector(normalized_head, dim=96)
        lexical = float(cosine_similarity(query_vector, np.asarray([head_vector], dtype=np.float64))[0])
        head_tokens = {normalize_keyword_token(token) for token in tokenize(normalized_head)}
        head_ngrams = set(char_ngrams(normalized_head, n=2))
        overlap = len(query_tokens & head_tokens) / max(len(head_tokens), 1)
        char_overlap = len(query_ngrams & head_ngrams) / max(len(head_ngrams), 1)
        keyword_hits = sum(
            1
            for token in query_tokens
            if len(token) > 1 and (token in normalized_head or token in normalized_head.replace(" ", ""))
        )
        score = (0.20 * lexical) + (0.15 * overlap) + (0.25 * char_overlap) + (0.40 * min(keyword_hits, 3))
        ranked.append((score, event_id, rows))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[:top_events]
    nodes = [
        {
            "node_id": -1,
            "node_text": rows[0].head,
            "lexical_score": score,
            "structural_score": 0.0,
            "final_score": score,
        }
        for score, _, rows in selected
    ]
    grounded: list[TripleRecord] = []
    for _, _, rows in selected:
        grounded.extend(rows)
    return nodes, grounded[:12]


def augment_dialogues(
    *,
    dialogues: list[DialogueRecord],
    graph: dict[str, Any],
    encoder: dict[str, Any],
    triples: list[TripleRecord],
    top_k: int = 5,
    mode: str = "template",
    client: OpenAICompatibleClient | None = None,
    query_source: str = "last-utterance",
    evidence_source: str = "dialogue-linked",
    history_window: int | None = 6,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = dialogues[:limit] if limit is not None else dialogues
    index: RetrievalIndex | None = None
    triple_lookup: dict[str, list[TripleRecord]] = defaultdict(list)
    dialogue_lookup: dict[str, list[TripleRecord]] = defaultdict(list)
    if evidence_source == "retrieval":
        index = RetrievalIndex(graph=graph, encoder=encoder)
        for triple in triples:
            triple_lookup[triple.head].append(triple)
    for triple in triples:
        dialogue_lookup[triple.dialogue_id].append(triple)
    augmented: list[dict[str, Any]] = []
    for dialogue in selected:
        query = build_dialogue_query(dialogue, query_source=query_source, history_window=history_window)
        if evidence_source == "retrieval":
            assert index is not None
            query_event_text = choose_event_text(query)
            search_results = index.search(query=query, top_k=top_k)
            grounded_triples = collect_grounded_triples_fast(
                search_results=search_results,
                triple_lookup=triple_lookup,
                query_event_text=query_event_text,
            )
            retrieved_nodes = [item.to_dict() for item in search_results]
        elif evidence_source == "history-linked":
            search_results = []
            retrieved_nodes, grounded_triples = rank_dialogue_events(
                query=query,
                dialogue_triples=dialogue_lookup.get(dialogue.dialogue_id, []),
                top_events=max(top_k, 1),
            )
        else:
            search_results = []
            grounded_triples = dialogue_lookup.get(dialogue.dialogue_id, [])[:12]
            retrieved_nodes = build_dialogue_linked_nodes(grounded_triples)
        baseline = build_baseline_response(query=query)
        if mode == "llm" and client is not None:
            kg_aware = build_llm_response(query=query, grounded_triples=grounded_triples, client=client)
        else:
            kg_aware = build_template_response(
                query=query,
                search_results=search_results,
                grounded_triples=grounded_triples,
            )
        context = build_augmentation_context(
            query=query,
            retrieved_nodes=retrieved_nodes,
            grounded_triples=[item.to_dict() for item in grounded_triples],
        )
        augmented.append(
            {
                "dialogue_id": dialogue.dialogue_id,
                "topic": dialogue.topic,
                "query": query,
                "query_source": query_source,
                "history_window": history_window,
                "utterances": [utterance.to_dict() for utterance in dialogue.utterances],
                "evidence_source": evidence_source,
                "augmentation_context": context,
                "baseline": baseline,
                "kg_aware": kg_aware,
                "retrieved_nodes": retrieved_nodes,
                "grounded_triples": [item.to_dict() for item in grounded_triples],
            }
        )
    return augmented

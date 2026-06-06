from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kg_reasoning.heuristics import choose_event_text, choose_primary_tag
from kg_reasoning.text import cosine_similarity, hash_vector, tokenize


@dataclass(slots=True)
class SearchResult:
    node_id: int
    node_text: str
    lexical_score: float
    structural_score: float
    final_score: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "node_id": self.node_id,
            "node_text": self.node_text,
            "lexical_score": self.lexical_score,
            "structural_score": self.structural_score,
            "final_score": self.final_score,
        }


class RetrievalIndex:
    def __init__(self, graph: dict[str, Any], encoder: dict[str, Any]) -> None:
        self.graph = graph
        self.all_node_texts = list(graph["nodes"])
        all_text_vectors = np.asarray(encoder["node_text_vectors"], dtype=np.float64)
        all_node_embeddings = np.asarray(encoder["node_embeddings"], dtype=np.float64)

        event_node_ids = [int(item) for item in graph.get("event_node_ids", list(range(len(self.all_node_texts))))]
        self.active_node_ids = event_node_ids
        self.node_texts = [self.all_node_texts[index] for index in self.active_node_ids]
        self.node_tokens = [set(tokenize(text)) for text in self.node_texts]
        self.node_tags = [choose_primary_tag(text) for text in self.node_texts]
        self.text_vectors = all_text_vectors[self.active_node_ids]
        self.node_embeddings = all_node_embeddings[self.active_node_ids]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self.node_texts:
            return []
        query_vector = hash_vector(query, dim=self.text_vectors.shape[1])
        query_tokens = set(tokenize(query))
        query_tag = choose_primary_tag(query)
        expected_event_text = choose_event_text(query)
        lexical_scores = cosine_similarity(query_vector, self.text_vectors)
        structural_scores = cosine_similarity(query_vector, self.node_embeddings)
        overlap_scores = np.asarray(
            [
                (len(query_tokens & node_tokens) / max(len(query_tokens), 1))
                for node_tokens in self.node_tokens
            ],
            dtype=np.float64,
        )
        tag_scores = np.asarray(
            [1.0 if query_tag == node_tag else 0.0 for node_tag in self.node_tags],
            dtype=np.float64,
        )
        event_match_scores = np.asarray(
            [1.0 if expected_event_text == node_text else 0.0 for node_text in self.node_texts],
            dtype=np.float64,
        )
        tag_weight = 0.10 if query_tag != "generic" else 0.0
        event_weight = 0.25 if query_tag != "generic" else 0.0
        final_scores = (
            (0.40 * lexical_scores)
            + (0.15 * structural_scores)
            + (0.10 * overlap_scores)
            + (tag_weight * tag_scores)
            + (event_weight * event_match_scores)
        )
        sorted_indices = np.argsort(final_scores)[::-1][:top_k]
        results: list[SearchResult] = []
        for local_index in sorted_indices.tolist():
            node_id = self.active_node_ids[local_index]
            results.append(
                SearchResult(
                    node_id=node_id,
                    node_text=self.all_node_texts[node_id],
                    lexical_score=float(lexical_scores[local_index]),
                    structural_score=float(structural_scores[local_index]),
                    final_score=float(final_scores[local_index]),
                )
            )
        return results

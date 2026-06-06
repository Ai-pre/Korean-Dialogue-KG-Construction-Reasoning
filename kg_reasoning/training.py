from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kg_reasoning.io import read_json, write_json
from kg_reasoning.legacy_rgcn import LegacyRGCNConfig, train_legacy_rgcn
from kg_reasoning.text import hash_vector, sigmoid


@dataclass(slots=True)
class TrainingConfig:
    dim: int = 96
    epochs: int = 120
    learning_rate: float = 0.06
    layers: int = 2
    seed: int = 7
    backend: str = "lightweight"
    hidden_dim: int = 128


def train_relational_encoder(graph: dict[str, Any], config: TrainingConfig | None = None) -> dict[str, Any]:
    cfg = config or TrainingConfig()
    if cfg.backend == "legacy-rgcn":
        legacy_config = LegacyRGCNConfig(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.layers,
            epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
        )
        return train_legacy_rgcn(graph=graph, config=legacy_config)
    rng = np.random.default_rng(cfg.seed)

    nodes = graph["nodes"]
    edges = graph["edges"]
    num_nodes = len(nodes)
    num_relations = len(graph["relation_to_id"])

    node_vectors = np.vstack([hash_vector(text, dim=cfg.dim) for text in nodes]).astype(np.float64)
    node_vectors += rng.normal(loc=0.0, scale=0.01, size=node_vectors.shape)
    relation_vectors = rng.normal(loc=0.0, scale=0.05, size=(num_relations, cfg.dim))

    edge_triplets = [(edge["head"], edge["relation"], edge["tail"]) for edge in edges]
    if not edge_triplets:
        raise ValueError("Cannot train an encoder without graph edges.")

    for _ in range(cfg.epochs):
        rng.shuffle(edge_triplets)
        for head_index, relation_index, tail_index in edge_triplets:
            negative_tail = int(rng.integers(0, num_nodes))
            if negative_tail == tail_index:
                negative_tail = (negative_tail + 1) % num_nodes

            positive_score = float(np.sum(node_vectors[head_index] * relation_vectors[relation_index] * node_vectors[tail_index]))
            negative_score = float(np.sum(node_vectors[head_index] * relation_vectors[relation_index] * node_vectors[negative_tail]))

            positive_factor = sigmoid(-positive_score)
            negative_factor = sigmoid(negative_score)

            positive_head = relation_vectors[relation_index] * node_vectors[tail_index]
            positive_tail = relation_vectors[relation_index] * node_vectors[head_index]
            positive_relation = node_vectors[head_index] * node_vectors[tail_index]

            node_vectors[head_index] += cfg.learning_rate * positive_factor * positive_head
            node_vectors[tail_index] += cfg.learning_rate * positive_factor * positive_tail
            relation_vectors[relation_index] += cfg.learning_rate * positive_factor * positive_relation

            negative_head = relation_vectors[relation_index] * node_vectors[negative_tail]
            negative_tail_vec = relation_vectors[relation_index] * node_vectors[head_index]
            negative_relation = node_vectors[head_index] * node_vectors[negative_tail]

            node_vectors[head_index] -= cfg.learning_rate * negative_factor * negative_head
            node_vectors[negative_tail] -= cfg.learning_rate * negative_factor * negative_tail_vec
            relation_vectors[relation_index] -= cfg.learning_rate * negative_factor * negative_relation

        node_vectors = _normalize_rows(node_vectors)
        relation_vectors = _normalize_rows(relation_vectors)

    contextualized = _relational_message_passing(
        node_vectors=node_vectors,
        relation_vectors=relation_vectors,
        edges=edge_triplets,
        layers=cfg.layers,
    )
    metrics = evaluate_encoder(node_vectors=contextualized, relation_vectors=relation_vectors, edges=edge_triplets, rng=rng)
    return {
        "node_embeddings": contextualized.tolist(),
        "node_text_vectors": np.vstack([hash_vector(text, dim=cfg.dim) for text in nodes]).tolist(),
        "relation_embeddings": relation_vectors.tolist(),
        "config": {
            "backend": cfg.backend,
            "dim": cfg.dim,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "layers": cfg.layers,
            "seed": cfg.seed,
            "hidden_dim": cfg.hidden_dim,
        },
        "metrics": metrics,
    }


def _relational_message_passing(
    node_vectors: np.ndarray,
    relation_vectors: np.ndarray,
    edges: list[tuple[int, int, int]],
    layers: int,
) -> np.ndarray:
    state = node_vectors.copy()
    for _ in range(layers):
        aggregate = state.copy()
        counts = np.ones((state.shape[0], 1), dtype=np.float64)
        for head_index, relation_index, tail_index in edges:
            forward_message = state[tail_index] * relation_vectors[relation_index]
            backward_message = state[head_index] * relation_vectors[relation_index]
            aggregate[head_index] += forward_message
            aggregate[tail_index] += 0.5 * backward_message
            counts[head_index] += 1.0
            counts[tail_index] += 0.5
        state = np.tanh(aggregate / counts)
        state = _normalize_rows(state)
    return state


def evaluate_encoder(
    node_vectors: np.ndarray,
    relation_vectors: np.ndarray,
    edges: list[tuple[int, int, int]],
    rng: np.random.Generator,
) -> dict[str, float]:
    positives = []
    negatives = []
    node_count = node_vectors.shape[0]
    for head_index, relation_index, tail_index in edges:
        positives.append(float(np.sum(node_vectors[head_index] * relation_vectors[relation_index] * node_vectors[tail_index])))
        negative_tail = int(rng.integers(0, node_count))
        if negative_tail == tail_index:
            negative_tail = (negative_tail + 1) % node_count
        negatives.append(float(np.sum(node_vectors[head_index] * relation_vectors[relation_index] * node_vectors[negative_tail])))
    margin = float(np.mean(positives) - np.mean(negatives))
    return {
        "avg_positive_score": float(np.mean(positives)),
        "avg_negative_score": float(np.mean(negatives)),
        "score_margin": margin,
    }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms


def save_encoder(path: str | Path, payload: dict[str, Any]) -> Path:
    return write_json(path, payload)


def load_encoder(path: str | Path) -> dict[str, Any]:
    return read_json(path)

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from kg_reasoning.io import read_json, write_json
from kg_reasoning.schema import RELATION_TYPES, TripleRecord


def build_graph(triples: list[TripleRecord]) -> dict[str, Any]:
    node_to_id: dict[str, int] = {}
    relation_to_id = {relation: index for index, relation in enumerate(RELATION_TYPES)}
    edges: list[dict[str, int]] = []
    event_node_ids: list[int] = []

    def get_node_id(text: str) -> int:
        if text not in node_to_id:
            node_to_id[text] = len(node_to_id)
        return node_to_id[text]

    for triple in triples:
        head_id = get_node_id(triple.head)
        tail_id = get_node_id(triple.tail)
        if head_id not in event_node_ids:
            event_node_ids.append(head_id)
        edges.append(
            {
                "head": head_id,
                "tail": tail_id,
                "relation": relation_to_id[triple.relation],
                "dialogue_id": triple.dialogue_id,
                "event_id": triple.event_id,
            }
        )

    adjacency: dict[str, list[dict[str, int]]] = defaultdict(list)
    for edge in edges:
        adjacency[str(edge["head"])].append(edge)

    id_to_node = {index: text for text, index in node_to_id.items()}
    return {
        "nodes": [id_to_node[index] for index in range(len(id_to_node))],
        "node_to_id": node_to_id,
        "relation_to_id": relation_to_id,
        "event_node_ids": event_node_ids,
        "event_nodes": [id_to_node[index] for index in event_node_ids],
        "edges": edges,
        "adjacency": adjacency,
    }


def save_graph(path: str | Path, graph: dict[str, Any]) -> Path:
    return write_json(path, graph)


def load_graph(path: str | Path) -> dict[str, Any]:
    return read_json(path)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LegacyRGCNConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


def train_legacy_rgcn(graph: dict[str, Any], config: LegacyRGCNConfig | None = None) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
    except ImportError as exc:
        raise RuntimeError("legacy-rgcn backend requires `torch`. Install torch, then rerun.") from exc

    cfg = config or LegacyRGCNConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class RelationalGraphConv(nn.Module):
        def __init__(self, hidden_dim: int, num_relations: int) -> None:
            super().__init__()
            self.relation_weights = nn.Parameter(torch.empty(num_relations, hidden_dim, hidden_dim))
            self.self_loop = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            nn.init.xavier_uniform_(self.relation_weights)
            nn.init.xavier_uniform_(self.self_loop.weight)

        def forward(self, hidden, edge_index, edge_type):
            aggregate = self.self_loop(hidden)
            counts = torch.ones((hidden.size(0), 1), device=hidden.device, dtype=hidden.dtype)
            for relation_index in range(self.relation_weights.size(0)):
                mask = edge_type == relation_index
                if not torch.any(mask):
                    continue
                heads = edge_index[0, mask]
                tails = edge_index[1, mask]
                messages = hidden[heads] @ self.relation_weights[relation_index]
                aggregate.index_add_(0, tails, messages)
                counts.index_add_(
                    0,
                    tails,
                    torch.ones((tails.numel(), 1), device=hidden.device, dtype=hidden.dtype),
                )
            return torch.relu((aggregate + self.bias) / counts.clamp_min(1.0))

    class RGCN(nn.Module):
        def __init__(self, num_nodes: int, num_relations: int, hidden_dim: int, num_layers: int) -> None:
            super().__init__()
            self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
            self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
            self.layers = nn.ModuleList(
                [RelationalGraphConv(hidden_dim=hidden_dim, num_relations=num_relations) for _ in range(num_layers)]
            )
            nn.init.xavier_uniform_(self.node_embedding.weight)
            nn.init.xavier_uniform_(self.relation_embedding.weight)

        def encode(self, edge_index, edge_type):
            hidden = self.node_embedding.weight
            for layer in self.layers:
                hidden = layer(hidden, edge_index, edge_type)
                hidden = F.normalize(hidden, dim=-1)
            return hidden

        def score_triplets(self, node_embeddings, heads, relations, tails):
            relation_vectors = self.relation_embedding(relations)
            return (node_embeddings[heads] * relation_vectors * node_embeddings[tails]).sum(dim=-1)

    if not graph["edges"]:
        raise ValueError("Cannot train an R-GCN encoder without graph edges.")

    edge_index = torch.tensor(
        [[edge["head"] for edge in graph["edges"]], [edge["tail"] for edge in graph["edges"]]],
        dtype=torch.long,
        device=device,
    )
    edge_type = torch.tensor([edge["relation"] for edge in graph["edges"]], dtype=torch.long, device=device)
    num_nodes = len(graph["nodes"])
    num_relations = len(graph["relation_to_id"])
    model = RGCN(num_nodes, num_relations, cfg.hidden_dim, cfg.num_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    num_edges = edge_index.shape[1]
    last_loss = 0.0

    for _ in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        node_embeddings = model.encode(edge_index, edge_type)

        positive_score = model.score_triplets(node_embeddings, edge_index[0], edge_type, edge_index[1])
        negative_tails = torch.randint(0, num_nodes, (num_edges,), device=device)
        collision_mask = negative_tails == edge_index[1]
        if torch.any(collision_mask):
            negative_tails = negative_tails.clone()
            negative_tails[collision_mask] = (negative_tails[collision_mask] + 1) % num_nodes
        negative_score = model.score_triplets(node_embeddings, edge_index[0], edge_type, negative_tails)

        labels = torch.cat([torch.ones_like(positive_score), torch.zeros_like(negative_score)])
        scores = torch.cat([positive_score, negative_score])
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        final_node_embeddings = model.encode(edge_index, edge_type)
        positive_score = model.score_triplets(final_node_embeddings, edge_index[0], edge_type, edge_index[1])
        negative_tails = torch.randint(0, num_nodes, (num_edges,), device=device)
        collision_mask = negative_tails == edge_index[1]
        if torch.any(collision_mask):
            negative_tails = negative_tails.clone()
            negative_tails[collision_mask] = (negative_tails[collision_mask] + 1) % num_nodes
        negative_score = model.score_triplets(final_node_embeddings, edge_index[0], edge_type, negative_tails)
        relation_embeddings = model.relation_embedding.weight.detach().cpu()
        node_text_vectors = F.normalize(model.node_embedding.weight.detach(), dim=-1).cpu()
        final_node_embeddings = final_node_embeddings.detach().cpu()

    return {
        "node_embeddings": final_node_embeddings.tolist(),
        "node_text_vectors": node_text_vectors.tolist(),
        "relation_embeddings": relation_embeddings.tolist(),
        "config": {
            "backend": "legacy-rgcn",
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "device": str(device),
        },
        "metrics": {
            "final_loss": last_loss,
            "avg_positive_score": float(positive_score.mean().item()),
            "avg_negative_score": float(negative_score.mean().item()),
            "score_margin": float((positive_score.mean() - negative_score.mean()).item()),
        },
    }

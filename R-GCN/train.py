# train.py
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from rgcn_model import RGCN


def negative_sampling(edge_index, num_nodes, num_neg=1):
    # simple corrupted tail negative sampling
    neg_edges = []
    E = edge_index.shape[1]

    for i in range(E):
        h = edge_index[0, i]
        for _ in range(num_neg):
            t = torch.randint(0, num_nodes, (1,))
            neg_edges.append([h, t.item()])

    neg_edges = torch.tensor(neg_edges).t().long()
    return neg_edges


def train():
    data = pickle.load(open("/home/jaesang/kg_project/output/graph_data.pkl", "rb"))

    edge_index = data["edge_index"].cuda()
    edge_type = data["edge_type"].cuda()
    num_nodes = data["num_nodes"]
    num_rel = len(data["rel2id"])

    model = RGCN(num_nodes, num_rel).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    node_ids = torch.arange(num_nodes).cuda()

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        # Forward: get node embeddings
        node_emb = model(node_ids, edge_index, edge_type)

        # Positive score
        h = node_emb[edge_index[0]]
        t = node_emb[edge_index[1]]
        pos_score = (h * t).sum(dim=-1)

        # Negative edges
        neg_edge_index = negative_sampling(edge_index, num_nodes).cuda()
        neg_h = node_emb[neg_edge_index[0]]
        neg_t = node_emb[neg_edge_index[1]]
        neg_score = (neg_h * neg_t).sum(dim=-1)

        # Labels
        y = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        scores = torch.cat([pos_score, neg_score])

        loss = criterion(scores, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "/home/jaesang/kg_project/output/rgcn_model.pt")
    print("Model saved to rgcn_model.pt")


if __name__ == "__main__":
    train()

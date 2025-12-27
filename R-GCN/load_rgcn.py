# load_rgcn.py
import pickle
import torch
from rgcn_model import RGCN

def load_rgcn_model(
    path_graph="/home/jaesang/kg_project/output/graph_data.pkl",
    path_weight="/home/jaesang/kg_project/output/rgcn_model.pt"
):
    data = pickle.load(open(path_graph, "rb"))

    num_nodes = data["num_nodes"]
    num_rel = len(data["rel2id"])

    model = RGCN(num_nodes, num_rel)
    model.load_state_dict(torch.load(path_weight, map_location="cpu"))
    model.eval()

    node_ids = torch.arange(num_nodes)
    with torch.no_grad():
        node_emb = model(
            node_ids,
            data["edge_index"],
            data["edge_type"]
        )

    return {
        "node_emb": node_emb,
        "node2id": data["node2id"],
        "id2node": {v: k for k, v in data["node2id"].items()}
    }

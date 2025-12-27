# build_graph.py
import json
import torch
from tqdm import tqdm
from collections import defaultdict
import pickle

REL_TYPES = [
    "xIntent", "xNeed", "xEffect", "xReact", "xAttr", "xWant",
    "oEffect", "oReact", "oWant"
]

rel2id = {r: i for i, r in enumerate(REL_TYPES)}

def load_triples(json_path):
    data = json.load(open(json_path, "r", encoding="utf-8"))
    triples = []
    for doc in data:
        for t in doc["triples"]:
            triples.append({
                "head": t["head"],
                "relation": t["relation"],
                "tail": t["tail"]
            })
    return triples


def build_graph(json_path, save_path="./graph_data.pkl"):
    triples = load_triples(json_path)

    node2id = {}
    edges = defaultdict(list)

    def get_id(x):
        if x not in node2id:
            node2id[x] = len(node2id)
        return node2id[x]

    print("Building graph...")
    for t in tqdm(triples):
        h = get_id(t["head"])
        tail = get_id(t["tail"])
        r = rel2id[t["relation"]]
        edges[r].append((h, tail))

    # PyG edge_index, edge_type 생성
    edge_index_list = []
    edge_type_list = []

    for r, pairs in edges.items():
        for h, t in pairs:
            edge_index_list.append([h, t])
            edge_type_list.append(r)

    edge_index = torch.tensor(edge_index_list).t().long()  # shape [2, E]
    edge_type = torch.tensor(edge_type_list).long()

    data = {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_nodes": len(node2id),
        "rel2id": rel2id,
        "node2id": node2id
    }

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print("Saved graph to:", save_path)
    print("Nodes:", data["num_nodes"])
    print("Edges:", edge_index.shape[1])


if __name__ == "__main__":
    build_graph(
        json_path="/home/jaesang/kg_project/output/dialog_triples.json",
        save_path="/home/jaesang/kg_project/output/graph_data.pkl"
    )

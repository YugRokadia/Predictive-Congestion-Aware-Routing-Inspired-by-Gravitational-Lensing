"""
Exports the trained GNN forecaster's actual weights + the raw congestion
history, so the browser can run REAL live inference (the actual forward
pass, using your trained weights) at every clock tick - instead of
replaying pre-computed predictions like the previous version did.

Run this once, then upload model_live.json.
"""
import json
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

from congestion_field import build_graph, node_density_from_link_load

TIMESERIES_PATH = "geant2_timeseries_v3.json"
MODEL_PATH = "gnn_forecaster.pt"
OUT_PATH = "model_live.json"
WINDOW = 3
HIDDEN_DIM = 32


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A_norm):
        return self.linear(A_norm @ X)


class GNNForecaster(nn.Module):
    def __init__(self, window=WINDOW, hidden=HIDDEN_DIM):
        super().__init__()
        self.gcn1 = GCNLayer(window, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.act = nn.ReLU()

    def forward(self, X, A_norm):
        h = self.act(self.gcn1(X, A_norm))
        h = self.act(self.gcn2(h, A_norm))
        delta = self.out(h).squeeze(-1)
        return X[:, -1] + delta


def main():
    with open(TIMESERIES_PATH) as f:
        data = json.load(f)
    data.sort(key=lambda d: d["t"])
    G = build_graph(data[0])
    n = G.number_of_nodes()

    rho = np.zeros((len(data), n))
    for i, d in enumerate(data):
        rho[i] = node_density_from_link_load(G, d["link_load"])

    rho_mean, rho_std = float(rho.mean()), float(rho.std())

    model = GNNForecaster()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    sd = model.state_dict()
    weights = {
        "gcn1_w": sd["gcn1.linear.weight"].tolist(),
        "gcn1_b": sd["gcn1.linear.bias"].tolist(),
        "gcn2_w": sd["gcn2.linear.weight"].tolist(),
        "gcn2_b": sd["gcn2.linear.bias"].tolist(),
        "out_w": sd["out.weight"].tolist(),
        "out_b": sd["out.bias"].tolist(),
    }

    output = {
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "rho_history": rho.tolist(),  # every real t, so the browser can build any window on demand
        "weights": weights,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f)
    print(f"Saved real model weights + {rho.shape[0]}-timestep rho history to {OUT_PATH}")


if __name__ == "__main__":
    main()
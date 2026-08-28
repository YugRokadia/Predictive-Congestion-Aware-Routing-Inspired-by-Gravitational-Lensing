"""
Exports everything needed for an interactive live-routing visualization:
- GEANT2 topology with fixed layout coordinates
- Reactive Phi(t) and anticipatory (GNN-predicted) Phi_hat(t) per node,
  for every held-out timestep (cycle 4)
- The curved-cost routing path chosen under EACH strategy, for EVERY
  src-dst pair, at every held-out timestep

Run this once, then upload the resulting viz_data.json.
"""
import json
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

from congestion_field import build_graph, node_density_from_link_load, solve_potential, normalize_field
from routing import curved_cost_path

TIMESERIES_PATH = "geant2_timeseries_v3.json"
MODEL_PATH = "gnn_forecaster.pt"
OUT_PATH = "viz_data.json"
STEPS_PER_CYCLE = 60
WINDOW = 3
HIDDEN_DIM = 32
ALPHA = 8.0


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


def normalized_adjacency(G: nx.Graph) -> torch.Tensor:
    nodes = sorted(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(float)
    A_hat = A + np.eye(len(nodes))
    D_hat = np.diag(A_hat.sum(axis=1) ** -0.5)
    A_norm = D_hat @ A_hat @ D_hat
    return torch.tensor(A_norm, dtype=torch.float32)


def build_rho_matrix(path: str):
    with open(path) as f:
        data = json.load(f)
    data.sort(key=lambda d: d["t"])
    G = build_graph(data[0])
    n = G.number_of_nodes()
    rho = np.zeros((len(data), n))
    link_loads = {}
    for i, d in enumerate(data):
        rho[i] = node_density_from_link_load(G, d["link_load"])
        link_loads[d["t"]] = d["link_load"]
    return G, rho, link_loads


def main():
    G, rho, link_loads = build_rho_matrix(TIMESERIES_PATH)
    n_nodes = G.number_of_nodes()
    nodes = sorted(G.nodes())
    print(f"loaded {rho.shape[0]} timesteps, {n_nodes} nodes")

    rho_mean, rho_std = rho.mean(), rho.std()
    rho_norm = (rho - rho_mean) / rho_std
    A_norm = normalized_adjacency(G)

    model = GNNForecaster()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # fixed layout, computed once, embedded so the HTML doesn't need networkx
    pos = nx.spring_layout(G, seed=42)
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    layout = {
        str(n): [
            (pos[n][0] - x_min) / (x_max - x_min),
            (pos[n][1] - y_min) / (y_max - y_min),
        ]
        for n in nodes
    }

    edges = [[int(u), int(v)] for u, v in G.edges()]

    train_cutoff = 4 * STEPS_PER_CYCLE
    test_ts = [t for t in range(WINDOW - 1, rho.shape[0] - 1) if t >= train_cutoff]
    print(f"exporting {len(test_ts)} held-out timesteps x {n_nodes*(n_nodes-1)} pairs")

    frames = []
    for t in test_ts:
        Phi_reactive = normalize_field(solve_potential(G, rho[t]))

        window = rho_norm[t - WINDOW + 1: t + 1].T
        X = torch.tensor(window, dtype=torch.float32)
        with torch.no_grad():
            rho_hat_norm = model(X, A_norm).numpy()
        rho_hat = (rho_hat_norm * rho_std) + rho_mean
        Phi_anticipatory = normalize_field(solve_potential(G, rho_hat))

        real_link_load_next = link_loads[t + 1]

        routes_reactive = {}
        routes_anticipatory = {}
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                key = f"{src}-{dst}"
                routes_reactive[key] = curved_cost_path(G, src, dst, Phi_reactive, alpha=ALPHA)
                routes_anticipatory[key] = curved_cost_path(G, src, dst, Phi_anticipatory, alpha=ALPHA)

        frames.append({
            "t": t,
            "phi_reactive": Phi_reactive.round(4).tolist(),
            "phi_anticipatory": Phi_anticipatory.round(4).tolist(),
            "rho_actual_next": rho[t + 1].round(1).tolist(),
            "routes_reactive": routes_reactive,
            "routes_anticipatory": routes_anticipatory,
        })
        if t % 10 == 0:
            print(f"  processed t={t}")

    output = {
        "nodes": [int(n) for n in nodes],
        "edges": edges,
        "layout": layout,
        "frames": frames,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
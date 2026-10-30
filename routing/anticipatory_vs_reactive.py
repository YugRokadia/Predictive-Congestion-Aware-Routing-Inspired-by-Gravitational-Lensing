"""
Anticipatory vs. reactive routing comparison.

Uses the trained GNN forecaster to predict rho(t+1), then routes using
that PREDICTED future field ("anticipatory") vs. routing using only the
CURRENT field at rho(t) ("reactive" - what every earlier routing test in
this project used). Both routes are then scored against the REAL rho(t+1)
that actually happens (from the held-out cycle 4 data) - this tells us
whether forecasting actually helps routing find better paths, or whether
the forecaster's edge (proven earlier: +19-20% MSE improvement over naive
persistence) is too small to change which path gets chosen.

Requires: gnn_forecaster.pt (from train_gnn_forcaster_fin.py) and
geant2_timeseries_v3.json in the same folder.
"""
import json
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

from congestion_field import build_graph, node_density_from_link_load, solve_potential, normalize_field
from routing import curved_cost_path, path_cost

TIMESERIES_PATH = "geant2_timeseries_v3.json"
MODEL_PATH = "gnn_forecaster.pt"
STEPS_PER_CYCLE = 60
WINDOW = 3
HIDDEN_DIM = 32
N_TEST_PAIRS = 30   # random src-dst pairs to test per timestep
ALPHA = 8.0          # same field-cost weight used throughout the project


# ---- must match train_gnn_forcaster_fin.py exactly, or state_dict load fails ----
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
    print(f"loaded {rho.shape[0]} timesteps, {n_nodes} nodes")

    # same normalization as training - must match exactly for correct predictions
    rho_mean, rho_std = rho.mean(), rho.std()
    rho_norm = (rho - rho_mean) / rho_std

    A_norm = normalized_adjacency(G)

    model = GNNForecaster()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # held-out cycle 4 range, same as training's test split
    train_cutoff = 4 * STEPS_PER_CYCLE
    test_ts = [t for t in range(WINDOW - 1, rho.shape[0] - 1) if t >= train_cutoff]
    print(f"testing on {len(test_ts)} held-out timesteps (cycle 4)")

    random.seed(0)
    nodes = list(G.nodes())
    test_pairs = []
    while len(test_pairs) < N_TEST_PAIRS:
        s, d = random.sample(nodes, 2)
        test_pairs.append((s, d))

    reactive_costs, anticipatory_costs = [], []
    reactive_wins, anticipatory_wins, ties = 0, 0, 0

    for t in test_ts:
        # reactive: route using the field AS OF NOW (rho(t))
        Phi_reactive = normalize_field(solve_potential(G, rho[t]))

        # anticipatory: predict rho(t+1), route using the PREDICTED field
        window = rho_norm[t - WINDOW + 1: t + 1].T  # (n_nodes, WINDOW)
        X = torch.tensor(window, dtype=torch.float32)
        with torch.no_grad():
            rho_hat_norm = model(X, A_norm).numpy()
        rho_hat = (rho_hat_norm * rho_std) + rho_mean
        Phi_anticipatory = normalize_field(solve_potential(G, rho_hat))

        # ground truth: the REAL congestion that actually happens at t+1
        real_link_load_next = link_loads[t + 1]

        for src, dst in test_pairs:
            path_reactive = curved_cost_path(G, src, dst, Phi_reactive, alpha=ALPHA)
            path_anticipatory = curved_cost_path(G, src, dst, Phi_anticipatory, alpha=ALPHA)

            cost_reactive = path_cost(path_reactive, real_link_load_next)
            cost_anticipatory = path_cost(path_anticipatory, real_link_load_next)

            reactive_costs.append(cost_reactive)
            anticipatory_costs.append(cost_anticipatory)

            if cost_anticipatory < cost_reactive:
                anticipatory_wins += 1
            elif cost_reactive < cost_anticipatory:
                reactive_wins += 1
            else:
                ties += 1

    reactive_costs = np.array(reactive_costs)
    anticipatory_costs = np.array(anticipatory_costs)

    print(f"\n=== Scored against REAL rho(t+1), {len(reactive_costs)} (timestep, pair) tests ===")
    print(f"Reactive routing:     mean real cost = {reactive_costs.mean():.1f}")
    print(f"Anticipatory routing: mean real cost = {anticipatory_costs.mean():.1f}")
    improvement = (reactive_costs.mean() - anticipatory_costs.mean()) / reactive_costs.mean() * 100
    print(f"Improvement: {improvement:+.1f}%")
    print(f"\nHead-to-head: anticipatory better {anticipatory_wins}x, reactive better {reactive_wins}x, tied {ties}x")

    np.save("reactive_costs.npy", reactive_costs)
    np.save("anticipatory_costs.npy", anticipatory_costs)


if __name__ == "__main__":
    main()
"""
Physics-informed GNN forecaster: predicts rho(t+1) at every GEANT2 node
using real graph convolution (message-passing over the actual adjacency
matrix) plus a physics-residual loss term - same L = L_data + lambda*L_phys
structure as Paper 1's PINTP framework, using the graph Laplacian for
the physics constraint.
 
Requires: pip install torch networkx numpy scikit-learn
 
Trains on cycles 0-3 (t=0..239), tests on held-out cycle 4 (t=240..299).
"""
 
import json
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
 
from congestion_field import build_graph, node_density_from_link_load
 
 
TIMESERIES_PATH = "geant2_timeseries_v4.json"
STEPS_PER_CYCLE = 60
WINDOW = 3
HIDDEN_DIM = 32
EPOCHS = 300
LR = 0.01
LAMBDA_PHYS = 0.01
 
 
def build_rho_matrix(path: str):
    with open(path) as f:
        data = json.load(f)
 
    data.sort(key=lambda d: d["t"])
 
    G = build_graph(data[0])
    n = G.number_of_nodes()
 
    rho = np.zeros((len(data), n))
 
    for i, d in enumerate(data):
        rho[i] = node_density_from_link_load(
            G,
            d["link_load"]
        )
 
    return G, rho
 
 
def normalized_adjacency(G: nx.Graph) -> torch.Tensor:
    """Standard GCN propagation matrix: D^-1/2 (A+I) D^-1/2."""
    nodes = sorted(G.nodes())
 
    A = nx.adjacency_matrix(
        G,
        nodelist=nodes
    ).toarray().astype(float)
 
    A_hat = A + np.eye(len(nodes))
 
    D_hat = np.diag(
        A_hat.sum(axis=1) ** -0.5
    )
 
    A_norm = D_hat @ A_hat @ D_hat
 
    return torch.tensor(
        A_norm,
        dtype=torch.float32
    )
 
 
def graph_laplacian(G: nx.Graph) -> torch.Tensor:
    L = nx.laplacian_matrix(
        G,
        nodelist=sorted(G.nodes())
    ).toarray().astype(float)
 
    return torch.tensor(
        L,
        dtype=torch.float32
    )
 
 
class GCNLayer(nn.Module):
    """
    One graph-convolution layer: propagate features across real edges,
    then a learned linear transform.
    """
 
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
        h = self.act(
            self.gcn1(X, A_norm)
        )
 
        h = self.act(
            self.gcn2(h, A_norm)
        )
 
        delta = self.out(h).squeeze(-1)
 
        # rho(t) + learned correction
        return X[:, -1] + delta
 
 
def build_dataset(rho: np.ndarray):
    """
    Each sample = one timestep t:
    X = (n_nodes, WINDOW) history
    y = (n_nodes,) target
    """
 
    n_t, n_nodes = rho.shape
 
    Xs, ys, ts = [], [], []
 
    for t in range(WINDOW - 1, n_t - 1):
 
        window = rho[
            t - WINDOW + 1 : t + 1
        ].T
 
        Xs.append(window)
        ys.append(rho[t + 1])
        ts.append(t)
 
    return Xs, ys, np.array(ts)
 
 
def main():
 
    G, rho = build_rho_matrix(
        TIMESERIES_PATH
    )
 
    print(
        f"rho matrix: {rho.shape[0]} timesteps "
        f"x {rho.shape[1]} nodes"
    )
 
    # Normalize rho for stable training
    rho_mean = rho.mean()
    rho_std = rho.std()
 
    rho_norm = (
        rho - rho_mean
    ) / rho_std
 
    A_norm = normalized_adjacency(G)
    L = graph_laplacian(G)
 
    Xs, ys, ts = build_dataset(
        rho_norm
    )
 
    train_cutoff = (
        4 * STEPS_PER_CYCLE
    )
 
    train_idx = [
        i
        for i, t in enumerate(ts)
        if t < train_cutoff
    ]
 
    test_idx = [
        i
        for i, t in enumerate(ts)
        if t >= train_cutoff
    ]
 
    print(
        f"train samples: {len(train_idx)}   "
        f"test samples (held-out cycle 4): "
        f"{len(test_idx)}"
    )
 
    X_train = torch.stack([
        torch.tensor(
            Xs[i],
            dtype=torch.float32
        )
        for i in train_idx
    ])
 
    y_train = torch.stack([
        torch.tensor(
            ys[i],
            dtype=torch.float32
        )
        for i in train_idx
    ])
 
    X_test = torch.stack([
        torch.tensor(
            Xs[i],
            dtype=torch.float32
        )
        for i in test_idx
    ])
 
    y_test = torch.stack([
        torch.tensor(
            ys[i],
            dtype=torch.float32
        )
        for i in test_idx
    ])
 
    model = GNNForecaster()
 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )
 
    mse_loss = nn.MSELoss()
 
    # ============================================================
    # TRAINING
    # ============================================================
 
    for epoch in range(EPOCHS):
 
        model.train()
        optimizer.zero_grad()
 
        preds = torch.stack([
            model(
                X_train[i],
                A_norm
            )
            for i in range(len(X_train))
        ])
 
        # Data loss
        data_loss = mse_loss(
            preds,
            y_train
        )
 
        # ========================================================
        # Physics loss:
        #
        # predicted_delta =
        #     rho_pred(t+1) - rho(t)
        #
        # diffusion_term =
        #     -L rho(t)
        #
        # Constraint:
        #
        #     predicted_delta ≈ 0.1 * diffusion_term
        #
        # This constrains the CHANGE in rho rather than forcing
        # the predicted rho field itself to be spatially smooth.
        # ========================================================
 
        phys_terms = []
 
        for i in range(len(preds)):
 
            predicted_delta = (
                preds[i]
                - X_train[i][:, -1]
            )
 
            diffusion_term = -(
                L @ X_train[i][:, -1]
            )
 
            phys_terms.append(
                (
                    predicted_delta
                    - 0.1 * diffusion_term
                ).pow(2).mean()
            )
 
        phys_loss = torch.stack(
            phys_terms
        ).mean()
 
        # Combined loss
        loss = (
            data_loss
            + LAMBDA_PHYS * phys_loss
        )
 
        loss.backward()
        optimizer.step()
 
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
 
            print(
                f"epoch {epoch:3d}  "
                f"data_loss={data_loss.item():.4f}  "
                f"phys_loss={phys_loss.item():.4f}  "
                f"total={loss.item():.4f}"
            )
 
    # ============================================================
    # TESTING
    # ============================================================
 
    model.eval()
 
    with torch.no_grad():
 
        test_preds = torch.stack([
            model(
                X_test[i],
                A_norm
            )
            for i in range(len(X_test))
        ])
 
    # Convert predictions back to original rho scale
    y_test_real = (
        y_test.numpy() * rho_std
    ) + rho_mean
 
    test_preds_real = (
        test_preds.numpy() * rho_std
    ) + rho_mean
 
    # GNN metrics
    mse = mean_squared_error(
        y_test_real.flatten(),
        test_preds_real.flatten()
    )
 
    r2 = r2_score(
        y_test_real.flatten(),
        test_preds_real.flatten()
    )
 
    # ============================================================
    # NAIVE PERSISTENCE BASELINE
    # ============================================================
 
    naive_preds_real = (
        X_test[:, :, -1].numpy()
        * rho_std
    ) + rho_mean
 
    naive_mse = mean_squared_error(
        y_test_real.flatten(),
        naive_preds_real.flatten()
    )
 
    naive_r2 = r2_score(
        y_test_real.flatten(),
        naive_preds_real.flatten()
    )
 
    # ============================================================
    # RESULTS
    # ============================================================
 
    print(
        "\n=== Held-out cycle 4 "
        "(genuinely unseen future) ==="
    )
 
    print(
        f"GNN forecaster:     "
        f"MSE={mse:.1f}   R2={r2:.4f}"
    )
 
    print(
        f"Naive persistence:  "
        f"MSE={naive_mse:.1f}   R2={naive_r2:.4f}"
    )
 
    print(
        f"Improvement over naive: "
        f"{(naive_mse - mse) / naive_mse * 100:+.1f}%"
    )
 
    # ============================================================
    # SAVE MODEL AND RESULTS
    # ============================================================
 
    torch.save(
        model.state_dict(),
        "gnn_forecaster.pt"
    )
 
    np.save(
        "gnn_test_predictions.npy",
        test_preds_real
    )
 
    np.save(
        "gnn_test_actual.npy",
        y_test_real
    )
 
    print(
        "\nSaved model to gnn_forecaster.pt"
    )
 
 
if __name__ == "__main__":
    main()
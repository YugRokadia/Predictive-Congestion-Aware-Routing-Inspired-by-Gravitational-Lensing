"""
Live GEANT2 routing simulator - Streamlit UI.

Unlike the HTML/JS version, this calls your ACTUAL trained torch.nn.Module
directly (model.forward(...)) - no reimplementation, no exported weights,
the real PyTorch model runs live every time the field needs recomputing.

Run with:
    pip install streamlit
    streamlit run app.py

Needs, in the same folder:
    gnn_forecaster.pt
    geant2_timeseries_v3.json
    congestion_field.py
    routing.py
"""
import json
import time
import math

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st

from congestion_field import build_graph, node_density_from_link_load, solve_potential, normalize_field
from routing import curved_cost_path, path_cost

TIMESERIES_PATH = "geant2_timeseries_v3.json"
MODEL_PATH = "gnn_forecaster.pt"
WINDOW = 3
HIDDEN_DIM = 32
ALPHA = 8.0
HOP_DURATION = 1.6          # seconds per hop
CLOCK_TICK_SECONDS = 1.3    # how often the network clock advances

st.set_page_config(page_title="GEANT2 Live Routing", layout="wide")


# ---------------- model definition (must match training exactly) ----------------
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


# ---------------- one-time setup, cached across reruns ----------------
@st.cache_resource
def load_everything():
    with open(TIMESERIES_PATH) as f:
        data = json.load(f)
    data.sort(key=lambda d: d["t"])
    G = build_graph(data[0])
    n = G.number_of_nodes()

    rho = np.zeros((len(data), n))
    for i, d in enumerate(data):
        rho[i] = node_density_from_link_load(G, d["link_load"])

    rho_mean, rho_std = float(rho.mean()), float(rho.std())
    A_norm = normalized_adjacency(G)

    model = GNNForecaster()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    pos = nx.spring_layout(G, seed=42)

    return G, rho, rho_mean, rho_std, A_norm, model, pos


G, rho_history, rho_mean, rho_std, A_norm, model, pos = load_everything()
nodes = sorted(G.nodes())
T_MIN, T_MAX = WINDOW - 1, rho_history.shape[0] - 1


def predict_rho_next(t_idx: int) -> np.ndarray:
    """Runs the REAL trained model's forward pass, live."""
    window = rho_history[t_idx - WINDOW + 1: t_idx + 1]  # (WINDOW, n)
    X = (window.T - rho_mean) / rho_std  # (n, WINDOW)
    X_t = torch.tensor(X, dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        rho_hat_norm = model(X_t, A_norm).numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    rho_hat = rho_hat_norm * rho_std + rho_mean
    return rho_hat, elapsed_ms


def field_reactive(t_idx: int) -> np.ndarray:
    return normalize_field(solve_potential(G, rho_history[t_idx]))


def field_anticipatory(t_idx: int):
    rho_hat, elapsed_ms = predict_rho_next(t_idx)
    return normalize_field(solve_potential(G, rho_hat)), elapsed_ms


# ---------------- session state ----------------
ss = st.session_state
if "clock_idx" not in ss:
    ss.clock_idx = T_MIN + 30
    ss.last_clock_tick = time.time()
    ss.packets = []
    ss.packet_counter = 0
    ss.log = []
    ss.src = nodes[0]
    ss.dst = nodes[5]
    ss.auto_send = False
    ss.last_auto_send = 0.0
    ss.last_inference_ms = 0.0


def log_event(text, kind="info"):
    ss.log.insert(0, (kind, text))
    ss.log = ss.log[:14]


def advance_clock_if_due():
    now = time.time()
    if now - ss.last_clock_tick >= CLOCK_TICK_SECONDS:
        ss.clock_idx += 1
        if ss.clock_idx > T_MAX:
            ss.clock_idx = T_MIN
        ss.last_clock_tick = now


def get_phi(strategy: str):
    if strategy == "reactive":
        return field_reactive(ss.clock_idx)
    else:
        phi, ms = field_anticipatory(ss.clock_idx)
        ss.last_inference_ms = ms
        return phi


def plan_path(strategy, src, dst):
    phi = get_phi(strategy)
    try:
        path = curved_cost_path(G, src, dst, phi, alpha=ALPHA)
    except nx.NetworkXNoPath:
        return None
    return path


def send_packet():
    if ss.src == ss.dst:
        return
    for strategy in ("reactive", "anticipatory"):
        path = plan_path(strategy, ss.src, ss.dst)
        if path is None or len(path) < 2:
            continue
        ss.packet_counter += 1
        ss.packets.append({
            "id": ss.packet_counter, "strategy": strategy, "path": path,
            "hop_idx": 0, "hop_start": time.time(),
        })
        log_event(f"#{ss.packet_counter} ({strategy}) departs {ss.src}\u2192{ss.dst}, via {'\u2192'.join(map(str,path))}")


def advance_packets():
    now = time.time()
    still_flying = []
    for p in ss.packets:
        elapsed = now - p["hop_start"]
        if elapsed < HOP_DURATION:
            still_flying.append(p)
            continue
        p["hop_idx"] += 1
        arrived = p["path"][p["hop_idx"]]
        if arrived == p["path"][-1]:
            log_event(f"#{p['id']} ({p['strategy']}) delivered at {arrived}", "delivered")
            continue
        old_remainder = p["path"][p["hop_idx"]:]
        new_path = plan_path(p["strategy"], arrived, p["path"][-1])
        if new_path and new_path != old_remainder:
            log_event(f"#{p['id']} ({p['strategy']}) REROUTED at {arrived}: now via {'\u2192'.join(map(str,new_path))}", "reroute")
            p["path"] = p["path"][:p["hop_idx"]] + new_path
        p["hop_start"] = now
        still_flying.append(p)
    ss.packets = still_flying


# ---------------- UI ----------------
st.title("GEANT2 Live Routing Simulator")
st.caption("The trained GNN's real forward pass runs live, on your actual `gnn_forecaster.pt` - not reimplemented, not precomputed.")

col_graph, col_controls = st.columns([2.2, 1])

with col_controls:
    st.subheader("Send a packet")
    ss.src = st.selectbox("Source", nodes, index=nodes.index(ss.src))
    ss.dst = st.selectbox("Destination", nodes, index=nodes.index(ss.dst))
    if st.button("SEND PACKET", use_container_width=True, disabled=(ss.src == ss.dst)):
        send_packet()
    ss.auto_send = st.checkbox("auto-send every 2.5s", value=ss.auto_send)

    st.subheader("Live model status")
    st.metric("GNN inference time", f"{ss.last_inference_ms:.2f} ms")
    st.metric("Network clock", f"t={ss.clock_idx}")

    st.subheader("Event log")
    log_box = st.container(height=280)
    with log_box:
        if not ss.log:
            st.caption("no packets sent yet")
        for kind, text in ss.log:
            if kind == "reroute":
                st.markdown(f":orange[{text}]")
            elif kind == "delivered":
                st.markdown(f":green[{text}]")
            else:
                st.caption(text)

# ---------------- simulation step ----------------
advance_clock_if_due()
if ss.auto_send and time.time() - ss.last_auto_send > 2.5 and ss.src != ss.dst:
    send_packet()
    ss.last_auto_send = time.time()
advance_packets()

# ---------------- render ----------------
phi_reactive = field_reactive(ss.clock_idx)
phi_anticipatory, _ = field_anticipatory(ss.clock_idx)

with col_graph:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    fig.patch.set_facecolor("#0A0E14")

    for ax, phi, title, strategy, color in [
        (axes[0], phi_reactive, "REACTIVE  (uses rho(t) as-is)", "reactive", "#4DD0E1"),
        (axes[1], phi_anticipatory, "ANTICIPATORY  (live GNN prediction)", "anticipatory", "#C77DFF"),
    ]:
        ax.set_facecolor("#0A0E14")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#263141", width=1.2)
        node_colors = [plt.cm.YlOrRd(phi[n]) for n in nodes]
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, node_color=node_colors, node_size=260)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white")

        relevant = set()
        for p in ss.packets:
            if p["strategy"] != strategy:
                continue
            remaining = p["path"][p["hop_idx"]:]
            relevant.update(remaining)
            xs = [pos[n][0] for n in remaining]
            ys = [pos[n][1] for n in remaining]
            ax.plot(xs, ys, color=color, alpha=0.5, linewidth=2, zorder=1)

            frm, to = p["path"][p["hop_idx"]], p["path"][p["hop_idx"] + 1] if p["hop_idx"] + 1 < len(p["path"]) else None
            if to is not None:
                f = min((time.time() - p["hop_start"]) / HOP_DURATION, 1.0)
                x = pos[frm][0] + (pos[to][0] - pos[frm][0]) * f
                y = pos[frm][1] + (pos[to][1] - pos[frm][1]) * f
                ax.scatter([x], [y], color=color, s=70, zorder=5, edgecolors="white", linewidths=0.8)

        for n in relevant:
            if phi[n] > 0.55:
                ax.scatter([pos[n][0]], [pos[n][1]], s=420, facecolors="none", edgecolors="#47C8FF", linewidths=1.6, zorder=4)

        for n in (ss.src, ss.dst):
            ax.scatter([pos[n][0]], [pos[n][1]], s=340, facecolors="none", edgecolors="white", linewidths=2, zorder=6)

        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# keep the app "live" without requiring manual interaction
time.sleep(0.35)
st.rerun()
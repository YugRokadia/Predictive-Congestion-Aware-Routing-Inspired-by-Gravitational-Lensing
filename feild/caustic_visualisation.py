"""
Caustic visualization: naive independent routing vs. Frank-Wolfe equilibrium,
overlaid on the real GEANT2 topology. Edge thickness/darkness = agent-induced
flow relative to a fixed reference (20,000).
"""
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from congestion_field import build_graph
from equilibrium import generate_requests, naive_independent_routing, frank_wolfe_equilibrium

with open("gnnet_samples.json") as f:
    data = json.load(f)
sample = next(d for d in data if d["topology"] == "GEANT2")
G = build_graph(sample)
background_load = sample["link_load"]

requests = generate_requests(G, n_requests=40, seed=1, demand=5000.0)
naive_paths, naive_flow = naive_independent_routing(G, requests, background_load, alpha=8.0)
eq_flow, history = frank_wolfe_equilibrium(G, requests, background_load, alpha=8.0, beta=4.0, power=4, iterations=30)

pos = nx.spring_layout(G, seed=42)


def edge_widths_colors(G, flow_dict, cap=20000):
    widths, colors = [], []
    for u, v in G.edges():
        key1, key2 = f"{u}-{v}", f"{v}-{u}"
        f = flow_dict.get(key1, flow_dict.get(key2, 0.0))
        widths.append(1 + 6 * (f / cap))
        colors.append(f / cap)
    return widths, colors


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

for ax, flow, title in [
    (axes[0], naive_flow, "Naive independent routing\n(caustic: agents pile onto same edges)"),
    (axes[1], eq_flow, "Frank-Wolfe equilibrium\n(flow spread across alternate paths)"),
]:
    widths, colors = edge_widths_colors(G, flow)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightsteelblue", node_size=400)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=colors,
                            edge_cmap=cm.Reds, edge_vmin=0, edge_vmax=1)
    ax.set_title(title)
    ax.axis("off")

sm = cm.ScalarMappable(cmap=cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.05)
cbar.set_label("agent-induced flow / 20,000 (thicker+darker = more agents piled on that edge)")

plt.savefig("caustic_visualization.png", dpi=130, bbox_inches="tight")
plt.show()
print("saved caustic_visualization.png")
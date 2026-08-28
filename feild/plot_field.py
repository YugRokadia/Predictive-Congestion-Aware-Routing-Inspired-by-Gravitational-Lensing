"""
Plots the real GEANT2 topology, then the same topology colored by the
smoothed congestion potential Phi computed from real link-load data.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from congestion_field import build_graph, node_density_from_link_load, solve_potential, normalize_field

with open("gnnet_samples.json") as f:
    data = json.load(f)

sample = next(d for d in data if d["topology"] == "GEANT2")
G = build_graph(sample)
rho = node_density_from_link_load(G, sample["link_load"])
Phi_norm = normalize_field(solve_potential(G, rho))

print("rho:", rho.round(1))
print("Phi_norm:", Phi_norm.round(3))
print("correlation(rho, Phi):", round(float(np.corrcoef(rho, Phi_norm)[0, 1]), 3))

pos = nx.spring_layout(G, seed=42)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

nx.draw(G, pos, ax=axes[0], with_labels=True, node_color="lightsteelblue",
        node_size=500, font_size=9, edge_color="gray")
axes[0].set_title(f"GEANT2 topology ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) - sanity check")

nodes = nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=Phi_norm,
                                cmap="YlOrRd", node_size=500)
nx.draw_networkx_edges(G, pos, ax=axes[1], edge_color="gray")
nx.draw_networkx_labels(G, pos, ax=axes[1], font_size=9)
plt.colorbar(nodes, ax=axes[1], label="Phi (smoothed congestion potential)")
axes[1].set_title("Congestion field Phi (real GEANT2 sample)")

plt.tight_layout()
plt.savefig("geant2_sanity_check.png", dpi=130)
plt.show()
print("saved geant2_sanity_check.png")

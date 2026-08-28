"""
Synthetic sanity check for the field solver: put a single congestion
'mass' on one node of a simple ring graph, and confirm the resulting
potential field decays smoothly and symmetrically with graph distance -
exactly like a point mass in a gravitational/electrostatic field.
"""
import networkx as nx
import numpy as np
from congestion_field import solve_potential, normalize_field

G = nx.cycle_graph(12)
rho = np.zeros(12)
rho[0] = 1000.0  # single mass at node 0

Phi_norm = normalize_field(solve_potential(G, rho))
dist = nx.shortest_path_length(G, source=0)

print("node : hops_from_mass : Phi_norm")
for node in sorted(G.nodes()):
    print(f"{node:4d} : {dist[node]:11d} : {Phi_norm[node]:.3f}")

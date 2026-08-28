"""
Curved-cost routing.

Three routing strategies compared on the same graph:

1. naive_shortest_path   - shortest path by hop count, ignores congestion
                            entirely. The "flat space" baseline.

2. reactive_shortest_path - shortest path weighted by *current* link load
                            directly (no smoothing). This is the
                            Waze/Maps-equivalent baseline: reacts to
                            congestion, but only locally, at the edge level.

3. curved_cost_path      - shortest path weighted by the smoothed potential
                            field Phi (from congestion_field.py). Edges near
                            high-congestion regions cost more even if the
                            edge itself isn't congested yet - paths bend
                            away from congestion before entering it.
"""

import networkx as nx
import numpy as np


def naive_shortest_path(G: nx.Graph, src: int, dst: int) -> list:
    return nx.shortest_path(G, src, dst)  # unweighted = hop count


def reactive_shortest_path(G: nx.Graph, src: int, dst: int, link_load: dict) -> list:
    Gw = G.copy()
    for u, v in Gw.edges():
        key1, key2 = f"{u}-{v}", f"{v}-{u}"
        load = link_load.get(key1, link_load.get(key2, 0.0))
        Gw[u][v]["weight"] = 1.0 + load  # local congestion penalty only
    return nx.shortest_path(Gw, src, dst, weight="weight")


def curved_cost_path(G: nx.Graph, src: int, dst: int, Phi_norm: np.ndarray,
                      alpha: float = 5.0) -> list:
    """
    Reweight each edge using the *average* smoothed potential of its two
    endpoints - this is what makes routes bend gradually, since an edge
    near (but not inside) a congested region still costs more than one
    far from it.
    """
    Gw = G.copy()
    for u, v in Gw.edges():
        field_cost = alpha * 0.5 * (Phi_norm[u] + Phi_norm[v])
        Gw[u][v]["weight"] = 1.0 + field_cost
    return nx.shortest_path(Gw, src, dst, weight="weight")


def path_cost(path: list, link_load: dict) -> float:
    """Score any path by its *real* total link load - the ground-truth-ish
    proxy we use to judge whether a path actually avoided congestion."""
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        key1, key2 = f"{u}-{v}", f"{v}-{u}"
        total += link_load.get(key1, link_load.get(key2, 0.0))
    return total


def top_k_paths(G: nx.Graph, src: int, dst: int, Phi_norm: np.ndarray,
                 k: int = 3, alpha: float = 5.0) -> list:
    """Return up to k distinct low-cost paths on the curved-cost graph
    ('multiple images' of the same source-destination pair)."""
    Gw = G.copy()
    for u, v in Gw.edges():
        field_cost = alpha * 0.5 * (Phi_norm[u] + Phi_norm[v])
        Gw[u][v]["weight"] = 1.0 + field_cost
    gen = nx.shortest_simple_paths(Gw, src, dst, weight="weight")
    paths = []
    for i, p in enumerate(gen):
        paths.append(p)
        if i + 1 >= k:
            break
    return paths

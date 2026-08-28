'''
"""
Congestion field solver.

Physical analogy:
    mass density rho(x)              -> congestion density at each node
    Poisson eq. grad^2 Phi = k*rho   -> smoothed potential field over the graph
    (we solve the discrete graph version: L @ Phi = k*rho)

We derive per-node congestion density from per-link load (a node's density
is the sum of load on all links touching it), then smooth that into a
potential field using the graph Laplacian. The Laplacian is singular
(constant vectors are in its null space, same as with the continuous
Poisson equation), so we solve it with the Moore-Penrose pseudoinverse,
which picks the minimum-norm solution - the discrete analogue of pinning
"potential at infinity" to zero.
"""

import numpy as np
import networkx as nx


def build_graph(sample: dict) -> nx.Graph:
    """Build a networkx graph from one parsed gnnet sample."""
    G = nx.Graph()
    G.add_nodes_from(range(sample["nodes"]))
    for e in sample["edges"]:
        u, v = e["src"], e["dst"]
        if not G.has_edge(u, v):
            G.add_edge(u, v, bandwidth=e["bandwidth"])
    return G


def node_density_from_link_load(G: nx.Graph, link_load: dict) -> np.ndarray:
    """
    Convert per-link load (keyed 'u-v') into a per-node congestion density,
    by summing the load on all links incident to each node.
    """
    n = G.number_of_nodes()
    rho = np.zeros(n)
    for key, load in link_load.items():
        u, v = (int(x) for x in key.split("-"))
        if u < n:
            rho[u] += load
        if v < n:
            rho[v] += load
    return rho


def solve_potential(G: nx.Graph, rho: np.ndarray, k: float = 1.0) -> np.ndarray:
    """
    Solve the discrete Poisson equation L @ Phi = k * rho for the smooth
    potential field Phi, using the combinatorial graph Laplacian L = D - A.

    L is singular (rank n-1), so we use the pseudoinverse -> minimum-norm
    solution, which is the standard way to solve Poisson's equation on a
    graph when there's no fixed boundary condition.
    """
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(float)
    L_pinv = np.linalg.pinv(L)
    Phi = L_pinv @ (k * rho)
    return Phi


def normalize_field(Phi: np.ndarray) -> np.ndarray:
    """Shift/scale Phi to [0, 1] purely for visualization / edge-cost weighting."""
    lo, hi = Phi.min(), Phi.max()
    if hi - lo < 1e-12:
        return np.zeros_like(Phi)
    return (Phi - lo) / (hi - lo)
'''

"""
Congestion field solver.

Physical analogy:
    mass density rho(x)              -> congestion density at each node
    Poisson eq. grad^2 Phi = k*rho   -> smoothed potential field over the graph
    (we solve the discrete graph version: L @ Phi = k*rho)

We derive per-node congestion density from per-link load (a node's density
is the sum of load on all links touching it), then smooth that into a
potential field using the graph Laplacian. The Laplacian is singular
(constant vectors are in its null space, same as with the continuous
Poisson equation), so we solve it with the Moore-Penrose pseudoinverse,
which picks the minimum-norm solution - the discrete analogue of pinning
"potential at infinity" to zero.
"""

import numpy as np
import networkx as nx


def build_graph(sample: dict) -> nx.Graph:
    """Build a networkx graph from one parsed gnnet sample."""
    G = nx.Graph()
    G.add_nodes_from(range(sample["nodes"]))
    for e in sample["edges"]:
        u, v = e["src"], e["dst"]
        if not G.has_edge(u, v):
            G.add_edge(u, v, bandwidth=e["bandwidth"])
    return G


def node_density_from_link_load(G: nx.Graph, link_load: dict) -> np.ndarray:
    """
    Convert per-link load (keyed 'u-v') into a per-node congestion density,
    by summing the load on all links incident to each node.
    """
    n = G.number_of_nodes()
    rho = np.zeros(n)
    for key, load in link_load.items():
        u, v = (int(x) for x in key.split("-"))
        if u < n:
            rho[u] += load
        if v < n:
            rho[v] += load
    return rho


def solve_potential(G: nx.Graph, rho: np.ndarray, k: float = 1.0) -> np.ndarray:
    """
    Solve the discrete Poisson equation L @ Phi = k * rho for the smooth
    potential field Phi, using the combinatorial graph Laplacian L = D - A.

    L is singular (rank n-1), so we use the pseudoinverse -> minimum-norm
    solution, which is the standard way to solve Poisson's equation on a
    graph when there's no fixed boundary condition.
    """
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(float)
    L_pinv = np.linalg.pinv(L)
    Phi = L_pinv @ (k * rho)
    return Phi


def normalize_field(Phi: np.ndarray) -> np.ndarray:
    """Shift/scale Phi to [0, 1] purely for visualization / edge-cost weighting."""
    lo, hi = Phi.min(), Phi.max()
    if hi - lo < 1e-12:
        return np.zeros_like(Phi)
    return (Phi - lo) / (hi - lo)


def edge_base_time(G: nx.Graph, gamma: float = 1.0) -> dict:
    """
    Real per-edge base cost, derived from real bandwidth capacity as an
    inverse-capacity latency proxy - this is the base_time(e) term in the
    original spec cost(e,t) = base_time(e) x (1 + f(Phi)), which was
    previously hardcoded to 1.0 for every edge (see project handoff,
    Section 9, item 1).

    base_time(e) = (max_bandwidth_in_graph / bandwidth(e)) ** gamma

    The highest-capacity edge in the graph gets base_time = 1.0 (fastest
    possible traversal); lower-capacity edges cost proportionally more to
    traverse, independent of congestion. gamma < 1.0 compresses the range
    so base_time doesn't swamp the congestion-field term when the two are
    multiplied together - gamma=1.0 is the literal inverse-bandwidth ratio
    (e.g. 10x for a 10Gbps edge vs a 100Gbps edge in this dataset's real
    capacity tiers); gamma=0.5 softens that to ~3.2x. Tune per use case.

    Returns a dict keyed both (u,v) and (v,u) so lookups work regardless
    of edge traversal direction, matching the convention used elsewhere
    in this codebase (e.g. link_load keys).
    """
    max_bw = max(d.get("bandwidth", 1.0) for _, _, d in G.edges(data=True))
    base_time = {}
    for u, v, d in G.edges(data=True):
        bw = d.get("bandwidth", max_bw)
        bt = (max_bw / bw) ** gamma
        base_time[(u, v)] = bt
        base_time[(v, u)] = bt
    return base_time
'''
"""
Equilibrium layer.

The problem: if every agent independently picks the "best" curved-cost path
against the SAME static field, they all pile onto the same low-cost route -
the caustic failure mode from the lensing analogy. This module shows that
failure, then fixes it with Frank-Wolfe traffic assignment (a standard way
to reach a Wardrop equilibrium): instead of moving 100% of every agent's
flow onto the current best path each round, we blend a small fraction of
flow toward the current best path, and recompute the cost (which now
reflects other agents' flow) each round. Over iterations this spreads flow
across multiple paths rather than concentrating it on one.
"""

from collections import defaultdict
import numpy as np
import networkx as nx

from congestion_field import node_density_from_link_load, solve_potential, normalize_field
from routing import curved_cost_path


def generate_requests(G: nx.Graph, n_requests: int, seed: int = 0, demand: float = 1000.0) -> list:
    """Generate n_requests random (src, dst, demand) triples. demand is in the
    same units as link_load (e.g. bandwidth) so it's comparable to real congestion."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    requests = []
    while len(requests) < n_requests:
        src, dst = rng.choice(nodes, size=2, replace=False)
        requests.append((int(src), int(dst), demand))
    return requests


def _edge_key(u: int, v: int) -> str:
    return f"{u}-{v}"


def paths_to_edge_flow(paths_with_weight: list) -> dict:
    """Sum weighted flow onto each edge across a list of (path, weight) pairs."""
    edge_flow = defaultdict(float)
    for path, w in paths_with_weight:
        for i in range(len(path) - 1):
            edge_flow[_edge_key(path[i], path[i + 1])] += w
    return dict(edge_flow)


def combined_edge_load(background_load: dict, agent_edge_flow: dict) -> dict:
    """background congestion (from real GEANT2 traffic) + agent-induced flow."""
    combined = dict(background_load)
    for k, v in agent_edge_flow.items():
        combined[k] = combined.get(k, 0.0) + v
    return combined


def field_from_edge_load(G: nx.Graph, edge_load: dict) -> np.ndarray:
    rho = node_density_from_link_load(G, edge_load)
    return normalize_field(solve_potential(G, rho))


def _edge_capacity(G: nx.Graph) -> dict:
    """Real bandwidth capacity per edge, from the topology data."""
    cap = {}
    for u, v, d in G.edges(data=True):
        bw = d.get("bandwidth", 1.0)
        cap[_edge_key(u, v)] = bw
        cap[_edge_key(v, u)] = bw
    return cap


def _bpr_weighted_graph(G: nx.Graph, Phi_norm: np.ndarray, agent_flow: dict,
                         capacity: dict, alpha: float = 8.0,
                         beta: float = 4.0, power: int = 4) -> nx.Graph:
    """
    Combine the background congestion field (Phi, from real GEANT2 traffic)
    with a BPR-style marginal-cost term driven by agent flow relative to
    each edge's REAL capacity: cost grows sharply as agent flow approaches
    capacity, the standard way traffic-assignment literature (Wardrop /
    Frank-Wolfe) makes a solver actually sensitive to crowding, instead of
    that signal being drowned out by (much larger, static) background load.
    """
    Gw = G.copy()
    for u, v in Gw.edges():
        field_cost = alpha * 0.5 * (Phi_norm[u] + Phi_norm[v])
        key = _edge_key(u, v)
        flow = agent_flow.get(key, agent_flow.get(_edge_key(v, u), 0.0))
        cap = capacity.get(key, 1.0)
        bpr_cost = beta * (flow / cap) ** power
        Gw[u][v]["weight"] = 1.0 + field_cost + bpr_cost
    return Gw


def naive_independent_routing(G: nx.Graph, requests: list, background_load: dict,
                               alpha: float = 8.0) -> tuple:
    """
    Every agent independently routes against the SAME static field
    (computed once, from background congestion only - agents don't see
    each other, and don't see each other's flow at all). This reproduces
    the caustic/pile-up failure mode: if many agents share a low-cost
    corridor, they all pile onto it with no penalty for doing so.
    """
    Phi_norm = field_from_edge_load(G, background_load)
    paths = [curved_cost_path(G, src, dst, Phi_norm, alpha=alpha) for src, dst, _ in requests]
    weighted_paths = [(p, requests[i][2]) for i, p in enumerate(paths)]
    agent_edge_flow = paths_to_edge_flow(weighted_paths)
    return paths, agent_edge_flow


def frank_wolfe_equilibrium(G: nx.Graph, requests: list, background_load: dict,
                             alpha: float = 8.0, beta: float = 4.0, power: int = 4,
                             iterations: int = 30) -> tuple:
    """
    Wardrop-style equilibrium via Frank-Wolfe traffic assignment.

    The background field Phi (real congestion) is computed once and held
    fixed - it represents the network's actual traffic pattern, which
    doesn't change based on how we route a handful of new requests. On top
    of it, a BPR-style term penalizes agent flow relative to each edge's
    real capacity, so the solver is actually sensitive to agents piling up.

    Each iteration:
      1. find the all-or-nothing "best path now" assignment under the
         CURRENT combined cost (field + BPR agent-crowding penalty)
      2. blend only a shrinking fraction gamma_t = 2/(t+2) of flow toward it

    gamma shrinking over iterations is what makes this converge to a spread
    equilibrium instead of oscillating between two "best" paths forever.
    """
    demands = [w for _, _, w in requests]
    capacity = _edge_capacity(G)
    Phi_bg = field_from_edge_load(G, background_load)

    # start: naive AON assignment (no agent-crowding penalty yet)
    current_paths = [curved_cost_path(G, s, d, Phi_bg, alpha=alpha) for s, d, _ in requests]
    current_flow = paths_to_edge_flow(list(zip(current_paths, demands)))

    history = [dict(current_flow)]

    for t in range(1, iterations + 1):
        Gw = _bpr_weighted_graph(G, Phi_bg, current_flow, capacity, alpha, beta, power)
        aux_paths = [nx.shortest_path(Gw, s, d, weight="weight") for s, d, _ in requests]
        aux_flow = paths_to_edge_flow(list(zip(aux_paths, demands)))

        gamma = 2.0 / (t + 2)
        all_keys = set(current_flow) | set(aux_flow)
        new_flow = {}
        for k in all_keys:
            cf = current_flow.get(k, 0.0)
            af = aux_flow.get(k, 0.0)
            new_flow[k] = (1 - gamma) * cf + gamma * af
        current_flow = new_flow
        history.append(dict(current_flow))

    return current_flow, history


def max_overload(background_load: dict, agent_edge_flow: dict) -> tuple:
    """Report the single most-overloaded edge: combined load vs. its
    background load alone - this is the caustic metric."""
    combined = combined_edge_load(background_load, agent_edge_flow)
    worst_key, worst_val = max(combined.items(), key=lambda kv: kv[1])
    bg = background_load.get(worst_key, 0.0)
    return worst_key, worst_val, bg
'''
"""
Equilibrium layer.

The problem: if every agent independently picks the "best" curved-cost path
against the SAME static field, they all pile onto the same low-cost route -
the caustic failure mode from the lensing analogy. This module shows that
failure, then fixes it with Frank-Wolfe traffic assignment (a standard way
to reach a Wardrop equilibrium): instead of moving 100% of every agent's
flow onto the current best path each round, we blend a small fraction of
flow toward the current best path, and recompute the cost (which now
reflects other agents' flow) each round. Over iterations this spreads flow
across multiple paths rather than concentrating it on one.
"""

from collections import defaultdict
import numpy as np
import networkx as nx

from congestion_field import node_density_from_link_load, solve_potential, normalize_field
from routing import curved_cost_path


def generate_requests(G: nx.Graph, n_requests: int, seed: int = 0, demand: float = 1000.0) -> list:
    """Generate n_requests random (src, dst, demand) triples. demand is in the
    same units as link_load (e.g. bandwidth) so it's comparable to real congestion."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    requests = []
    while len(requests) < n_requests:
        src, dst = rng.choice(nodes, size=2, replace=False)
        requests.append((int(src), int(dst), demand))
    return requests


def _edge_key(u: int, v: int) -> str:
    return f"{u}-{v}"


def paths_to_edge_flow(paths_with_weight: list) -> dict:
    """Sum weighted flow onto each edge across a list of (path, weight) pairs."""
    edge_flow = defaultdict(float)
    for path, w in paths_with_weight:
        for i in range(len(path) - 1):
            edge_flow[_edge_key(path[i], path[i + 1])] += w
    return dict(edge_flow)


def combined_edge_load(background_load: dict, agent_edge_flow: dict) -> dict:
    """background congestion (from real GEANT2 traffic) + agent-induced flow."""
    combined = dict(background_load)
    for k, v in agent_edge_flow.items():
        combined[k] = combined.get(k, 0.0) + v
    return combined


def field_from_edge_load(G: nx.Graph, edge_load: dict) -> np.ndarray:
    rho = node_density_from_link_load(G, edge_load)
    return normalize_field(solve_potential(G, rho))


def _edge_capacity(G: nx.Graph) -> dict:
    """Real bandwidth capacity per edge, from the topology data."""
    cap = {}
    for u, v, d in G.edges(data=True):
        bw = d.get("bandwidth", 1.0)
        cap[_edge_key(u, v)] = bw
        cap[_edge_key(v, u)] = bw
    return cap


def _bpr_weighted_graph(G: nx.Graph, Phi_norm: np.ndarray, agent_flow: dict,
                         capacity: dict, alpha: float = 8.0,
                         beta: float = 4.0, power: int = 4,
                         base_time: dict = None) -> nx.Graph:
    """
    Combine a real per-edge base cost (base_time, e.g. bandwidth-derived -
    see congestion_field.edge_base_time) with the background congestion
    field (Phi, from real GEANT2 traffic) and a BPR-style marginal-cost
    term driven by agent flow relative to each edge's REAL capacity: cost
    grows sharply as agent flow approaches capacity, the standard way
    traffic-assignment literature (Wardrop / Frank-Wolfe) makes a solver
    actually sensitive to crowding, instead of that signal being drowned
    out by (much larger, static) background load.

    cost(e) = base_time(e) x (1 + field_cost) + bpr_cost

    The BPR crowding term is left additive (not multiplied by base_time):
    it already scales against the edge's own real capacity, so folding in
    base_time again would double-count the same bandwidth signal.
    base_time defaults to 1.0 per edge if not provided (pre-fix behavior).
    """
    Gw = G.copy()
    for u, v in Gw.edges():
        field_cost = alpha * 0.5 * (Phi_norm[u] + Phi_norm[v])
        key = _edge_key(u, v)
        flow = agent_flow.get(key, agent_flow.get(_edge_key(v, u), 0.0))
        cap = capacity.get(key, 1.0)
        bpr_cost = beta * (flow / cap) ** power
        bt = 1.0 if base_time is None else base_time.get((u, v), 1.0)
        Gw[u][v]["weight"] = bt * (1.0 + field_cost) + bpr_cost
    return Gw


def naive_independent_routing(G: nx.Graph, requests: list, background_load: dict,
                               alpha: float = 8.0) -> tuple:
    """
    Every agent independently routes against the SAME static field
    (computed once, from background congestion only - agents don't see
    each other, and don't see each other's flow at all). This reproduces
    the caustic/pile-up failure mode: if many agents share a low-cost
    corridor, they all pile onto it with no penalty for doing so.
    """
    Phi_norm = field_from_edge_load(G, background_load)
    paths = [curved_cost_path(G, src, dst, Phi_norm, alpha=alpha) for src, dst, _ in requests]
    weighted_paths = [(p, requests[i][2]) for i, p in enumerate(paths)]
    agent_edge_flow = paths_to_edge_flow(weighted_paths)
    return paths, agent_edge_flow


def frank_wolfe_equilibrium(G: nx.Graph, requests: list, background_load: dict,
                             alpha: float = 8.0, beta: float = 4.0, power: int = 4,
                             iterations: int = 30) -> tuple:
    """
    Wardrop-style equilibrium via Frank-Wolfe traffic assignment.

    The background field Phi (real congestion) is computed once and held
    fixed - it represents the network's actual traffic pattern, which
    doesn't change based on how we route a handful of new requests. On top
    of it, a BPR-style term penalizes agent flow relative to each edge's
    real capacity, so the solver is actually sensitive to agents piling up.

    Each iteration:
      1. find the all-or-nothing "best path now" assignment under the
         CURRENT combined cost (field + BPR agent-crowding penalty)
      2. blend only a shrinking fraction gamma_t = 2/(t+2) of flow toward it

    gamma shrinking over iterations is what makes this converge to a spread
    equilibrium instead of oscillating between two "best" paths forever.
    """
    demands = [w for _, _, w in requests]
    capacity = _edge_capacity(G)
    Phi_bg = field_from_edge_load(G, background_load)

    # start: naive AON assignment (no agent-crowding penalty yet)
    current_paths = [curved_cost_path(G, s, d, Phi_bg, alpha=alpha) for s, d, _ in requests]
    current_flow = paths_to_edge_flow(list(zip(current_paths, demands)))

    history = [dict(current_flow)]

    for t in range(1, iterations + 1):
        Gw = _bpr_weighted_graph(G, Phi_bg, current_flow, capacity, alpha, beta, power)
        aux_paths = [nx.shortest_path(Gw, s, d, weight="weight") for s, d, _ in requests]
        aux_flow = paths_to_edge_flow(list(zip(aux_paths, demands)))

        gamma = 2.0 / (t + 2)
        all_keys = set(current_flow) | set(aux_flow)
        new_flow = {}
        for k in all_keys:
            cf = current_flow.get(k, 0.0)
            af = aux_flow.get(k, 0.0)
            new_flow[k] = (1 - gamma) * cf + gamma * af
        current_flow = new_flow
        history.append(dict(current_flow))

    return current_flow, history


def max_overload(background_load: dict, agent_edge_flow: dict) -> tuple:
    """Report the single most-overloaded edge: combined load vs. its
    background load alone - this is the caustic metric."""
    combined = combined_edge_load(background_load, agent_edge_flow)
    worst_key, worst_val = max(combined.items(), key=lambda kv: kv[1])
    bg = background_load.get(worst_key, 0.0)
    return worst_key, worst_val, bg
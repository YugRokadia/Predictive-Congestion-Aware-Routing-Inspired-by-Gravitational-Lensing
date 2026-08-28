"""
Equilibrium layer demo: shows the caustic failure mode (naive independent
routing) vs. the Frank-Wolfe/Wardrop fix, on real GEANT2 congestion data.
"""
import json
from congestion_field import build_graph
from equilibrium import (
    generate_requests,
    naive_independent_routing,
    frank_wolfe_equilibrium,
    field_from_edge_load,
    _edge_capacity,
    _edge_key,
)

with open("gnnet_samples.json") as f:
    data = json.load(f)

sample = next(d for d in data if d["topology"] == "GEANT2")
G = build_graph(sample)
background_load = sample["link_load"]
capacity = _edge_capacity(G)

# 40 simultaneous routing requests, each carrying 5000 units of demand
requests = generate_requests(G, n_requests=40, seed=1, demand=5000.0)

# --- naive: everyone independently routes against the same static field ---
naive_paths, naive_flow = naive_independent_routing(G, requests, background_load, alpha=8.0)

print("=== Top 5 edges by AGENT-INDUCED flow (naive) ===")
for k, v in sorted(naive_flow.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  edge {k}: agent_flow={v:.0f}")

# --- equilibrium: Frank-Wolfe traffic assignment ---
eq_flow, history = frank_wolfe_equilibrium(
    G, requests, background_load, alpha=8.0, beta=4.0, power=4, iterations=30
)

print("\n=== Top 5 edges by AGENT-INDUCED flow (Frank-Wolfe equilibrium) ===")
for k, v in sorted(eq_flow.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  edge {k}: agent_flow={v:.0f}")

max_naive = max(naive_flow.values())
max_eq = max(eq_flow.values())
print(f"\nmax single-edge agent load -- naive: {max_naive:.0f}   equilibrium: {max_eq:.0f}")
print(f"peak-edge reduction: {(max_naive - max_eq) / max_naive * 100:.1f}%")


# --- the metric that actually matters: total system cost ---
def total_system_cost(edge_flow, alpha=8.0, beta=4.0, power=4):
    Phi_bg = field_from_edge_load(G, background_load)
    total = 0.0
    for u, v in G.edges():
        key = _edge_key(u, v)
        flow = edge_flow.get(key, edge_flow.get(_edge_key(v, u), 0.0))
        cap = capacity.get(key, 1.0)
        field_cost = alpha * 0.5 * (Phi_bg[u] + Phi_bg[v])
        bpr_cost = beta * (flow / cap) ** power
        cost_per_unit = 1.0 + field_cost + bpr_cost
        total += flow * cost_per_unit
    return total


naive_cost = total_system_cost(naive_flow)
eq_cost = total_system_cost(eq_flow)
print(f"\ntotal system cost -- naive: {naive_cost:.0f}   equilibrium: {eq_cost:.0f}")
print(f"system-wide reduction: {(naive_cost - eq_cost) / naive_cost * 100:.1f}%")
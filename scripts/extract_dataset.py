import datanetAPI
import json

DATA_PATH = r"C:\Users\Yug Rokadia\Documents\Capstone\dataset\gnnet_data_set_training"
OUT_PATH = "gnnet_samples.json"
MAX_PER_TOPOLOGY = 100

reader = datanetAPI.DatanetAPI(DATA_PATH, [], shuffle=False)
it = iter(reader)

results = []
counts = {14: 0, 24: 0}

for sample in it:
    n = sample.get_network_size()
    if n not in counts or counts[n] >= MAX_PER_TOPOLOGY:
        if all(c >= MAX_PER_TOPOLOGY for c in counts.values()):
            break
        continue

    G = sample.get_topology_object()
    routing = sample.get_routing_matrix()
    traffic = sample.get_traffic_matrix()
    perf = sample.get_performance_matrix()

    edges = [{"src": u, "dst": v, "bandwidth": G[u][v][0]["bandwidth"]} for u, v in G.edges()]

    link_load = {}
    od_delay = []

    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            path = routing[src, dst]
            bw = traffic[src, dst]["AggInfo"]["AvgBw"]
            for i in range(len(path) - 1):
                key = f"{path[i]}-{path[i+1]}"
                link_load[key] = link_load.get(key, 0) + bw

            agg = perf[src, dst]["AggInfo"]
            od_delay.append({
                "src": src, "dst": dst,
                "avg_delay": agg["AvgDelay"],
                "jitter": agg["Jitter"],
                "pkts_drop": agg["PktsDrop"],
            })

    results.append({
        "topology": "NSFNET" if n == 14 else "GEANT2",
        "nodes": n,
        "edges": edges,
        "link_load": link_load,
        "od_delay": od_delay,
    })
    counts[n] += 1

print(f"Collected: {counts}")

with open(OUT_PATH, "w") as f:
    json.dump(results, f)

print(f"Saved to {OUT_PATH}")
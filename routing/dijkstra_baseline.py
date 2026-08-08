"""
Dijkstra Routing Baseline

Classical shortest-path routing baseline for the
Predictive Congestion-Aware Routing project.
"""

import time

import networkx as nx


def run_dijkstra(graph, source, destination, weight="weight"):
    """
    Find the shortest path between two nodes using Dijkstra's algorithm.

    Parameters
    ----------
    graph : nx.Graph
        Network topology.
    source : str
        Starting node.
    destination : str
        Destination node.
    weight : str
        Edge attribute used as the routing cost.

    Returns
    -------
    dict
        Routing results containing the path, cost, hop count,
        and execution time.
    """

    if source not in graph:
        raise ValueError(f"Source node '{source}' does not exist.")

    if destination not in graph:
        raise ValueError(f"Destination node '{destination}' does not exist.")

    start_time = time.perf_counter()

    path = nx.shortest_path(
        graph,
        source=source,
        target=destination,
        weight=weight,
        method="dijkstra",
    )

    total_cost = nx.path_weight(
        graph,
        path,
        weight=weight,
    )

    execution_time = time.perf_counter() - start_time

    return {
        "algorithm": "Dijkstra",
        "source": source,
        "destination": destination,
        "path": path,
        "total_cost": total_cost,
        "hop_count": len(path) - 1,
        "execution_time_ms": execution_time * 1000,
    }


def create_test_graph():
    """
    Create a small weighted graph for testing the baseline.

    This graph is only for verifying that the implementation works.
    Project experiments should later use the actual network topology.
    """

    graph = nx.Graph()

    edges = [
        ("A", "B", 4),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 5),
        ("C", "D", 8),
        ("C", "E", 10),
        ("D", "E", 2),
    ]

    for source, destination, weight in edges:
        graph.add_edge(
            source,
            destination,
            weight=weight,
        )

    return graph


def main():
    """Run a simple test of the Dijkstra baseline."""

    graph = create_test_graph()

    results = run_dijkstra(
        graph=graph,
        source="A",
        destination="E",
    )

    print("=== Dijkstra Baseline ===")
    print(f"Source: {results['source']}")
    print(f"Destination: {results['destination']}")
    print(f"Path: {' -> '.join(results['path'])}")
    print(f"Total Cost: {results['total_cost']}")
    print(f"Hop Count: {results['hop_count']}")
    print(f"Execution Time: {results['execution_time_ms']:.4f} ms")


if __name__ == "__main__":
    main()
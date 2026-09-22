"""Hyperbolic (Poincare disk) layout for the inferred airport network.

The inferred network (scripts.analytics.inferred_network) is hub-and-spoke:
a handful of high-degree hub airports connect to many low-degree spokes. Laid
out with a normal force-directed Euclidean graph, that structure crowds
everything into one blob at the center. Hyperbolic space has exponentially
more room the further out you go, so a tree naturally spreads across the disk
with each generation getting its own arc of circumference to expand into.

This is a lightweight, dependency-free hyperbolic embedding (no networkx):
a BFS spanning tree from the busiest hub is laid out with the classic
"angular subdivision" construction (each node's angular wedge is split
between its children), and depth is mapped to Poincare-disk radius with
r = tanh(depth * scale / 2) so nodes never reach the boundary.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from scripts.airports import AIRPORTS
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from airports import AIRPORTS

DEPTH_SCALE = 0.9
MIN_NODES_FOR_LAYOUT = 3


@dataclass
class HyperbolicLayout:
    ready: bool
    message: str
    nodes: pd.DataFrame = field(default_factory=pd.DataFrame)
    edges: pd.DataFrame = field(default_factory=pd.DataFrame)


def _poincare_point(radius: float, angle: float) -> tuple[float, float]:
    return radius * math.cos(angle), radius * math.sin(angle)


def _build_adjacency(edges_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    adjacency: dict[str, dict[str, int]] = {}
    for row in edges_df.itertuples():
        weight = int(row.observations)
        adjacency.setdefault(row.origin, {})[row.destination] = (
            adjacency.get(row.origin, {}).get(row.destination, 0) + weight
        )
        adjacency.setdefault(row.destination, {})[row.origin] = (
            adjacency.get(row.destination, {}).get(row.origin, 0) + weight
        )
    return adjacency


def _spanning_forest(adjacency: dict[str, dict[str, int]]) -> tuple[list[str], dict[str, str | None], dict[str, int]]:
    """Return BFS visit order, parent pointers, and depth for every node.

    Airports that never appear in an edge (no inferred transitions yet) are
    skipped; the layout only makes sense for nodes with at least one
    observed connection.
    """
    degree = {node: sum(neighbors.values()) for node, neighbors in adjacency.items()}
    unvisited = set(adjacency)
    order: list[str] = []
    parent: dict[str, str | None] = {}
    depth: dict[str, int] = {}

    while unvisited:
        root = max(unvisited, key=lambda node: degree.get(node, 0))
        parent[root] = None
        depth[root] = 0
        queue = deque([root])
        unvisited.discard(root)
        while queue:
            current = queue.popleft()
            order.append(current)
            neighbors = sorted(
                adjacency.get(current, {}).items(), key=lambda item: item[1], reverse=True
            )
            for neighbor, _weight in neighbors:
                if neighbor in unvisited:
                    unvisited.discard(neighbor)
                    parent[neighbor] = current
                    depth[neighbor] = depth[current] + 1
                    queue.append(neighbor)
    return order, parent, depth


def _assign_angles(order: list[str], parent: dict[str, str | None]) -> dict[str, float]:
    """Split each node's angular wedge evenly among its children (Reingold-Tilford style)."""
    children: dict[str | None, list[str]] = {}
    for node in order:
        children.setdefault(parent[node], []).append(node)

    wedge: dict[str, tuple[float, float]] = {}
    angle: dict[str, float] = {}

    roots = children.get(None, [])
    root_span = 2 * math.pi / max(len(roots), 1)
    for index, root in enumerate(roots):
        start = index * root_span
        wedge[root] = (start, start + root_span)
        angle[root] = start + root_span / 2

    for node in order:
        node_children = children.get(node, [])
        if not node_children:
            continue
        start, stop = wedge.get(node, (0.0, 2 * math.pi))
        span = (stop - start) / len(node_children)
        for index, child in enumerate(node_children):
            child_start = start + index * span
            wedge[child] = (child_start, child_start + span)
            angle[child] = child_start + span / 2
    return angle


def build_hyperbolic_layout(network_edges: pd.DataFrame) -> HyperbolicLayout:
    """Embed the inferred airport network in the Poincare disk."""
    if network_edges is None or network_edges.empty:
        return HyperbolicLayout(False, "No inferred network edges are available yet for a hyperbolic layout.")

    adjacency = _build_adjacency(network_edges)
    if len(adjacency) < MIN_NODES_FOR_LAYOUT:
        return HyperbolicLayout(
            False, f"At least {MIN_NODES_FOR_LAYOUT} connected airports are needed for a hyperbolic layout."
        )

    order, parent, depth = _spanning_forest(adjacency)
    angles = _assign_angles(order, parent)
    degree = {node: sum(neighbors.values()) for node, neighbors in adjacency.items()}

    node_rows = []
    coordinates: dict[str, tuple[float, float]] = {}
    for node in order:
        radius = math.tanh(depth[node] * DEPTH_SCALE / 2)
        x, y = _poincare_point(radius, angles[node])
        coordinates[node] = (x, y)
        airport = AIRPORTS.get(node, {})
        node_rows.append(
            {
                "airport_code": node,
                "city": airport.get("city", node),
                "depth": depth[node],
                "observations": degree[node],
                "x": x,
                "y": y,
                "radius": radius,
                "is_hub": depth[node] == 0,
            }
        )

    edge_rows = []
    seen_pairs = set()
    for row in network_edges.itertuples():
        pair = tuple(sorted((row.origin, row.destination)))
        if pair in seen_pairs or row.origin not in coordinates or row.destination not in coordinates:
            continue
        seen_pairs.add(pair)
        x0, y0 = coordinates[row.origin]
        x1, y1 = coordinates[row.destination]
        edge_rows.append(
            {
                "origin": row.origin,
                "destination": row.destination,
                "observations": row.observations,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            }
        )

    return HyperbolicLayout(
        True,
        "Hub airports sit near the disk center; each generation of spokes gets its own "
        "angular wedge, so hierarchy is legible without the crowding a Euclidean force "
        "layout would produce.",
        nodes=pd.DataFrame(node_rows),
        edges=pd.DataFrame(edge_rows),
    )

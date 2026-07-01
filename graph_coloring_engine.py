from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple


class GraphColoringSolver:
    @staticmethod
    def _normalize_vertices(vertices: Any) -> List[str]:
        if isinstance(vertices, str):
            raw_vertices = [part.strip() for part in vertices.replace("\n", ",").split(",") if part.strip()]
        elif isinstance(vertices, (list, tuple)):
            raw_vertices = [str(vertex).strip() for vertex in vertices if str(vertex).strip()]
        else:
            raw_vertices = []

        normalized: List[str] = []
        seen = set()
        for vertex in raw_vertices:
            if vertex not in seen:
                normalized.append(vertex)
                seen.add(vertex)
        return normalized

    @staticmethod
    def _normalize_edges(edges: Any) -> List[Tuple[str, str]]:
        raw_edges: List[Tuple[str, str]] = []

        if isinstance(edges, str):
            lines = [line.strip() for line in edges.splitlines() if line.strip()]
            for line in lines:
                left, right = line.replace("—", "-").replace("–", "-").split("-", 1) if "-" in line else line.split(",", 1)
                source = left.strip()
                target = right.strip()
                if source and target:
                    raw_edges.append((source, target))
        elif isinstance(edges, (list, tuple)):
            for edge in edges:
                source = target = ""
                if isinstance(edge, dict):
                    source = str(edge.get("source", "")).strip()
                    target = str(edge.get("target", "")).strip()
                elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    source = str(edge[0]).strip()
                    target = str(edge[1]).strip()
                else:
                    continue
                if source and target:
                    raw_edges.append((source, target))

        normalized: List[Tuple[str, str]] = []
        seen = set()
        for source, target in raw_edges:
            key = tuple(sorted((source, target)))
            if key not in seen:
                normalized.append((source, target))
                seen.add(key)
        return normalized

    @staticmethod
    def solve(vertices, edges, color_count, mode: str | None = None, transformation: Dict[str, Any] | None = None):
        vertex_list = GraphColoringSolver._normalize_vertices(vertices)
        edge_list = GraphColoringSolver._normalize_edges(edges)
        color_count = int(color_count)

        if color_count < 1:
            raise ValueError("Color count must be at least 1.")
        if not vertex_list:
            raise ValueError("Provide at least one vertex.")

        for source, target in edge_list:
            if source not in vertex_list:
                vertex_list.append(source)
            if target not in vertex_list:
                vertex_list.append(target)

        adjacency: Dict[str, set[str]] = {vertex: set() for vertex in vertex_list}
        for source, target in edge_list:
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
            if source == target:
                adjacency[source].add(target)

        order = sorted(vertex_list, key=lambda vertex: (-len(adjacency.get(vertex, set())), vertex))
        base_palette = [
            "#e53935",
            "#8e24aa",
            "#1e88e5",
            "#43a047",
            "#fb8c00",
            "#00acc1",
            "#fdd835",
            "#6d4c41",
        ]
        palette = [base_palette[index % len(base_palette)] for index in range(color_count)]

        coloring: Dict[str, int] = {}
        steps = []

        def snapshot(action: str, active_vertex: str | None = None, active_color: int | None = None, conflict_neighbor: str | None = None, accepted: bool = False, terminal: bool = False, message: str | None = None):
            steps.append({
                "action": action,
                "metrics": {
                    "vertex_count": len(vertex_list),
                    "edge_count": len(edge_list),
                    "color_count": color_count,
                    "colored_vertices": len(coloring),
                },
                "state": {
                    "vertices": vertex_list,
                    "edges": [{"source": source, "target": target} for source, target in edge_list],
                    "adjacency": {vertex: sorted(list(neighbors)) for vertex, neighbors in adjacency.items()},
                    "coloring": deepcopy(coloring),
                    "order": order,
                    "colorCount": color_count,
                    "colorPalette": palette,
                    "activeVertex": active_vertex,
                    "activeColor": active_color,
                    "conflictNeighbor": conflict_neighbor,
                    "message": message,
                    "isTerminal": terminal,
                    "isAccepted": accepted,
                },
            })

        snapshot("Initialized graph coloring search", message=f"Attempting to color {len(vertex_list)} vertices with {color_count} colors.")

        # If a transformation request is present and requests SAT/ILP encoding, emit a transformation frame
        if transformation and isinstance(transformation, dict) and transformation.get("id") in ("graph-coloring-to-sat", "graph-coloring-to-ilp"):
            steps.append({
                "action": "Graph Coloring → SAT/ILP encoding",
                "metrics": {"reduction": transformation.get("id"), "vertex_count": len(vertex_list), "edge_count": len(edge_list)},
                "state": {
                    "canReduce": True,
                    "formula": "Encode vertex-color boolean vars and edge constraints into CNF/linear constraints",
                    "encodedVariables": len(vertex_list) * color_count,
                    "isTerminal": True,
                },
            })

        # If mode requests minimization, attempt increasing k from 1..color_count
        minimize_mode = False
        if mode and str(mode).lower().startswith("min"):
            minimize_mode = True

        def is_safe(vertex: str, candidate_color: int) -> Tuple[bool, str | None]:
            for neighbor in adjacency.get(vertex, set()):
                if coloring.get(neighbor) == candidate_color:
                    return False, neighbor
            return True, None

        def backtrack(index: int) -> bool:
            if index >= len(order):
                return True

            vertex = order[index]
            for candidate_color in range(color_count):
                snapshot(
                    f"Try {vertex} with color {candidate_color + 1}",
                    active_vertex=vertex,
                    active_color=candidate_color,
                    message=f"Trying color {candidate_color + 1} on {vertex}."
                )
                safe, conflict_neighbor = is_safe(vertex, candidate_color)
                if not safe:
                    snapshot(
                        f"Conflict at {vertex} with color {candidate_color + 1}",
                        active_vertex=vertex,
                        active_color=candidate_color,
                        conflict_neighbor=conflict_neighbor,
                        message=f"{vertex} cannot use color {candidate_color + 1} because of {conflict_neighbor}."
                    )
                    continue

                coloring[vertex] = candidate_color
                snapshot(
                    f"Assigned {vertex} to color {candidate_color + 1}",
                    active_vertex=vertex,
                    active_color=candidate_color,
                    message=f"Assigned {vertex} to color {candidate_color + 1}."
                )

                if backtrack(index + 1):
                    return True

                snapshot(
                    f"Backtrack from {vertex} color {candidate_color + 1}",
                    active_vertex=vertex,
                    active_color=candidate_color,
                    message=f"Backtracking from {vertex} color {candidate_color + 1}."
                )
                coloring.pop(vertex, None)

            return False

        # If minimization requested, iterate k from 1..color_count
        if minimize_mode:
            for k in range(1, color_count + 1):
                # try with k colors
                snapshot(f"Minimization: try k={k}", message=f"Attempting chromatic search with k={k}")
                # rebuild palette for k
                palette = [base_palette[idx % len(base_palette)] for idx in range(k)]
                # reset coloring and steps offset
                coloring.clear()
                # local backtrack using current k
                def backtrack_k(index: int, k_local: int) -> bool:
                    if index >= len(order):
                        return True
                    vertex = order[index]
                    for candidate_color in range(k_local):
                        snapshot(f"Try {vertex} with color {candidate_color + 1}", active_vertex=vertex, active_color=candidate_color, message=f"Trying color {candidate_color + 1} on {vertex}.")
                        safe, conflict_neighbor = is_safe(vertex, candidate_color)
                        if not safe:
                            snapshot(f"Conflict at {vertex} with color {candidate_color + 1}", active_vertex=vertex, active_color=candidate_color, conflict_neighbor=conflict_neighbor, message=f"{vertex} cannot use color {candidate_color + 1} because of {conflict_neighbor}.")
                            continue
                        coloring[vertex] = candidate_color
                        snapshot(f"Assigned {vertex} to color {candidate_color + 1}", active_vertex=vertex, active_color=candidate_color, message=f"Assigned {vertex} to color {candidate_color + 1}.")
                        if backtrack_k(index + 1, k_local):
                            return True
                        snapshot(f"Backtrack from {vertex} color {candidate_color + 1}", active_vertex=vertex, active_color=candidate_color, message=f"Backtracking from {vertex} color {candidate_color + 1}.")
                        coloring.pop(vertex, None)
                    return False

                if backtrack_k(0, k):
                    snapshot("Graph coloring complete", accepted=True, terminal=True, message=f"Chromatic number ≤ {k}")
                    return k, steps
            # failed to find within bound
            snapshot("No coloring found within bounds", accepted=False, terminal=True, message=f"No valid coloring found with ≤ {color_count} colors.")
            return None, steps
        else:
            accepted = backtrack(0)

            if accepted:
                snapshot(
                    "Graph coloring complete",
                    accepted=True,
                    terminal=True,
                    message="A valid coloring was found."
                )
                chromatic_guess = max(coloring.values()) + 1 if coloring else 0
                return chromatic_guess, steps
            else:
                snapshot(
                    "No coloring found",
                    accepted=False,
                    terminal=True,
                    message=f"No valid coloring exists with {color_count} colors."
                )
                return None, steps

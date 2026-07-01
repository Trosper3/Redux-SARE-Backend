from collections import deque
from typing import Any, Dict, List, Set, Tuple


class MinSTCutDualSolver:

    # ── Graph helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_residual(
        nodes: List[str],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, int]]:
        residual: Dict[str, Dict[str, int]] = {n: {} for n in nodes}
        for e in edges:
            u, v, w = e["from"], e["to"], int(e["weight"])
            residual[u][v] = residual[u].get(v, 0) + w
            residual[v].setdefault(u, 0)
        return residual

    @staticmethod
    def _bfs_path(
        residual: Dict[str, Dict[str, int]],
        source: str,
        sink: str,
    ) -> Dict[str, str] | None:
        parent: Dict[str, str] = {source: source}
        q = deque([source])
        while q:
            cur = q.popleft()
            if cur == sink:
                return parent
            for nxt, cap in residual[cur].items():
                if cap > 0 and nxt not in parent:
                    parent[nxt] = cur
                    q.append(nxt)
        return None if sink not in parent else parent

    @staticmethod
    def _augment(
        residual: Dict[str, Dict[str, int]],
        parent: Dict[str, str],
        source: str,
        sink: str,
    ) -> Tuple[int, List[str]]:
        path: List[str] = []
        node = sink
        while node != source:
            path.append(node)
            node = parent[node]
        path.append(source)
        path.reverse()

        bottleneck = min(residual[path[i]][path[i + 1]] for i in range(len(path) - 1))
        for i in range(len(path) - 1):
            residual[path[i]][path[i + 1]] -= bottleneck
            residual[path[i + 1]][path[i]] = residual[path[i + 1]].get(path[i], 0) + bottleneck
        return bottleneck, path

    @staticmethod
    def _bfs_forward(
        residual: Dict[str, Dict[str, int]],
        source: str,
    ) -> Set[str]:
        visited: Set[str] = {source}
        q = deque([source])
        while q:
            cur = q.popleft()
            for nxt, cap in residual[cur].items():
                if cap > 0 and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return visited

    @staticmethod
    def _bfs_backward(
        residual: Dict[str, Dict[str, int]],
        nodes: List[str],
        sink: str,
    ) -> Set[str]:
        # Nodes from which sink is reachable — BFS on transpose of residual.
        transpose: Dict[str, List[str]] = {n: [] for n in nodes}
        for u, neighbors in residual.items():
            for v, cap in neighbors.items():
                if cap > 0 and v in transpose:
                    transpose[v].append(u)

        visited: Set[str] = {sink}
        q = deque([sink])
        while q:
            cur = q.popleft()
            for prev in transpose[cur]:
                if prev not in visited:
                    visited.add(prev)
                    q.append(prev)
        return visited

    # ── Step-frame helpers ───────────────────────────────────────────────────

    @staticmethod
    def _edge_list(edges: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        return [{"source": e["from"], "target": e["to"], "weight": e["weight"]} for e in edges]

    @staticmethod
    def _path_edges(path: List[str]) -> List[Dict[str, str]]:
        return [{"source": path[i], "target": path[i + 1]} for i in range(len(path) - 1)]

    # ── Main entry point ─────────────────────────────────────────────────────

    @staticmethod
    def solve(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes: List[str] = inputs.get("nodes", [])
        edges: List[Dict[str, Any]] = inputs.get("edges", [])
        source: str = inputs.get("source", "")
        sink: str = inputs.get("sink", "")
        algorithm: str = inputs.get("algorithm", "standard")

        if not nodes or source not in nodes or sink not in nodes:
            return [{"action": "Error: invalid graph", "metrics": {}, "state": {"isTerminal": True, "isAccepted": False}}]

        residual = MinSTCutDualSolver._build_residual(nodes, edges)
        edge_list = MinSTCutDualSolver._edge_list(edges)

        steps: List[Dict[str, Any]] = []

        # ── Initial frame ─────────────────────────────────────────────────
        steps.append({
            "action": f"Initial graph — source: {source}, sink: {sink}",
            "metrics": {"nodes": len(nodes), "edges": len(edges), "algorithm": algorithm},
            "state": {
                "nodes": nodes,
                "edges": edge_list,
                "activeNodes": [source, sink],
                "activeEdges": [],
                "phase": "initial",
                "algorithm": algorithm,
                "isTerminal": False,
            },
        })

        # ── Edmonds-Karp max-flow augmentation ────────────────────────────
        total_flow = 0
        while True:
            parent = MinSTCutDualSolver._bfs_path(residual, source, sink)
            if parent is None:
                break
            bottleneck, path = MinSTCutDualSolver._augment(residual, parent, source, sink)
            total_flow += bottleneck
            steps.append({
                "action": f"Augmenting path: {' → '.join(path)} (bottleneck = {bottleneck}, total flow = {total_flow})",
                "metrics": {"bottleneck": bottleneck, "total_flow": total_flow, "path_length": len(path)},
                "state": {
                    "nodes": nodes,
                    "edges": edge_list,
                    "activeNodes": path,
                    "activeEdges": MinSTCutDualSolver._path_edges(path),
                    "phase": "augmenting",
                    "algorithm": algorithm,
                    "isTerminal": False,
                },
            })

        steps.append({
            "action": f"Max flow saturated — total flow = {total_flow}. No more augmenting paths.",
            "metrics": {"max_flow": total_flow},
            "state": {
                "nodes": nodes,
                "edges": edge_list,
                "activeNodes": [],
                "activeEdges": [],
                "phase": "max-flow-done",
                "algorithm": algorithm,
                "isTerminal": False,
            },
        })

        # ── Source-side BFS ───────────────────────────────────────────────
        R_s = MinSTCutDualSolver._bfs_forward(residual, source)
        source_cut_edges = [
            e for e in edges if e["from"] in R_s and e["to"] not in R_s
        ]
        steps.append({
            "action": f"Source BFS: nodes reachable from '{source}' in residual graph → S-side = {{{', '.join(sorted(R_s))}}}",
            "metrics": {"s_side_size": len(R_s), "source_cut_edges": len(source_cut_edges)},
            "state": {
                "nodes": nodes,
                "edges": edge_list,
                "activeNodes": list(R_s),
                "activeEdges": [{"source": e["from"], "target": e["to"]} for e in source_cut_edges],
                "sourceSide": list(R_s),
                "phase": "source-bfs",
                "algorithm": algorithm,
                "isTerminal": False,
            },
        })

        if algorithm == "standard":
            # Standard: just the source-side cut
            cut_capacity = sum(e["weight"] for e in source_cut_edges)
            steps.append({
                "action": f"Min S-T Cut found — capacity = max flow = {total_flow}",
                "metrics": {"cut_capacity": cut_capacity, "max_flow": total_flow},
                "state": {
                    "nodes": nodes,
                    "edges": edge_list,
                    "activeNodes": list(R_s),
                    "activeEdges": [{"source": e["from"], "target": e["to"]} for e in source_cut_edges],
                    "sourceSide": list(R_s),
                    "sourceCutEdges": [{"from": e["from"], "to": e["to"], "weight": e["weight"]} for e in source_cut_edges],
                    "phase": "result",
                    "algorithm": "standard",
                    "cutCapacity": cut_capacity,
                    "isTerminal": True,
                    "isAccepted": True,
                },
            })

        else:
            # Dual-BFS: also compute sink-side and universal bottlenecks
            R_t = MinSTCutDualSolver._bfs_backward(residual, nodes, sink)
            sink_side = [n for n in nodes if n not in R_t]  # V \ R_t

            sink_cut_edges = [
                e for e in edges if e["from"] not in R_t and e["to"] in R_t
            ]
            steps.append({
                "action": f"Sink BFS: nodes that can reach '{sink}' in residual → R_t = {{{', '.join(sorted(R_t))}}}  |  Sink-side S = V\\R_t = {{{', '.join(sorted(sink_side))}}}",
                "metrics": {"r_t_size": len(R_t), "sink_side_size": len(sink_side), "sink_cut_edges": len(sink_cut_edges)},
                "state": {
                    "nodes": nodes,
                    "edges": edge_list,
                    "activeNodes": sink_side,
                    "activeEdges": [{"source": e["from"], "target": e["to"]} for e in sink_cut_edges],
                    "sourceSide": list(R_s),
                    "sinkSide": sink_side,
                    "phase": "sink-bfs",
                    "algorithm": algorithm,
                    "isTerminal": False,
                },
            })

            # Universal bottleneck edges = edges in BOTH cuts
            source_cut_set = {(e["from"], e["to"]) for e in source_cut_edges}
            sink_cut_set = {(e["from"], e["to"]) for e in sink_cut_edges}
            universal_pairs = source_cut_set & sink_cut_set
            universal_edges = [
                e for e in edges if (e["from"], e["to"]) in universal_pairs
            ]

            cut_capacity = sum(e["weight"] for e in source_cut_edges)

            universal_description = (
                f"{{{', '.join(f'{p[0]}→{p[1]}' for p in sorted(universal_pairs))}}}"
                if universal_pairs else "∅ (none — the two cuts differ completely)"
            )

            steps.append({
                "action": f"Universal bottleneck edges (in both cuts) = {universal_description}",
                "metrics": {
                    "source_cut_edges": len(source_cut_edges),
                    "sink_cut_edges": len(sink_cut_edges),
                    "universal_edges": len(universal_edges),
                    "cut_capacity": cut_capacity,
                },
                "state": {
                    "nodes": nodes,
                    "edges": edge_list,
                    "activeNodes": list(R_s),
                    "activeEdges": [{"source": e["from"], "target": e["to"]} for e in universal_edges],
                    "sourceSide": list(R_s),
                    "sinkSide": sink_side,
                    "sourceCutEdges": [{"from": e["from"], "to": e["to"], "weight": e["weight"]} for e in source_cut_edges],
                    "sinkCutEdges": [{"from": e["from"], "to": e["to"], "weight": e["weight"]} for e in sink_cut_edges],
                    "universalEdges": [{"from": e["from"], "to": e["to"], "weight": e["weight"]} for e in universal_edges],
                    "phase": "result",
                    "algorithm": "dual-bfs",
                    "cutCapacity": cut_capacity,
                    "isTerminal": True,
                    "isAccepted": True,
                },
            })

        return steps

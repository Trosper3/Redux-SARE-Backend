import re
from typing import Any, Dict, List, Tuple


class CliqueSolver:
    @staticmethod
    def _normalize_nodes(nodes: Any) -> List[str]:
        if isinstance(nodes, str):
            raw_tokens = re.split(r"[\n,]", nodes)
        elif isinstance(nodes, (list, tuple, set)):
            raw_tokens = [str(token) for token in nodes]
        else:
            raw_tokens = [str(nodes)] if nodes is not None else []

        normalized: List[str] = []
        seen = set()
        for token in raw_tokens:
            name = str(token).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    @staticmethod
    def _normalize_edges(edges: Any) -> List[Tuple[str, str]]:
        if isinstance(edges, str):
            raw_lines = edges.splitlines()
        elif isinstance(edges, (list, tuple, set)):
            raw_lines = list(edges)
        else:
            raw_lines = []

        normalized: List[Tuple[str, str]] = []
        seen = set()

        for raw_line in raw_lines:
            if isinstance(raw_line, (list, tuple)) and len(raw_line) >= 2:
                left = str(raw_line[0]).strip()
                right = str(raw_line[1]).strip()
            else:
                text = str(raw_line).strip()
                if not text:
                    continue
                parts = [part.strip() for part in re.split(r"[\-,>]", text) if part.strip()]
                if len(parts) < 2:
                    parts = [part.strip() for part in re.split(r"\s+", text) if part.strip()]
                if len(parts) < 2:
                    continue
                left, right = parts[0], parts[1]

            if not left or not right or left == right:
                continue

            edge = tuple(sorted((left, right)))
            if edge in seen:
                continue

            seen.add(edge)
            normalized.append(edge)

        return normalized

    @staticmethod
    def _format_subset(subset: List[str]) -> str:
        return "{" + ", ".join(subset) + "}" if subset else "∅"

    @staticmethod
    def _pairwise_edges(subset: List[str]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for index, left in enumerate(subset):
            for right in subset[index + 1 :]:
                pairs.append(tuple(sorted((left, right))))
        return pairs

    @staticmethod
    def _build_state(
        nodes: List[str],
        graph_edges: List[Dict[str, str]],
        active_nodes: List[str],
        active_edges: List[Tuple[str, str]],
        best_clique: List[str],
        *,
        is_terminal: bool = False,
        is_accepted: bool = False,
        explored_edges: List[Tuple[str, str]] | None = None,
    ) -> Dict[str, Any]:
        return {
            "nodes": nodes,
            "edges": graph_edges,
            "activeNodes": active_nodes,
            "activeEdges": [{"source": left, "target": right} for left, right in active_edges],
            "allActiveEdges": [{"source": left, "target": right} for left, right in (explored_edges or active_edges)],
            "currentClique": active_nodes,
            "bestClique": best_clique,
            "isTerminal": is_terminal,
            "isAccepted": is_accepted,
        }

    @staticmethod
    def solve(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = CliqueSolver._normalize_nodes(inputs.get("nodes", []))
        edges = CliqueSolver._normalize_edges(inputs.get("edges", []))
        algorithm = str(inputs.get("algorithm") or "brute-force").strip().lower()

        if len(nodes) == 0:
            raise ValueError("Provide at least one node.")

        if len(nodes) > 12:
            raise ValueError("Clique visualization is limited to 12 nodes so the step trace stays readable.")

        node_set = set(nodes)
        for left, right in edges:
            if left not in node_set:
                nodes.append(left)
                node_set.add(left)
            if right not in node_set:
                nodes.append(right)
                node_set.add(right)

        graph_edge_list = [{"source": left, "target": right} for left, right in edges]
        adjacency = {node: set() for node in nodes}
        for left, right in edges:
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)

        def has_edge(left: str, right: str) -> bool:
            return right in adjacency.get(left, set())

        def is_clique(subset: List[str]) -> bool:
            for index, left in enumerate(subset):
                for right in subset[index + 1 :]:
                    if not has_edge(left, right):
                        return False
            return True

        steps: List[Dict[str, Any]] = []
        best_clique: List[str] = []
        explored_best_edges: List[Tuple[str, str]] = []
        subset_checks = 0
        pruned_branches = 0

        steps.append({
            "action": f"Initialized clique search using {algorithm.replace('-', ' ')}",
            "metrics": {
                "nodes": len(nodes),
                "edges": len(edges),
                "algorithm": algorithm,
            },
            "state": CliqueSolver._build_state(nodes, graph_edge_list, [], [], [], explored_edges=[]),
        })

        if algorithm == "branch-and-bound":
            ordered_nodes = sorted(nodes, key=lambda node: (-len(adjacency.get(node, set())), node))

            def recurse(current: List[str], candidates: List[str], depth: int) -> None:
                nonlocal best_clique, explored_best_edges, pruned_branches

                current_edges = CliqueSolver._pairwise_edges(current)
                best_edges = CliqueSolver._pairwise_edges(best_clique)
                steps.append({
                    "action": f"Exploring depth {depth} with clique {CliqueSolver._format_subset(current)} and {len(candidates)} candidates",
                    "metrics": {
                        "depth": depth,
                        "current_size": len(current),
                        "candidate_count": len(candidates),
                        "best_size": len(best_clique),
                    },
                    "state": CliqueSolver._build_state(nodes, graph_edge_list, current[:], current_edges, best_clique[:], explored_edges=best_edges),
                })

                if len(current) + len(candidates) <= len(best_clique):
                    pruned_branches += 1
                    steps.append({
                        "action": f"Pruned branch at depth {depth}: even the upper bound cannot beat clique size {len(best_clique)}",
                        "metrics": {
                            "depth": depth,
                            "upper_bound": len(current) + len(candidates),
                            "best_size": len(best_clique),
                            "pruned_branches": pruned_branches,
                        },
                        "state": CliqueSolver._build_state(nodes, graph_edge_list, current[:], current_edges, best_clique[:], explored_edges=best_edges),
                    })
                    return

                if not candidates:
                    if len(current) > len(best_clique):
                        best_clique = current[:]
                        explored_best_edges = CliqueSolver._pairwise_edges(best_clique)
                        steps.append({
                            "action": f"Found new best clique {CliqueSolver._format_subset(best_clique)}",
                            "metrics": {
                                "best_size": len(best_clique),
                                "best_clique": best_clique[:],
                                "pruned_branches": pruned_branches,
                            },
                            "state": CliqueSolver._build_state(nodes, graph_edge_list, best_clique[:], CliqueSolver._pairwise_edges(best_clique), best_clique[:], explored_edges=explored_best_edges),
                        })
                    return

                for index, vertex in enumerate(candidates):
                    next_current = current + [vertex]
                    next_candidates = [
                        candidate
                        for candidate in candidates[index + 1 :]
                        if candidate in adjacency.get(vertex, set()) and all(candidate in adjacency.get(member, set()) for member in current)
                    ]
                    steps.append({
                        "action": f"Choose vertex {vertex} and continue the search",
                        "metrics": {
                            "vertex": vertex,
                            "current_size": len(next_current),
                            "remaining_candidates": len(next_candidates),
                        },
                        "state": CliqueSolver._build_state(nodes, graph_edge_list, next_current[:], CliqueSolver._pairwise_edges(next_current), best_clique[:], explored_edges=explored_best_edges),
                    })
                    recurse(next_current, next_candidates, depth + 1)

            recurse([], ordered_nodes, 0)
        else:
            total_masks = 1 << len(nodes)
            for mask in range(1, total_masks):
                subset_checks += 1
                subset = [nodes[index] for index in range(len(nodes)) if mask & (1 << index)]
                clique_flag = is_clique(subset)
                candidate_edges = CliqueSolver._pairwise_edges(subset)

                if clique_flag and len(subset) > len(best_clique):
                    best_clique = subset[:]
                    explored_best_edges = candidate_edges[:]
                    steps.append({
                        "action": f"New best clique found: {CliqueSolver._format_subset(best_clique)}",
                        "metrics": {
                            "subset_checks": subset_checks,
                            "best_size": len(best_clique),
                            "subset": subset[:],
                            "is_clique": True,
                        },
                        "state": CliqueSolver._build_state(nodes, graph_edge_list, subset[:], candidate_edges, best_clique[:], explored_edges=explored_best_edges),
                    })

                steps.append({
                    "action": f"Checking subset {CliqueSolver._format_subset(subset)}",
                    "metrics": {
                        "subset_checks": subset_checks,
                        "subset_size": len(subset),
                        "is_clique": clique_flag,
                        "best_size": len(best_clique),
                    },
                    "state": CliqueSolver._build_state(nodes, graph_edge_list, subset[:], candidate_edges, best_clique[:], explored_edges=explored_best_edges),
                })

        steps.append({
            "action": f"Clique search complete using {algorithm.replace('-', ' ')}",
            "metrics": {
                "algorithm": algorithm,
                "best_size": len(best_clique),
                "best_clique": best_clique[:],
                "subset_checks": subset_checks,
                "pruned_branches": pruned_branches,
            },
            "state": CliqueSolver._build_state(
                nodes,
                graph_edge_list,
                best_clique[:],
                CliqueSolver._pairwise_edges(best_clique),
                best_clique[:],
                is_terminal=True,
                is_accepted=len(best_clique) > 0,
                explored_edges=explored_best_edges,
            ),
        })

        return steps
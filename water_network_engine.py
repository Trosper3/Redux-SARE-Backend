"""
water_network_engine.py — Phase 1 (Physics) engine for the water-allocation lab.

Extends min_st_cut_dual_engine's Edmonds-Karp + dual-BFS core with:
  - Min-slack computation: how much a universal bottleneck edge's capacity can
    grow before a different cut becomes the minimum cut (GGT-style incremental
    augmentation on the residual graph).
  - Warm-started re-solve: the solve_upgrade path deep-copies the before-residual
    and injects the capacity delta before re-augmenting.  Labelled "warm start"
    in all frame actions — never claimed as O(local).

Routing:
  problem_id == 'water-allocation'  (no transformation)  → WaterNetworkEngine.solve()
  problem_id == 'water-allocation'  + transformation.id == 'upgrade-pipe'  → .solve_upgrade()

Accepts either:
  - WaterNetwork JSON  (has "pipes" key)  — adapted via water_model.water_to_flow_inputs
  - Pre-adapted flow JSON  (has "nodes"/"edges"/"source"/"sink" keys)
"""

from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from water_model import parse_water_network, water_to_flow_inputs


# ── Low-level residual-graph helpers ──────────────────────────────────────────

def _build_residual(
    nodes: List[str], edges: List[Dict[str, Any]]
) -> Dict[str, Dict[str, int]]:
    res: Dict[str, Dict[str, int]] = {n: {} for n in nodes}
    for e in edges:
        u, v, w = e["from"], e["to"], int(e["weight"])
        res[u][v] = res[u].get(v, 0) + w
        res[v].setdefault(u, 0)
    return res


def _bfs_path(
    res: Dict[str, Dict[str, int]], source: str, sink: str
) -> Optional[Dict[str, str]]:
    parent: Dict[str, str] = {source: source}
    q = deque([source])
    while q:
        cur = q.popleft()
        if cur == sink:
            return parent
        for nxt, cap in res[cur].items():
            if cap > 0 and nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    return None if sink not in parent else parent


def _augment(
    res: Dict[str, Dict[str, int]],
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
    bn = min(res[path[i]][path[i + 1]] for i in range(len(path) - 1))
    for i in range(len(path) - 1):
        res[path[i]][path[i + 1]] -= bn
        res[path[i + 1]][path[i]] = res[path[i + 1]].get(path[i], 0) + bn
    return bn, path


def _bfs_forward(res: Dict[str, Dict[str, int]], source: str) -> Set[str]:
    visited: Set[str] = {source}
    q = deque([source])
    while q:
        cur = q.popleft()
        for nxt, cap in res[cur].items():
            if cap > 0 and nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
    return visited


def _bfs_backward(
    res: Dict[str, Dict[str, int]], nodes: List[str], sink: str
) -> Set[str]:
    transpose: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, nbrs in res.items():
        for v, cap in nbrs.items():
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


def _edge_list(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"source": e["from"], "target": e["to"], "weight": e["weight"]} for e in edges]


def _path_edges(path: List[str]) -> List[Dict[str, str]]:
    return [{"source": path[i], "target": path[i + 1]} for i in range(len(path) - 1)]


# ── Max-flow augmentation loop ────────────────────────────────────────────────

def _run_max_flow(
    nodes: List[str],
    edges: List[Dict[str, Any]],
    source: str,
    sink: str,
    warm_residual: Optional[Dict[str, Dict[str, int]]] = None,
    warm_flow: int = 0,
) -> Tuple[int, Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    """
    Edmonds-Karp max-flow.  Pass warm_residual + warm_flow for a warm start
    (residual already has prior flow baked in).
    Returns (total_flow, final_residual, step_frames).
    """
    el = _edge_list(edges)

    if warm_residual is not None:
        res = deepcopy(warm_residual)
        total_flow = warm_flow
        warm_note = " (warm-started from previous residual)"
    else:
        res = _build_residual(nodes, edges)
        total_flow = 0
        warm_note = ""

    steps: List[Dict[str, Any]] = [{
        "action": f"Initial graph — source: {source}, sink: {sink}{warm_note}",
        "metrics": {"nodes": len(nodes), "edges": len(edges), "warm_start": warm_residual is not None},
        "state": {
            "nodes": nodes, "edges": el,
            "activeNodes": [source, sink], "activeEdges": [],
            "phase": "initial", "isTerminal": False,
            "warmStart": warm_residual is not None,
        },
    }]

    while True:
        parent = _bfs_path(res, source, sink)
        if parent is None:
            break
        bn, path = _augment(res, parent, source, sink)
        total_flow += bn
        steps.append({
            "action": f"Augmenting path: {' → '.join(path)} (bottleneck = {bn}, total = {total_flow})",
            "metrics": {"bottleneck": bn, "total_flow": total_flow, "path_length": len(path)},
            "state": {
                "nodes": nodes, "edges": el,
                "activeNodes": path, "activeEdges": _path_edges(path),
                "phase": "augmenting", "isTerminal": False,
            },
        })

    steps.append({
        "action": f"Max flow saturated — total flow = {total_flow}. No augmenting paths remain.",
        "metrics": {"max_flow": total_flow},
        "state": {
            "nodes": nodes, "edges": el,
            "activeNodes": [], "activeEdges": [],
            "phase": "max-flow-done", "isTerminal": False,
        },
    })
    return total_flow, res, steps


# ── Dual-BFS cut analysis ─────────────────────────────────────────────────────

def _run_dual_bfs(
    nodes: List[str],
    edges: List[Dict[str, Any]],
    source: str,
    sink: str,
    res: Dict[str, Dict[str, int]],
    total_flow: int,
    prior_steps: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Append dual-BFS cut frames.  Returns (all_steps, terminal_state_dict).
    """
    steps = list(prior_steps)
    el = _edge_list(edges)

    R_s = _bfs_forward(res, source)
    src_cut = [e for e in edges if e["from"] in R_s and e["to"] not in R_s]
    steps.append({
        "action": f"Source BFS: S-side = {{{', '.join(sorted(R_s))}}}",
        "metrics": {"s_side_size": len(R_s), "source_cut_edges": len(src_cut)},
        "state": {
            "nodes": nodes, "edges": el,
            "activeNodes": list(R_s),
            "activeEdges": [{"source": e["from"], "target": e["to"]} for e in src_cut],
            "sourceSide": list(R_s), "phase": "source-bfs", "isTerminal": False,
        },
    })

    R_t = _bfs_backward(res, nodes, sink)
    sink_side = [n for n in nodes if n not in R_t]
    snk_cut = [e for e in edges if e["from"] not in R_t and e["to"] in R_t]
    steps.append({
        "action": f"Sink BFS: R_t = {{{', '.join(sorted(R_t))}}}",
        "metrics": {"r_t_size": len(R_t), "sink_cut_edges": len(snk_cut)},
        "state": {
            "nodes": nodes, "edges": el,
            "activeNodes": sink_side,
            "activeEdges": [{"source": e["from"], "target": e["to"]} for e in snk_cut],
            "sourceSide": list(R_s), "sinkSide": sink_side,
            "phase": "sink-bfs", "isTerminal": False,
        },
    })

    src_set = {(e["from"], e["to"]) for e in src_cut}
    snk_set = {(e["from"], e["to"]) for e in snk_cut}
    universal_pairs = src_set & snk_set
    universal = [e for e in edges if (e["from"], e["to"]) in universal_pairs]

    cut_capacity = sum(e["weight"] for e in src_cut)

    # Per-edge flows for residual hover labels: flow = capacity − residual_forward_capacity
    edge_flows: Dict[str, int] = {}
    for e in edges:
        u, v, cap = e["from"], e["to"], int(e["weight"])
        edge_flows[f"{u}|{v}"] = max(0, cap - res.get(u, {}).get(v, 0))

    slack_data = _compute_min_slack(nodes, edges, res, source, sink, universal_pairs)

    u_desc = (
        f"{{{', '.join(f'{p[0]}→{p[1]}' for p in sorted(universal_pairs))}}}"
        if universal_pairs else "∅ (no universal edge — the two cuts differ)"
    )

    terminal_state: Dict[str, Any] = {
        "nodes": nodes, "edges": el,
        "activeNodes": list(R_s),
        "activeEdges": [{"source": e["from"], "target": e["to"]} for e in universal],
        "sourceSide": list(R_s), "sinkSide": sink_side,
        "sourceCutEdges": [{"from": e["from"], "to": e["to"], "weight": e["weight"]} for e in src_cut],
        "sinkCutEdges":   [{"from": e["from"], "to": e["to"], "weight": e["weight"]} for e in snk_cut],
        "universalEdges": [{"from": e["from"], "to": e["to"], "weight": e["weight"]} for e in universal],
        "edgeFlows": edge_flows,
        "phase": "result", "cutCapacity": cut_capacity,
        "slackUp": slack_data.get("slackUp"),
        "nextBottleneck": slack_data.get("nextBottleneck"),
        "isTerminal": True, "isAccepted": True,
    }

    steps.append({
        "action": (
            f"Universal bottleneck edges = {u_desc}. "
            f"Cut capacity = max flow = {total_flow}."
            + (f"  Slack on {slack_data.get('targetEdge')}: +{slack_data.get('slackUp')} before cut shifts."
               if slack_data.get("slackUp") else "")
        ),
        "metrics": {
            "cut_capacity": cut_capacity, "max_flow": total_flow,
            "universal_edges": len(universal),
            "slackUp": slack_data.get("slackUp"),
        },
        "state": terminal_state,
    })
    return steps, terminal_state


# ── Min-slack computation ──────────────────────────────────────────────────────

def _compute_min_slack(
    nodes: List[str],
    edges: List[Dict[str, Any]],
    res: Dict[str, Dict[str, int]],
    source: str,
    sink: str,
    universal_pairs: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    For each universal bottleneck edge (u,v), compute slackUp: the number of
    additional capacity units that can flow through (u,v) before a DIFFERENT
    cut becomes binding.

    Method: incremental single-unit augmentation on a copy of the residual.
    We add 1 capacity unit to (u,v) and augment; if the path goes through
    (u,v) we count it as slack; if not (or no path), we report the new
    bottleneck.  Worst-case O(slackUp * V * E) — fine for demo-sized graphs.
    """
    if not universal_pairs:
        return {}

    u, v = sorted(universal_pairs)[0]

    test_res = deepcopy(res)
    slack = 0
    next_bottleneck: Optional[List[str]] = None

    for _ in range(200):
        test_res[u][v] = test_res[u].get(v, 0) + 1
        parent = _bfs_path(test_res, source, sink)
        if parent is None:
            break

        path: List[str] = []
        node = sink
        while node != source:
            path.append(node)
            node = parent[node]
        path.append(source)
        path.reverse()

        path_edge_set = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
        if (u, v) not in path_edge_set:
            # Bottleneck has shifted to a different edge
            bn = min(test_res[path[i]][path[i + 1]] for i in range(len(path) - 1))
            for i in range(len(path) - 1):
                if test_res[path[i]][path[i + 1]] == bn:
                    next_bottleneck = [path[i], path[i + 1]]
                    break
            break

        # Route 1 unit through (u,v)
        for i in range(len(path) - 1):
            test_res[path[i]][path[i + 1]] -= 1
            test_res[path[i + 1]][path[i]] = test_res[path[i + 1]].get(path[i], 0) + 1
        slack += 1

    return {
        "slackUp": slack if slack > 0 else None,
        "nextBottleneck": next_bottleneck,
        "targetEdge": [u, v],
    }


# ── Input normalisation ────────────────────────────────────────────────────────

def _normalise_inputs(data: Dict[str, Any]) -> Dict[str, Any]:
    if "pipes" in data:
        network = parse_water_network(data)
        return water_to_flow_inputs(network)
    fi = dict(data)
    if fi.get("edges") and "source" in (fi["edges"][0] if fi["edges"] else {}):
        fi["edges"] = [
            {"from": e["source"], "to": e["target"],
             "weight": e.get("weight", e.get("capacity", 1))}
            for e in fi["edges"]
        ]
    return fi


# ── Public API ─────────────────────────────────────────────────────────────────

class WaterNetworkEngine:

    @staticmethod
    def solve(data: Dict[str, Any], detail: str = "full") -> List[Dict[str, Any]]:
        """
        Phase 1 solve: max-flow + dual-BFS cut + min-slack.
        detail='full'   → full step movie (default, backward-compatible)
        detail='result' → terminal frame only (cuts serialization cost)
        """
        fi = _normalise_inputs(data)
        nodes, edges, source, sink = fi["nodes"], fi["edges"], fi["source"], fi["sink"]

        if not nodes or source not in nodes or sink not in nodes:
            return [{
                "action": "Error: invalid graph (check source/sink node ids)",
                "metrics": {}, "state": {"isTerminal": True, "isAccepted": False},
            }]

        flow, res, aug_steps = _run_max_flow(nodes, edges, source, sink)
        steps, _ = _run_dual_bfs(nodes, edges, source, sink, res, flow, aug_steps)
        return steps if detail == "full" else [steps[-1]]

    @staticmethod
    def solve_upgrade(data: Dict[str, Any], detail: str = "full") -> Dict[str, Any]:
        """
        Pipe-upgrade what-if (Task A.2).

        Expects data["transformation"] = {"id": "upgrade-pipe", "pipe": [u, v], "delta": N}.
        Returns {"before": List[StepFrame], "after": List[StepFrame], "delta": summary}.
        """
        t = data.get("transformation") or {}
        pipe_pair = t.get("pipe", [])
        delta = int(t.get("delta", 1))

        if len(pipe_pair) != 2:
            return {"error": "transformation.pipe must be [u, v]",
                    "before": [], "after": [], "delta": {}}

        u, v = str(pipe_pair[0]), str(pipe_pair[1])

        fi = _normalise_inputs({k: val for k, val in data.items() if k != "transformation"})
        nodes, edges, source, sink = fi["nodes"], fi["edges"], fi["source"], fi["sink"]

        # ── Before ──────────────────────────────────────────────────────────
        before_flow, before_res, before_aug = _run_max_flow(nodes, edges, source, sink)
        before_steps, before_terminal = _run_dual_bfs(
            nodes, edges, source, sink, before_res, before_flow, before_aug
        )

        # ── Check the pipe exists ────────────────────────────────────────────
        if not any(e["from"] == u and e["to"] == v for e in edges):
            return {
                "error": f"Pipe {u}→{v} not found in graph",
                "before": before_steps, "after": [], "delta": {},
            }

        # ── After: build upgraded edge list ─────────────────────────────────
        after_edges = [
            {**e, "weight": e["weight"] + delta}
            if e["from"] == u and e["to"] == v else e
            for e in edges
        ]

        # Warm-start: copy before_res and inject delta onto (u,v)
        warm_res = deepcopy(before_res)
        warm_res.setdefault(u, {})
        warm_res[u][v] = warm_res[u].get(v, 0) + delta

        after_warm_steps: List[Dict[str, Any]] = [{
            "action": (
                f"Warm start — upgrade {u}→{v} +{delta} capacity injected into previous residual. "
                f"Prior flow preserved (no recompute from scratch)."
            ),
            "metrics": {"before_flow": before_flow, "warm_start": True, "delta": delta},
            "state": {
                "nodes": nodes, "edges": _edge_list(after_edges),
                "activeNodes": [u, v],
                "activeEdges": [{"source": u, "target": v}],
                "phase": "warm-start", "isTerminal": False, "warmStart": True,
            },
        }]

        add_flow = 0
        while True:
            parent = _bfs_path(warm_res, source, sink)
            if parent is None:
                break
            bn, path = _augment(warm_res, parent, source, sink)
            add_flow += bn
            after_warm_steps.append({
                "action": (
                    f"Warm augmenting: {' → '.join(path)} "
                    f"(bottleneck = {bn}, additional = {add_flow})"
                ),
                "metrics": {"bottleneck": bn, "additional_flow": add_flow},
                "state": {
                    "nodes": nodes, "edges": _edge_list(after_edges),
                    "activeNodes": path, "activeEdges": _path_edges(path),
                    "phase": "augmenting", "isTerminal": False, "warmStart": True,
                },
            })

        after_flow = before_flow + add_flow
        after_warm_steps.append({
            "action": (
                f"After upgrade: max flow = {after_flow} "
                f"(was {before_flow}, Δ = +{add_flow})."
            ),
            "metrics": {"after_flow": after_flow, "additional_flow": add_flow},
            "state": {
                "nodes": nodes, "edges": _edge_list(after_edges),
                "activeNodes": [], "activeEdges": [],
                "phase": "max-flow-done", "isTerminal": False,
            },
        })

        after_steps, after_terminal = _run_dual_bfs(
            nodes, after_edges, source, sink, warm_res, after_flow, after_warm_steps
        )

        # ── Build diff summary ───────────────────────────────────────────────
        b_univ = before_terminal.get("universalEdges", [])
        a_univ = after_terminal.get("universalEdges", [])
        b_set = {(e["from"], e["to"]) for e in b_univ}
        a_set = {(e["from"], e["to"]) for e in a_univ}

        out_before = before_steps if detail == "full" else [before_steps[-1]]
        out_after  = after_steps  if detail == "full" else [after_steps[-1]]
        return {
            "before": out_before,
            "after": out_after,
            "delta": {
                "maxFlowBefore": before_flow,
                "maxFlowAfter": after_flow,
                "deltaFlow": after_flow - before_flow,
                "upgradedPipe": [u, v],
                "pipeDelta": delta,
                "bottleneckBefore": [f"{e['from']}→{e['to']}" for e in b_univ],
                "bottleneckAfter":  [f"{e['from']}→{e['to']}" for e in a_univ],
                "bottleneckMoved":  b_set != a_set,
                "warmStart": True,
            },
        }

"""
farm_selection_engine.py — Phase 2 engine: Branch & Bound farm selection.

Input: WaterNetwork JSON (same format as Phase 1) + optional fields:
    bound_mode: 'fractional' | 'network'  (default: 'fractional')
    max_flow:   int  (pre-computed Phase 1 capacity; computed if omitted)

Output: ordered step frames per the universal engine contract.
    state.tree          — B&B tree node list (for BranchBoundTree.tsx)
    state.selectedFarms — current incumbent farm-ID set

Two bound modes:
  fractional — O(n log n) fractional-knapsack upper bound. Fast and pedagogical;
               default for kid/student layers. Ignores network topology (a bound
               that fires on the single max-flow scalar may accept a set that can't
               actually route — leaves are validated by demand-sum feasibility only).
  network    — Every leaf is validated with an actual max-flow feasibility check:
               a subset is accepted only if flow(subset) ≥ sum(selected demands).
               Respects interior cuts; used for the professional layer. Interior
               nodes still use fractional bounding; only leaves pay the flow cost.

Fallback: len(farms) > FPTAS_THRESHOLD → FPTAS(ε=0.001).

Honesty notes embedded in frame 'action' strings:
  - fractional bound result: "single-capacity bound — may include sets that can't route"
  - network bound result: "network-feasible, exact for this size"
  - FPTAS result: "approximate (≤0.1% error)"
"""

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from .water_model import WaterFarm, WaterNetwork, parse_water_network, water_to_flow_inputs
from .water_network_engine import _run_max_flow

FPTAS_THRESHOLD = 10_000
EXACT_TIMEOUT_SEC = 10.0   # wall-clock cap; falls back to greedy on timeout


def _reachable_from_source(network: WaterNetwork) -> Set[str]:
    """BFS over pipe graph to find all nodes reachable from the flow source."""
    adj: Dict[str, List[str]] = {}
    for pipe in network.pipes:
        adj.setdefault(pipe.from_, []).append(pipe.to)

    # Multi-source: virtual super-source fans out to each real source
    start_nodes = network.sources if len(network.sources) > 1 else [network.source]

    visited: Set[str] = set(start_nodes)
    queue: List[str] = list(start_nodes)
    while queue:
        node = queue.pop(0)
        for nbr in adj.get(node, []):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return visited


# ── Fractional-knapsack bound ─────────────────────────────────────────────────

def _frac_bound(
    farms: List[WaterFarm],
    start_depth: int,
    current_demand: int,
    current_value: int,
    capacity: int,
) -> float:
    """
    Upper bound via fractional knapsack on farms[start_depth:].
    farms must be sorted by value/demand descending for the bound to be tight.
    Returns -1.0 if current_demand already exceeds capacity.
    """
    remaining = capacity - current_demand
    if remaining < 0:
        return -1.0
    val = float(current_value)
    for i in range(start_depth, len(farms)):
        f = farms[i]
        if remaining >= f.demand:
            val += f.value
            remaining -= f.demand
        else:
            val += f.value * (remaining / max(f.demand, 1))
            break
    return val


# ── Network feasibility check (used for 'network' bound at leaves) ────────────

def _check_network_feasible(
    included: List[WaterFarm],
    network: WaterNetwork,
) -> bool:
    """
    True iff a valid flow can simultaneously satisfy all included farms' demands.
    Builds a sub-graph with only the included farms' sink edges and checks
    max-flow == total demand.
    """
    if not included:
        return True
    total_demand = sum(f.demand for f in included)
    nodes = [n.id for n in network.nodes] + [network.sink]
    edges = [{"from": p.from_, "to": p.to, "weight": p.capacity} for p in network.pipes]
    for farm in included:
        edges.append({"from": farm.nodeId, "to": network.sink, "weight": farm.demand})
    flow, _, _ = _run_max_flow(nodes, edges, network.source, network.sink)
    return flow >= total_demand


# ── Greedy heuristic (used as warm start + timeout fallback) ─────────────────

def _greedy_heuristic(
    farms: List[WaterFarm],
    capacity: int,
    network: Optional[WaterNetwork] = None,
    check_network: bool = False,
) -> Tuple[List[WaterFarm], int]:
    """
    Sort by value/demand ratio, greedily include farms that fit.
    If check_network, validate feasibility via max-flow before accepting.
    Returns (selected_farms, total_value).
    """
    sorted_farms = sorted(farms, key=lambda f: f.value / max(f.demand, 1), reverse=True)
    selected: List[WaterFarm] = []
    total_val = 0
    total_dem = 0
    for f in sorted_farms:
        if total_dem + f.demand <= capacity:
            candidate = selected + [f]
            if check_network and network is not None:
                if not _check_network_feasible(candidate, network):
                    continue
            selected.append(f)
            total_val += f.value
            total_dem += f.demand
    return selected, total_val


# ── Branch & Bound ────────────────────────────────────────────────────────────

def _branch_and_bound(
    network: WaterNetwork,
    capacity: int,
    bound_mode: str,
) -> List[Dict[str, Any]]:
    """
    Branch & Bound with frame emission.  Returns ordered frame list.
    """
    farms = sorted(network.farms, key=lambda f: f.value / max(f.demand, 1), reverse=True)
    n = len(farms)

    # Warm-start incumbent from greedy
    greedy_sel, greedy_val = _greedy_heuristic(
        farms, capacity,
        network=network if bound_mode == "network" else None,
        check_network=(bound_mode == "network"),
    )
    state = {
        "incumbent": greedy_sel,
        "incumbent_value": greedy_val,
        "node_counter": 0,
        "explored": 0,
        "pruned": 0,
        "timed_out": False,
    }
    tree_nodes: List[Dict[str, Any]] = []
    frames: List[Dict[str, Any]] = []
    deadline = time.monotonic() + EXACT_TIMEOUT_SEC

    def _snap_tree() -> List[Dict[str, Any]]:
        return [dict(nd) for nd in tree_nodes]

    def _make_frame(action: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action": action,
            "metrics": {
                **metrics,
                "explored": state["explored"],
                "pruned": state["pruned"],
                "incumbent": state["incumbent_value"],
                "incumbentFarms": len(state["incumbent"]),
            },
            "state": {
                "tree": _snap_tree(),
                "selectedFarms": [f.id for f in state["incumbent"]],
                "phase": "branch-and-bound",
                "isTerminal": False,
                "capacity": capacity,
                "boundMode": bound_mode,
            },
        }

    def explore(
        depth: int,
        included: List[WaterFarm],
        current_value: int,
        current_demand: int,
        parent_id: Optional[str],
    ) -> None:
        if state["timed_out"] or time.monotonic() > deadline:
            state["timed_out"] = True
            return

        nid = f"n{state['node_counter']}"
        state["node_counter"] += 1
        state["explored"] += 1

        # Compute fractional bound (always fast)
        bound = _frac_bound(farms, depth, current_demand, current_value, capacity)

        node_rec: Dict[str, Any] = {
            "id": nid,
            "parent": parent_id,
            "depth": depth,
            "included": [f.id for f in included],
            "bound": round(bound, 2),
            "value": current_value,
            "demand": current_demand,
            "pruned": False,
            "feasible": None,
        }
        tree_nodes.append(node_rec)

        inc_str = ", ".join(f.id for f in included) or "∅"
        frames.append(_make_frame(
            f"Node {nid} (depth {depth}): {{{inc_str}}}, val={current_value}, bound={bound:.1f}",
            {"bound": bound, "depth": depth, "value": current_value, "demand": current_demand},
        ))

        # Prune: bound cannot beat incumbent
        if bound <= state["incumbent_value"]:
            node_rec["pruned"] = True
            state["pruned"] += 1
            frames.append(_make_frame(
                f"Prune {nid}: fractional bound {bound:.1f} ≤ incumbent {state['incumbent_value']} — no improvement possible",
                {"reason": "bound", "bound": bound, "incumbent": state["incumbent_value"]},
            ))
            return

        # Prune: demand overflow
        if current_demand > capacity:
            node_rec["pruned"] = True
            state["pruned"] += 1
            frames.append(_make_frame(
                f"Prune {nid}: demand {current_demand} > capacity {capacity}",
                {"reason": "infeasible_demand"},
            ))
            return

        # Leaf node
        if depth == n:
            if bound_mode == "network":
                feasible = _check_network_feasible(included, network)
                node_rec["feasible"] = feasible
                if not feasible:
                    node_rec["pruned"] = True
                    state["pruned"] += 1
                    frames.append(_make_frame(
                        f"Node {nid} fails network feasibility — flow cannot route all selected demands",
                        {"reason": "network_infeasible", "demand": current_demand},
                    ))
                    return

            if current_value > state["incumbent_value"]:
                state["incumbent"] = list(included)
                state["incumbent_value"] = current_value
                feasibility_note = (
                    " (network-feasible, exact for this size)"
                    if bound_mode == "network"
                    else " (single-capacity bound — network routing not verified)"
                )
                frames.append(_make_frame(
                    f"★ New incumbent: value={current_value}, farms={{{inc_str}}}{feasibility_note}",
                    {"new_incumbent": True, "value": current_value},
                ))
            return

        farm = farms[depth]

        # Branch 1: include farm
        if current_demand + farm.demand <= capacity:
            explore(
                depth + 1,
                included + [farm],
                current_value + farm.value,
                current_demand + farm.demand,
                nid,
            )

        # Branch 2: exclude farm
        explore(depth + 1, included, current_value, current_demand, nid)

    # ── Opening frame ─────────────────────────────────────────────────────────
    frames.append({
        "action": (
            f"Branch & Bound start — {n} farms, capacity={capacity}, mode={bound_mode}. "
            f"Greedy warm-start incumbent: value={greedy_val}."
        ),
        "metrics": {"farms": n, "capacity": capacity, "greedy_incumbent": greedy_val},
        "state": {
            "tree": [],
            "selectedFarms": [f.id for f in greedy_sel],
            "phase": "branch-and-bound",
            "isTerminal": False,
            "capacity": capacity,
            "boundMode": bound_mode,
        },
    })

    explore(0, [], 0, 0, None)

    # ── Terminal frame ────────────────────────────────────────────────────────
    incumbent_ids = [f.id for f in state["incumbent"]]
    timeout_note = " [TIMEOUT — returned best found so far]" if state["timed_out"] else ""
    quality = "approximate" if state["timed_out"] else "exact"
    feasibility_claim = (
        "network-feasible, exact for realistic sizes"
        if bound_mode == "network" and not state["timed_out"]
        else "single-capacity bound" if bound_mode == "fractional"
        else "approximate (timeout)"
    )

    frames.append({
        "action": (
            f"B&B complete{timeout_note} — optimal value={state['incumbent_value']}, "
            f"selected={{{', '.join(incumbent_ids) or '∅'}}}, "
            f"explored={state['explored']}, pruned={state['pruned']}. "
            f"Result: {feasibility_claim}."
        ),
        "metrics": {
            "optimal_value": state["incumbent_value"],
            "selected_count": len(state["incumbent"]),
            "explored": state["explored"],
            "pruned": state["pruned"],
            "timed_out": state["timed_out"],
        },
        "state": {
            "tree": _snap_tree(),
            "selectedFarms": incumbent_ids,
            "phase": "result",
            "isTerminal": True,
            "isAccepted": True,
            "capacity": capacity,
            "boundMode": bound_mode,
            "optimalValue": state["incumbent_value"],
            "quality": quality,
        },
    })

    return frames


# ── FPTAS fallback ────────────────────────────────────────────────────────────

def _fptas_solve(
    network: WaterNetwork,
    capacity: int,
    epsilon: float = 0.001,
) -> List[Dict[str, Any]]:
    """
    FPTAS for 0/1 knapsack (farms > FPTAS_THRESHOLD).
    Scales values, runs DP, traces back selected set.
    Result is within (1 - epsilon) of optimal.
    """
    farms = network.farms
    n = len(farms)
    max_val = max((f.value for f in farms), default=1)
    scale = n / (epsilon * max(max_val, 1))

    scaled = [max(1, int(f.value * scale)) for f in farms]

    # 0/1 knapsack DP (capacity as knapsack size)
    dp: List[int] = [0] * (capacity + 1)
    # Track choices for traceback
    keep: List[List[bool]] = [[False] * (capacity + 1) for _ in range(n)]

    for i, farm in enumerate(farms):
        d = farm.demand
        for c in range(capacity, d - 1, -1):
            if dp[c - d] + scaled[i] > dp[c]:
                dp[c] = dp[c - d] + scaled[i]
                keep[i][c] = True

    # Traceback
    selected: List[WaterFarm] = []
    c = capacity
    for i in range(n - 1, -1, -1):
        if keep[i][c]:
            selected.append(farms[i])
            c -= farms[i].demand

    total_val = sum(f.value for f in selected)

    return [
        {
            "action": (
                f"FPTAS fallback — {n} farms exceeds exact-B&B threshold ({FPTAS_THRESHOLD}). "
                f"ε={epsilon}, scaling factor={scale:.1f}."
            ),
            "metrics": {"farms": n, "capacity": capacity, "epsilon": epsilon},
            "state": {
                "tree": [],
                "selectedFarms": [],
                "phase": "fptas",
                "isTerminal": False,
                "capacity": capacity,
                "boundMode": "fptas",
            },
        },
        {
            "action": (
                f"FPTAS result (≤{epsilon * 100:.1f}% error): value={total_val}, "
                f"selected {len(selected)}/{n} farms. "
                f"This is approximate, not exact — stage-wise optimal ≠ globally optimal."
            ),
            "metrics": {
                "approximate_value": total_val,
                "selected_count": len(selected),
                "epsilon": epsilon,
            },
            "state": {
                "tree": [],
                "selectedFarms": [f.id for f in selected],
                "phase": "result",
                "isTerminal": True,
                "isAccepted": True,
                "quality": "approximate",
                "epsilon": epsilon,
                "optimalValue": total_val,
                "boundMode": "fptas",
                "capacity": capacity,
            },
        },
    ]


# ── Public API ────────────────────────────────────────────────────────────────

class FarmSelectionEngine:

    @staticmethod
    def solve(data: Dict[str, Any], detail: str = "full") -> List[Dict[str, Any]]:
        """
        Phase 2 solve: B&B farm selection.

        Accepts WaterNetwork JSON plus optional:
            bound_mode: 'fractional' | 'network'
            max_flow:   int  (Phase 1 result; computed internally if omitted)

        detail='full'   → full B&B tree step movie (default, backward-compatible)
        detail='result' → terminal frame only
        """
        network = parse_water_network(data)
        bound_mode = str(data.get("bound_mode", "fractional"))

        # Drop farms whose node has no pipe path from the source — they can
        # never receive water, so they must not enter the selection problem.
        reachable = _reachable_from_source(network)
        excluded = [f for f in network.farms if f.nodeId not in reachable]
        if excluded:
            network.farms = [f for f in network.farms if f.nodeId in reachable]

        # Determine capacity
        max_flow = data.get("max_flow")
        if max_flow is None:
            fi = water_to_flow_inputs(network)
            max_flow, _, _ = _run_max_flow(
                fi["nodes"], fi["edges"], fi["source"], fi["sink"]
            )
        capacity = int(max_flow)

        if len(network.farms) > FPTAS_THRESHOLD:
            steps = _fptas_solve(network, capacity)
        else:
            steps = _branch_and_bound(network, capacity, bound_mode)

        # Prepend a warning frame when unreachable farms were dropped
        if excluded:
            ids = ", ".join(f.id for f in excluded)
            warning_frame: Dict[str, Any] = {
                "action": (
                    f"⚠ Excluded {len(excluded)} farm(s) with no pipe path from source: {{{ids}}}. "
                    "Water cannot reach their nodes — they are ineligible for selection."
                ),
                "metrics": {"excluded_farms": len(excluded), "excluded_ids": [f.id for f in excluded]},
                "state": {
                    "tree": [], "selectedFarms": [], "phase": "branch-and-bound",
                    "isTerminal": False, "capacity": capacity, "boundMode": bound_mode,
                },
            }
            steps = [warning_frame] + steps

        return steps if detail == "full" else [steps[-1]]

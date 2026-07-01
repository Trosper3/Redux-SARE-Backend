"""
water_scheduling_engine.py — Phase 3 engine: NP-hard scheduling modules.

Input: WaterNetwork JSON + selected_farms list (from Phase 2) + mode.

mode:
  coloring    — conflict focus: no two neighboring farms in the same time slot.
                Welsh-Powell greedy, then exact chromatic number via backtracking.
  bin-packing — quota focus: pack selected farms into daily extraction-limit bins.
                FFD greedy, then exact BnB for small n.
  job-shop    — time focus: assign heterogeneous durationHrs to non-overlapping windows.
                Greedy earliest-start (heuristic only at this phase).
  mis         — dispatch focus: maximum-weight independent set — the largest
                non-conflicting high-value farm set runnable right now.
                Greedy by value, then exact BnB for small n.

Frame convention:
  Greedy frames come first (quality='heuristic'); exact frames follow (quality='exact').
  The last frame has isTerminal=True. The first isTerminal=True frame with
  quality='heuristic' can be used by the frontend for instant display.

Honesty notes embedded in frame 'action' strings:
  - Greedy results: "Heuristic — …"
  - Exact results carry explicit size/timeout qualifications.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

from .water_model import WaterFarm, WaterNetwork, parse_water_network, water_to_flow_inputs
from .water_network_engine import _run_max_flow
from .farm_selection_engine import _reachable_from_source

EXACT_TIMEOUT_SEC = 5.0

COLOR_PALETTE = [
    "#e53935", "#8e24aa", "#1e88e5", "#43a047",
    "#fb8c00", "#00acc1", "#fdd835", "#6d4c41",
    "#00897b", "#c0ca33",
]


def _get_farms(network: WaterNetwork, selected_ids: List[str]) -> List[WaterFarm]:
    if not selected_ids:
        return list(network.farms)
    id_set = set(selected_ids)
    return [f for f in network.farms if f.id in id_set]


def _build_adj(farms: List[WaterFarm]) -> Dict[str, Set[str]]:
    farm_ids = {f.id for f in farms}
    adj: Dict[str, Set[str]] = {f.id: set() for f in farms}
    for f in farms:
        for nbr in f.neighbors:
            if nbr in farm_ids:
                adj[f.id].add(nbr)
                adj[nbr].add(f.id)
    return adj


# ── Coloring mode ──────────────────────────────────────────────────────────────

def _solve_coloring(farms: List[WaterFarm]) -> List[Dict[str, Any]]:
    adj = _build_adj(farms)
    frames: List[Dict[str, Any]] = []
    coloring: Dict[str, int] = {}

    def _snap(
        action: str,
        terminal: bool = False,
        quality: str = "heuristic",
        col_override: Optional[Dict[str, int]] = None,
    ) -> None:
        c = col_override if col_override is not None else dict(coloring)
        slot_count = max(c.values()) + 1 if c else 0
        frames.append({
            "action": action,
            "metrics": {
                "farmCount": len(farms), "assignedCount": len(c),
                "slotCount": slot_count, "quality": quality, "mode": "coloring",
            },
            "state": {
                "phase": "scheduling", "mode": "coloring", "quality": quality,
                "nodeColoring": c, "colorPalette": COLOR_PALETTE,
                "selectedFarms": [f.id for f in farms],
                "isTerminal": terminal, "isAccepted": terminal,
            },
        })

    # Welsh-Powell: sort by degree descending
    order = sorted(farms, key=lambda f: -len(adj[f.id]))
    _snap(
        "Greedy slot assignment — Welsh-Powell order (highest-degree first): "
        + ", ".join(f.id for f in order)
    )

    for f in order:
        used = {coloring[nbr] for nbr in adj[f.id] if nbr in coloring}
        slot = 0
        while slot in used:
            slot += 1
        coloring[f.id] = slot
        conflict_note = f" (neighbor slots: {sorted(used)})" if used else " (no conflicts)"
        _snap(f"Assign {f.id} → time-slot {slot}{conflict_note}")

    greedy_k = max(coloring.values()) + 1 if coloring else 0
    greedy_coloring = dict(coloring)
    _snap(
        f"Greedy coloring: {greedy_k} slot(s). "
        "Heuristic — chromatic number may be lower.",
        terminal=False, quality="heuristic",
    )

    # Exact: search for chromatic number < greedy_k via backtracking
    exact_coloring = greedy_coloring
    exact_k = greedy_k

    if len(farms) <= 15 and greedy_k > 1:
        deadline = time.monotonic() + EXACT_TIMEOUT_SEC
        for k in range(1, greedy_k):
            if time.monotonic() > deadline:
                break
            trial: Dict[str, int] = {}

            def bt(idx: int, k_local: int = k) -> bool:
                if idx >= len(order):
                    return True
                fid = order[idx].id
                used_colors = {trial[nbr] for nbr in adj[fid] if nbr in trial}
                for c in range(k_local):
                    if c not in used_colors:
                        trial[fid] = c
                        if bt(idx + 1, k_local):
                            return True
                        del trial[fid]
                return False

            if bt(0):
                exact_coloring = dict(trial)
                exact_k = k
                break

    _snap(
        f"Exact coloring: {exact_k} slot(s) — chromatic number ≤ {exact_k}. "
        "No two neighboring farms share a slot.",
        terminal=True, quality="exact", col_override=exact_coloring,
    )
    return frames


# ── Bin-packing mode ──────────────────────────────────────────────────────────

def _solve_bin_packing(farms: List[WaterFarm], daily_quota: int) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    bins: List[Dict[str, Any]] = []

    def _snap(
        action: str,
        active_farm: Optional[str] = None,
        active_bin: Optional[int] = None,
        terminal: bool = False,
        quality: str = "heuristic",
        bins_override: Optional[List[Dict]] = None,
    ) -> None:
        b = bins_override if bins_override is not None else deepcopy(bins)
        frames.append({
            "action": action,
            "metrics": {
                "farmCount": len(farms), "binCount": len(b),
                "dailyQuota": daily_quota, "quality": quality, "mode": "bin-packing",
            },
            "state": {
                "phase": "scheduling", "mode": "bin-packing", "quality": quality,
                "bins": deepcopy(b), "dailyQuota": daily_quota,
                "activeFarmId": active_farm, "activeBinIndex": active_bin,
                "selectedFarms": [f.id for f in farms],
                "isTerminal": terminal, "isAccepted": terminal,
            },
        })

    sorted_farms = sorted(farms, key=lambda f: -f.demand)
    _snap(
        f"Greedy FFD bin-packing — {len(farms)} farm(s), daily quota={daily_quota}. "
        f"Order (largest demand first): {', '.join(f.id for f in sorted_farms)}"
    )

    for f in sorted_farms:
        placed = False
        for i, b in enumerate(bins):
            if b["usedCapacity"] + f.demand <= daily_quota:
                b["farms"].append({"id": f.id, "demand": f.demand})
                b["usedCapacity"] += f.demand
                b["remaining"] = daily_quota - b["usedCapacity"]
                placed = True
                _snap(f"Place {f.id} (demand={f.demand}) → day {i}", active_farm=f.id, active_bin=i)
                break
        if not placed:
            new_idx = len(bins)
            bins.append({
                "index": new_idx,
                "farms": [{"id": f.id, "demand": f.demand}],
                "usedCapacity": f.demand,
                "remaining": daily_quota - f.demand,
            })
            _snap(f"Open day {new_idx} for {f.id} (demand={f.demand})", active_farm=f.id, active_bin=new_idx)

    greedy_k = len(bins)
    greedy_bins = deepcopy(bins)
    _snap(
        f"FFD greedy: {greedy_k} day(s). "
        "Heuristic — exact BnB may find fewer.",
        terminal=False, quality="heuristic",
    )

    # Exact BnB for small n
    best_bins: List[Dict[str, Any]] = deepcopy(greedy_bins)
    best_k = greedy_k

    if len(farms) <= 20:
        deadline = time.monotonic() + EXACT_TIMEOUT_SEC
        trial_bins: List[Dict[str, Any]] = []

        def bnb(idx: int) -> None:
            nonlocal best_bins, best_k
            if time.monotonic() > deadline:
                return
            if idx >= len(sorted_farms):
                if len(trial_bins) < best_k:
                    best_k = len(trial_bins)
                    best_bins = [
                        {
                            "index": i,
                            "farms": deepcopy(b["farms"]),
                            "usedCapacity": b["usedCapacity"],
                            "remaining": b["remaining"],
                        }
                        for i, b in enumerate(trial_bins)
                    ]
                return

            f = sorted_farms[idx]
            seen_caps: Set[int] = set()
            for b in trial_bins:
                # Symmetry breaking: skip bins with identical remaining capacity
                cap = b["usedCapacity"]
                if cap in seen_caps:
                    continue
                seen_caps.add(cap)
                if b["usedCapacity"] + f.demand <= daily_quota:
                    b["farms"].append({"id": f.id, "demand": f.demand})
                    old = b["usedCapacity"]
                    b["usedCapacity"] += f.demand
                    b["remaining"] = daily_quota - b["usedCapacity"]
                    bnb(idx + 1)
                    b["farms"].pop()
                    b["usedCapacity"] = old
                    b["remaining"] = daily_quota - b["usedCapacity"]

            # Open new bin only if it can still beat best_k
            if len(trial_bins) + 1 < best_k:
                trial_bins.append({
                    "index": len(trial_bins),
                    "farms": [{"id": f.id, "demand": f.demand}],
                    "usedCapacity": f.demand,
                    "remaining": daily_quota - f.demand,
                })
                bnb(idx + 1)
                trial_bins.pop()

        bnb(0)

    _snap(
        f"Exact bin-packing: {best_k} day(s) — no day exceeds quota {daily_quota}.",
        terminal=True, quality="exact", bins_override=best_bins,
    )
    return frames


# ── Job-shop mode ─────────────────────────────────────────────────────────────

def _solve_job_shop(farms: List[WaterFarm], horizon: int) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    schedule: List[Dict[str, Any]] = []

    def _snap(
        action: str,
        terminal: bool = False,
        quality: str = "heuristic",
        active_farm: Optional[str] = None,
    ) -> None:
        frames.append({
            "action": action,
            "metrics": {
                "farmCount": len(farms), "scheduledCount": len(schedule),
                "horizon": horizon, "quality": quality, "mode": "job-shop",
            },
            "state": {
                "phase": "scheduling", "mode": "job-shop", "quality": quality,
                "schedule": deepcopy(schedule), "horizon": horizon,
                "activeFarmId": active_farm,
                "selectedFarms": [f.id for f in farms],
                "isTerminal": terminal, "isAccepted": terminal,
            },
        })

    # Longest-job-first greedy (reduces makespan)
    sorted_farms = sorted(farms, key=lambda f: -f.durationHrs)
    slot_ends: List[float] = []  # end time per parallel slot

    _snap(
        f"Greedy job-shop (longest-first) — {len(farms)} farm(s), horizon={horizon}h. "
        f"Order: {', '.join(f.id for f in sorted_farms)}"
    )

    for f in sorted_farms:
        if not slot_ends:
            slot = 0
            start = 0.0
        else:
            best_slot = min(range(len(slot_ends)), key=lambda i: slot_ends[i])
            if slot_ends[best_slot] + f.durationHrs <= horizon:
                slot = best_slot
                start = slot_ends[best_slot]
            else:
                slot = len(slot_ends)
                start = 0.0

        end = start + f.durationHrs
        while len(slot_ends) <= slot:
            slot_ends.append(0.0)
        slot_ends[slot] = end

        schedule.append({"farmId": f.id, "startHr": start, "endHr": end, "slot": slot})
        _snap(
            f"Schedule {f.id} ({f.durationHrs}h): slot {slot}, "
            f"hours {start:.1f}–{end:.1f}",
            active_farm=f.id,
        )

    makespan = max(slot_ends, default=0.0)
    slots_used = len(slot_ends)
    _snap(
        f"Job-shop schedule: {slots_used} parallel slot(s), makespan={makespan:.1f}h. "
        "Heuristic — optimal may use fewer slots or a shorter makespan.",
        terminal=True, quality="heuristic",
    )
    return frames


# ── MIS mode ──────────────────────────────────────────────────────────────────

def _solve_mis(farms: List[WaterFarm]) -> List[Dict[str, Any]]:
    adj = _build_adj(farms)
    frames: List[Dict[str, Any]] = []

    # nodeColoring convention: 0=selected (green), -1=excluded (dark), 2=active (amber)
    def _snap(
        action: str,
        selected: List[str],
        terminal: bool = False,
        quality: str = "heuristic",
        active: Optional[str] = None,
    ) -> None:
        coloring = {f.id: (0 if f.id in selected else -1) for f in farms}
        if active:
            coloring[active] = 2
        val = sum(f.value for f in farms if f.id in selected)
        frames.append({
            "action": action,
            "metrics": {
                "farmCount": len(farms), "selectedCount": len(selected),
                "selectedValue": val, "quality": quality, "mode": "mis",
            },
            "state": {
                "phase": "scheduling", "mode": "mis", "quality": quality,
                "nodeColoring": coloring, "colorPalette": COLOR_PALETTE,
                "selectedFarms": selected,
                "isTerminal": terminal, "isAccepted": terminal,
            },
        })

    sorted_farms = sorted(farms, key=lambda f: -f.value)
    _snap(
        "Greedy MIS — sort by value (highest first): "
        + ", ".join(f.id for f in sorted_farms),
        [],
    )

    greedy_sel: List[str] = []
    greedy_set: Set[str] = set()

    for f in sorted_farms:
        conflict = adj[f.id].intersection(greedy_set)
        if not conflict:
            greedy_sel.append(f.id)
            greedy_set.add(f.id)
            _snap(f"Include {f.id} (value={f.value}, no conflicts)", greedy_sel, active=f.id)
        else:
            _snap(
                f"Skip {f.id} (value={f.value}, conflicts: {sorted(conflict)})",
                greedy_sel, active=f.id,
            )

    greedy_val = sum(f.value for f in farms if f.id in greedy_set)
    _snap(
        f"Greedy MIS: {len(greedy_sel)} farm(s), value={greedy_val}. "
        "Heuristic — optimal may have higher value.",
        greedy_sel, terminal=False, quality="heuristic",
    )

    # Exact: weighted MIS via B&B for small n
    best_sel = list(greedy_sel)
    best_val = greedy_val

    if len(farms) <= 15:
        deadline = time.monotonic() + EXACT_TIMEOUT_SEC
        trial: List[str] = []
        trial_set: Set[str] = set()
        order = sorted(farms, key=lambda f: -f.value)

        def bt_mis(idx: int, current_val: int) -> None:
            nonlocal best_sel, best_val
            if time.monotonic() > deadline:
                return
            if idx >= len(order):
                if current_val > best_val:
                    best_val = current_val
                    best_sel = list(trial)
                return
            # Upper bound: current + all remaining values (ignoring conflicts)
            ub = current_val + sum(order[i].value for i in range(idx, len(order)))
            if ub <= best_val:
                return  # prune

            f = order[idx]
            if not adj[f.id].intersection(trial_set):
                trial.append(f.id)
                trial_set.add(f.id)
                bt_mis(idx + 1, current_val + f.value)
                trial.pop()
                trial_set.discard(f.id)

            bt_mis(idx + 1, current_val)  # exclude

        bt_mis(0, 0)

    exact_val = sum(f.value for f in farms if f.id in best_sel)
    _snap(
        f"Exact MIS: {len(best_sel)} farm(s), value={exact_val}. "
        "Maximum non-conflicting set — runnable simultaneously right now.",
        best_sel, terminal=True, quality="exact",
    )
    return frames


# ── Public API ────────────────────────────────────────────────────────────────

class WaterSchedulingEngine:

    @staticmethod
    def solve(data: Dict[str, Any], detail: str = "full") -> List[Dict[str, Any]]:
        """
        Phase 3 solve: scheduling module dispatcher.

        Accepts WaterNetwork JSON plus optional:
            selected_farms: list[str]  farm IDs from Phase 2 (all farms if omitted)
            mode:           str        'coloring' | 'bin-packing' | 'job-shop' | 'mis'
            daily_quota:    int        bin capacity (bin-packing); auto from Phase 1 if omitted
            horizon_hrs:    int        planning horizon in hours (job-shop; default 24)

        detail='full'   → full step movie (default, backward-compatible)
        detail='result' → terminal frame only
        """
        network = parse_water_network(data)
        mode = str(data.get("mode", "coloring"))
        selected_ids: List[str] = data.get("selected_farms", [])
        farms = _get_farms(network, selected_ids)

        # Guard: drop any farm whose node has no pipe path from the source.
        # Phase 2 already filters these when it runs, but Phase 3 may be called
        # with Phase 2 disabled (frontend fallback: all farms → scheduling).
        reachable = _reachable_from_source(network)
        farms = [f for f in farms if f.nodeId in reachable]

        if not farms:
            return [{
                "action": "No farms to schedule.",
                "metrics": {"farmCount": 0, "mode": mode},
                "state": {
                    "phase": "scheduling", "mode": mode, "selectedFarms": [],
                    "isTerminal": True, "isAccepted": False,
                },
            }]

        if mode == "coloring":
            steps = _solve_coloring(farms)
        elif mode == "bin-packing":
            daily_quota = data.get("daily_quota")
            if daily_quota is None:
                fi = water_to_flow_inputs(network)
                mf, _, _ = _run_max_flow(fi["nodes"], fi["edges"], fi["source"], fi["sink"])
                daily_quota = max(
                    int(mf),
                    sum(f.demand for f in farms) // max(len(farms) // 2, 1),
                )
            steps = _solve_bin_packing(farms, int(daily_quota))
        elif mode == "job-shop":
            horizon = int(data.get("horizon_hrs", 24))
            steps = _solve_job_shop(farms, horizon)
        elif mode == "mis":
            steps = _solve_mis(farms)
        else:
            steps = [{
                "action": f"Unknown scheduling mode: {mode}",
                "metrics": {"mode": mode},
                "state": {
                    "phase": "scheduling", "mode": mode, "selectedFarms": [],
                    "isTerminal": True, "isAccepted": False,
                },
            }]

        return steps if detail == "full" else [steps[-1]]

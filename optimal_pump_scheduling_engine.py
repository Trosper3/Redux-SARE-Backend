"""
Optimal Pump Scheduling Engine — DAG Shortest/Longest Path
===========================================================
Models the 24-hour pump scheduling problem as a Directed Acyclic Graph (DAG)
and finds the optimal schedule using Dynamic Programming.

DAG structure
-------------
  Node   : (hour, volume_bucket, prev_pump_mask)
           volume_bucket = discretised tank level
           prev_pump_mask = pump combination active in the PREVIOUS hour
                            (needed to compute per-transition startup costs)
  Edge   : applying a pump bitmask for one hour
  Weight : energy cost + startup penalties for pumps that just turned ON

Modes
-----
  cost_minimization    — Shortest path: minimise total cost over 24 h
  emergency_resilience — Constrained longest path: maximise cumulative stored
                         water subject to total cost ≤ budget_limit

API contract
------------
Input  (data dict, from payload.inputs):
  pumps               : str   — newline-separated "name,flowGpm,powerKw,startupCost"
  demand_curve_profile: str   — 24 comma-separated hourly demand values (gallons)
  peak_tariff         : float — $/kWh during peak hours
  off_peak_tariff     : float — $/kWh during off-peak hours
  tank_capacity       : float — gallons
  initial_tank_pct    : float — starting tank level as a percentage (0–100)
  tank_step_gal       : float — volume discretisation step (gallons per bucket)
  T                   : int   — horizon length (always 24)
  budget_limit        : float — (emergency_resilience only) max allowed cost

Output (list with one frame):
  [{ "action": str,
     "metrics": { "energy_cost_total": float,
                  "pump_schedule_matrix": list[list[int]] },
     "state": {} }]
"""

from __future__ import annotations

PEAK_WINDOWS: list[tuple[int, int]] = [(7, 10), (17, 21)]
MIN_TANK_FRACTION = 0.20   # default lower bound — 20 % of capacity
DEFAULT_BUDGET_SLACK = 1.5  # emergency mode: budget = min_cost × slack



def _parse_pumps(raw: str) -> list[dict]:
    pumps = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                pumps.append({
                    "name":         parts[0],
                    "flow_gph":     float(parts[1]) * 60.0,  # GPM → gal/h
                    "power_kw":     float(parts[2]),
                    "startup_cost": float(parts[3]),
                })
            except ValueError:
                pass
    return pumps


def _parse_demand(raw: str) -> list[float]:
    """Accept comma-separated or newline-separated demand values."""
    tokens = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    result = []
    for t in tokens:
        try:
            result.append(float(t))
        except ValueError:
            pass
    return result


def _parse_peak_hours(text: str) -> list[tuple[int, int]]:
    """Parse '7-10,17-21' or '7,8,9,17-21' into (start, end_exclusive) tuples."""
    windows: list[tuple[int, int]] = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start = max(0, min(23, int(parts[0].strip())))
                end   = max(0, min(24, int(parts[1].strip()) + 1))
                if start < end:
                    windows.append((start, end))
            except ValueError:
                pass
        else:
            try:
                h = max(0, min(23, int(token)))
                windows.append((h, h + 1))
            except ValueError:
                pass
    return windows or PEAK_WINDOWS


def _build_dag_and_solve(
    pumps:         list[dict],
    demand:        list[float],
    tariffs:       list[float],
    tank_capacity: float,
    tank_initial:  float,
    tank_min:      float,
    tank_step:     float,
    T:             int,
    mode:          str,
    budget_limit:  float | None,
) -> tuple[list[list[int]], float]:
    """
    DAG DP solver.

    State per timestep: (volume_idx, prev_mask)
      - volume_idx  : discretised tank level after applying the hour's action
      - prev_mask   : the pump bitmask used in that hour
                      (carried forward so the NEXT hour can charge startup costs)

    For cost_minimization  → dp holds minimum cumulative cost.
    For emergency_resilience → dp holds maximum cumulative water score
                               subject to cost ≤ budget_limit.

    Returns (schedule_matrix [n_pumps × T], total_energy_cost).
    """
    n = len(pumps)
    n_masks = 1 << n
    n_vol = int(tank_capacity / tank_step) + 1

    def lvl2idx(v: float) -> int:
        return min(n_vol - 1, max(0, round(v / tank_step)))

    # Precompute per-mask aggregates
    mask_flow  = [0.0] * n_masks
    mask_power = [0.0] * n_masks
    for mask in range(n_masks):
        for p in range(n):
            if mask & (1 << p):
                mask_flow[mask]  += pumps[p]["flow_gph"]
                mask_power[mask] += pumps[p]["power_kw"]

    def startup_cost(prev_mask: int, new_mask: int) -> float:
        return sum(
            pumps[p]["startup_cost"]
            for p in range(n)
            if (new_mask & (1 << p)) and not (prev_mask & (1 << p))
        )

    INF = float("inf")

    # ── State arrays ────────────────────────────────────────────────────────
    # dp_cost[vol_idx][mask]  = min cumulative cost to reach this state
    # dp_score[vol_idx][mask] = max cumulative volume score along the same path
    #                           (tracked in both modes for reporting)
    dp_cost  = [[INF] * n_masks for _ in range(n_vol)]
    dp_score = [[-INF] * n_masks for _ in range(n_vol)]

    # Before hour 0 the system is idle → prev_mask = 0
    init_idx = lvl2idx(tank_initial)
    dp_cost[init_idx][0]  = 0.0
    dp_score[init_idx][0] = 0.0

    # backtrack[h][(vol_idx, mask)] = (prev_vol_idx, prev_mask)
    backtrack: list[dict] = [{} for _ in range(T)]

    # ── Forward DP ──────────────────────────────────────────────────────────
    for h in range(T):
        new_cost  = [[INF]  * n_masks for _ in range(n_vol)]
        new_score = [[-INF] * n_masks for _ in range(n_vol)]
        bt = backtrack[h]

        for vol_idx in range(n_vol):
            for prev_mask in range(n_masks):
                cur_cost = dp_cost[vol_idx][prev_mask]
                if cur_cost == INF:
                    continue
                cur_score = dp_score[vol_idx][prev_mask]
                level = vol_idx * tank_step

                for new_mask in range(n_masks):
                    next_level = level - demand[h] + mask_flow[new_mask]

                    # Hard physics: tank must stay within bounds
                    if next_level < tank_min or next_level > tank_capacity:
                        continue

                    edge_cost = (
                        mask_power[new_mask] * tariffs[h]
                        + startup_cost(prev_mask, new_mask)
                    )
                    edge_score = next_level  # yield = volume stored after this hour

                    total_cost  = cur_cost  + edge_cost
                    total_score = cur_score + edge_score

                    nidx = lvl2idx(next_level)
                    key  = (nidx, new_mask)

                    if mode == "emergency_resilience" and budget_limit is not None:
                        # Constrained longest path: maximise score within budget
                        if total_cost > budget_limit:
                            continue
                        if total_score > new_score[nidx][new_mask]:
                            new_cost[nidx][new_mask]  = total_cost
                            new_score[nidx][new_mask] = total_score
                            bt[key] = (vol_idx, prev_mask)
                    else:
                        # Shortest path: minimise cost
                        if total_cost < new_cost[nidx][new_mask]:
                            new_cost[nidx][new_mask]  = total_cost
                            new_score[nidx][new_mask] = total_score
                            bt[key] = (vol_idx, prev_mask)

        dp_cost  = new_cost
        dp_score = new_score

    # ── Pick best terminal state ─────────────────────────────────────────────
    best_val  = INF if mode != "emergency_resilience" else -INF
    best_vidx = -1
    best_mask = -1

    for vol_idx in range(n_vol):
        for mask in range(n_masks):
            c = dp_cost[vol_idx][mask]
            s = dp_score[vol_idx][mask]
            if c == INF:
                continue
            if mode == "emergency_resilience":
                if s > best_val:
                    best_val  = s
                    best_vidx = vol_idx
                    best_mask = mask
            else:
                if c < best_val:
                    best_val  = c
                    best_vidx = vol_idx
                    best_mask = mask

    # Fallback: no feasible solution → run all pumps every hour
    if best_vidx == -1:
        schedule = [[1] * T for _ in range(n)]
        energy   = sum(pumps[p]["power_kw"] * tariffs[h]
                       for p in range(n) for h in range(T))
        return schedule, energy

    # ── Backtrack through prev_state table ──────────────────────────────────
    masks: list[int] = []
    vol_idx = best_vidx
    mask    = best_mask
    for h in range(T - 1, -1, -1):
        masks.insert(0, mask)
        prev_vol_idx, prev_mask = backtrack[h][(vol_idx, mask)]
        vol_idx = prev_vol_idx
        mask    = prev_mask

    # ── Build schedule matrix [pump][hour] ───────────────────────────────────
    schedule = [[0] * T for _ in range(n)]
    for h, m in enumerate(masks):
        for p in range(n):
            if m & (1 << p):
                schedule[p][h] = 1

    energy = sum(
        pumps[p]["power_kw"] * tariffs[h]
        for h in range(T) for p in range(n) if schedule[p][h]
    )
    return schedule, energy


class OptimalPumpSchedulingSolver:
    @staticmethod
    def solve(data: dict, mode: str | None = None) -> list[dict]:
        mode = (mode or "cost_minimization").strip().lower()

        # ── Parse pumps (accepts key "pump_lines" or "pumps") ────────────────
        pump_raw = str(data.get("pump_lines") or data.get("pumps") or "")
        pumps = _parse_pumps(pump_raw)
        if not pumps:
            return [{
                "action":  "No pumps defined",
                "metrics": {"energy_cost_total": 0.0, "pump_schedule_matrix": []},
                "state":   {},
            }]

        # ── Parse demand — accepts comma- or newline-separated values ────────
        demand_raw = str(data.get("demand_curve_profile", ""))
        demand = _parse_demand(demand_raw)
        T = int(data.get("T", 24))
        if len(demand) < T:
            last = demand[-1] if demand else 10000.0
            demand += [last] * (T - len(demand))
        demand = demand[:T]

        # ── Tariffs — honour custom peak_hours if provided ───────────────────
        peak_tariff     = float(data.get("peak_tariff", 0.28))
        off_peak_tariff = float(data.get("off_peak_tariff", 0.12))
        peak_hours_raw  = str(data.get("peak_hours", "")).strip()
        peak_windows    = _parse_peak_hours(peak_hours_raw) if peak_hours_raw else PEAK_WINDOWS
        tariffs = [
            peak_tariff if any(s <= h < e for s, e in peak_windows) else off_peak_tariff
            for h in range(T)
        ]

        # ── Tank parameters ──────────────────────────────────────────────────
        tank_capacity   = float(data.get("tank_capacity", 24000.0))
        initial_pct     = float(data.get("initial_tank_pct", 60.0))
        initial_tank_gal = data.get("initial_tank_gal")
        tank_initial    = float(initial_tank_gal) if initial_tank_gal is not None else tank_capacity * initial_pct / 100.0
        # Prefer explicit tank_min_gal from UI; fall back to 20 % of capacity
        tank_min_gal  = data.get("tank_min_gal")
        tank_min      = float(tank_min_gal) if tank_min_gal is not None else tank_capacity * MIN_TANK_FRACTION
        tank_step     = float(data.get("tank_step_gal", max(1.0, tank_capacity / 200.0)))

        # ── Budget for emergency mode (accepts budget_limit or budget_limit_dollars) ──
        budget_limit: float | None = data.get("budget_limit") or data.get("budget_limit_dollars")
        if budget_limit is not None:
            budget_limit = float(budget_limit)

        # For emergency_resilience with no explicit budget: solve cost-min first
        # and use DEFAULT_BUDGET_SLACK × that cost as the budget.
        if mode == "emergency_resilience" and budget_limit is None:
            _, min_cost = _build_dag_and_solve(
                pumps, demand, tariffs, tank_capacity, tank_initial,
                tank_min, tank_step, T, "cost_minimization", None,
            )
            budget_limit = min_cost * DEFAULT_BUDGET_SLACK

        # ── Solve ────────────────────────────────────────────────────────────
        schedule, energy_cost = _build_dag_and_solve(
            pumps, demand, tariffs, tank_capacity, tank_initial,
            tank_min, tank_step, T, mode, budget_limit,
        )

        # Startup costs: sum of per-transition penalties already baked into
        # the DP edge weights.  Here we recompute for accurate reporting.
        n = len(pumps)
        startup_total = 0.0
        for p in range(n):
            prev_on = False
            for h in range(T):
                on = bool(schedule[p][h])
                if on and not prev_on:
                    startup_total += pumps[p]["startup_cost"]
                prev_on = on

        total_cost = energy_cost + startup_total

        # Simulate tank trajectory for the frontend's per-hour display
        n = len(pumps)
        mask_flow = [0.0] * (1 << n)
        for mask in range(1 << n):
            for p in range(n):
                if mask & (1 << p):
                    mask_flow[mask] += pumps[p]["flow_gph"]
        level = tank_initial
        storage_trajectory: list[float] = [round(level, 1)]  # index 0 = initial level before any pumping
        for h in range(T):
            active_mask = sum((1 << p) for p in range(n) if schedule[p][h])
            level = level - demand[h] + mask_flow[active_mask]
            level = max(tank_min, min(tank_capacity, level))
            storage_trajectory.append(round(level, 1))

        action_label = (
            "Pump Schedule Optimized — Cost Minimization"
            if mode != "emergency_resilience"
            else "Pump Schedule Optimized — Emergency Resilience"
        )

        # ── Build per-hour process trace ─────────────────────────────────────
        algo_name = "DAG Shortest Path (DP)" if mode != "emergency_resilience" else "DAG Constrained Longest Path (DP)"
        process_trace: list[dict] = []

        # Setup row
        n_masks = 1 << n
        n_vol   = int(tank_capacity / tank_step) + 1
        process_trace.append({
            "Step":  1,
            "Stage": "Initialization",
            "Algorithm": algo_name,
            "What Happened": (
                f"DAG constructed: {T} time-steps × {n_vol} tank-level buckets × {n_masks} pump combinations "
                f"= {T * n_vol * n_masks:,} nodes. "
                f"Bucket size: {tank_step:.0f} gal. "
                f"Tank starts at {tank_initial:,.0f} gal ({float(data.get('initial_tank_pct', 60)):.0f}% of {tank_capacity:,.0f} gal capacity). "
                f"Min allowed level: {tank_min:,.0f} gal."
            ),
        })

        # Per-hour DP rows
        level_sim = tank_initial
        cumulative_cost = 0.0
        prev_mask_sim = 0
        mask_flow_local = [0.0] * n_masks
        for mask in range(n_masks):
            for p in range(n):
                if mask & (1 << p):
                    mask_flow_local[mask] += pumps[p]["flow_gph"]

        for h in range(T):
            active_mask = sum((1 << p) for p in range(n) if schedule[p][h])
            active_pump_names = [pumps[p]["name"] for p in range(n) if schedule[p][h]]
            is_peak = tariffs[h] > off_peak_tariff
            tariff_label = f"${tariffs[h]:.2f}/kWh ({'peak' if is_peak else 'off-peak'})"

            net_inflow = mask_flow_local[active_mask] - demand[h]
            level_before = level_sim
            level_sim = max(tank_min, min(tank_capacity, level_sim + net_inflow))

            # startup cost this hour
            hour_startup = sum(
                pumps[p]["startup_cost"]
                for p in range(n)
                if (active_mask & (1 << p)) and not (prev_mask_sim & (1 << p))
            )
            hour_energy = sum(pumps[p]["power_kw"] for p in range(n) if active_mask & (1 << p)) * tariffs[h]
            hour_cost = hour_energy + hour_startup
            cumulative_cost += hour_cost
            prev_mask_sim = active_mask

            if active_pump_names:
                pump_str = ", ".join(active_pump_names) + " ON"
            else:
                pump_str = "All pumps OFF"

            startup_note = f" (+${hour_startup:.2f} startup)" if hour_startup > 0 else ""
            what_happened = (
                f"{pump_str}. "
                f"Net flow: {net_inflow:+,.0f} gal/h. "
                f"Tank: {level_before:,.0f} → {level_sim:,.0f} gal. "
                f"Energy cost: ${hour_energy:.2f}{startup_note}. "
                f"Tariff: {tariff_label}. "
                f"Cumulative cost: ${cumulative_cost:.2f}."
            )

            process_trace.append({
                "Step":  h + 2,
                "Stage": f"Hour {h} ({'Peak' if is_peak else 'Off-Peak'})",
                "Algorithm": algo_name,
                "What Happened": what_happened,
                "Reduction Mapping": (
                    f"DAG edge: (hour={h}, tank≈{level_before:,.0f} gal, prev_mask={prev_mask_sim:0{n}b}) "
                    f"→ (hour={h+1}, tank≈{level_sim:,.0f} gal, mask={active_mask:0{n}b})"
                ),
            })

        # Summary row
        process_trace.append({
            "Step":  T + 2,
            "Stage": "Result",
            "Algorithm": algo_name,
            "What Happened": (
                f"Optimal path found via backtracking through {T}-step DAG. "
                f"Total energy cost: ${energy_cost:.2f}. "
                f"Total startup cost: ${startup_total:.2f}. "
                f"Grand total: ${total_cost:.2f}. "
                f"Mode: {mode.replace('_', ' ').title()}."
            ),
        })

        return [{
            "action":  action_label,
            "metrics": {
                "energy_cost_total":           round(total_cost, 2),
                "pump_schedule_matrix":         schedule,
                "simulated_storage_trajectory": storage_trajectory,
            },
            "state": {
                "processTrace": process_trace,
            },
        }]

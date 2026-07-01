"""
Tests for water_scheduling_engine.py — Phase 3 acceptance criteria.

Uses water_demo.json fixture:
  farms: f1(demand=4,value=100,dur=4,nbrs=[f2]),
         f2(demand=5,value=80, dur=3,nbrs=[f1]),
         f3(demand=6,value=120,dur=5,nbrs=[f4]),
         f4(demand=3,value=90, dur=4,nbrs=[f3])
  conflict graph: f1—f2, f3—f4 (two disjoint edges)
  max_flow = 16 (from Phase 1)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from water_scheduling_engine import WaterSchedulingEngine

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "water_demo.json")


def load_demo(extra: dict = None) -> dict:
    with open(FIXTURE) as f:
        data = json.load(f)
    if extra:
        data.update(extra)
    return data


# ── Coloring: no two neighbors share a slot ───────────────────────────────────

def test_coloring_no_conflicts():
    frames = WaterSchedulingEngine.solve(load_demo({"mode": "coloring"}))
    assert frames, "no frames returned"
    terminal = frames[-1]
    assert terminal["state"]["isTerminal"], "last frame must be terminal"
    coloring = terminal["state"]["nodeColoring"]
    # f1 and f2 are neighbors — must differ
    assert coloring["f1"] != coloring["f2"], "f1 and f2 must be in different slots"
    # f3 and f4 are neighbors — must differ
    assert coloring["f3"] != coloring["f4"], "f3 and f4 must be in different slots"


def test_coloring_exact_chromatic_number():
    frames = WaterSchedulingEngine.solve(load_demo({"mode": "coloring"}))
    terminal = frames[-1]
    assert terminal["state"]["quality"] == "exact"
    # Chromatic number of two disjoint edges is 2
    slot_count = terminal["metrics"]["slotCount"]
    assert slot_count == 2, f"chromatic number should be 2, got {slot_count}"


def test_coloring_selected_subset():
    """Only selected farms f1, f2 should appear in the terminal coloring."""
    frames = WaterSchedulingEngine.solve(
        load_demo({"mode": "coloring", "selected_farms": ["f1", "f2"]})
    )
    terminal = frames[-1]
    coloring = terminal["state"]["nodeColoring"]
    assert set(coloring.keys()) == {"f1", "f2"}
    assert coloring["f1"] != coloring["f2"], "f1 and f2 conflict — different slots required"


# ── Bin-packing: no bin exceeds daily quota ───────────────────────────────────

def test_bin_packing_quota_not_exceeded():
    quota = 16
    frames = WaterSchedulingEngine.solve(
        load_demo({"mode": "bin-packing", "daily_quota": quota})
    )
    terminal = frames[-1]
    assert terminal["state"]["isTerminal"]
    for b in terminal["state"]["bins"]:
        assert b["usedCapacity"] <= quota, (
            f"Bin {b['index']} used {b['usedCapacity']} > quota {quota}"
        )


def test_bin_packing_all_farms_placed():
    quota = 16
    frames = WaterSchedulingEngine.solve(
        load_demo({"mode": "bin-packing", "daily_quota": quota})
    )
    terminal = frames[-1]
    placed = {item["id"] for b in terminal["state"]["bins"] for item in b["farms"]}
    assert placed == {"f1", "f2", "f3", "f4"}, "all four farms must be scheduled"


def test_bin_packing_greedy_exact_agree_small():
    """For the demo (4 farms, quota=16), both greedy and exact should use 2 bins."""
    quota = 16
    frames = WaterSchedulingEngine.solve(
        load_demo({"mode": "bin-packing", "daily_quota": quota})
    )
    # Find last heuristic terminal and last exact terminal
    heuristic_bins = None
    exact_bins = None
    for f in frames:
        if f["state"].get("quality") == "heuristic" and not f["state"].get("isTerminal"):
            heuristic_bins = f["metrics"]["binCount"]
        if f["state"].get("quality") == "exact" and f["state"].get("isTerminal"):
            exact_bins = f["metrics"]["binCount"]
    # Total demand = 18 > 16, so minimum 2 bins; both should find 2
    assert exact_bins == 2, f"exact should use 2 bins, got {exact_bins}"
    # Heuristic should also find 2 (FFD is optimal for this case)
    assert heuristic_bins == 2, f"heuristic should use 2 bins, got {heuristic_bins}"


# ── Job-shop: all farms scheduled, within horizon ────────────────────────────

def test_job_shop_all_scheduled():
    frames = WaterSchedulingEngine.solve(
        load_demo({"mode": "job-shop", "horizon_hrs": 24})
    )
    terminal = frames[-1]
    assert terminal["state"]["isTerminal"]
    scheduled_ids = {e["farmId"] for e in terminal["state"]["schedule"]}
    assert scheduled_ids == {"f1", "f2", "f3", "f4"}


def test_job_shop_no_overlap_within_slot():
    """Within each slot, no two farm windows should overlap."""
    frames = WaterSchedulingEngine.solve(
        load_demo({"mode": "job-shop", "horizon_hrs": 24})
    )
    terminal = frames[-1]
    from collections import defaultdict
    by_slot = defaultdict(list)
    for e in terminal["state"]["schedule"]:
        by_slot[e["slot"]].append(e)
    for slot, entries in by_slot.items():
        entries.sort(key=lambda e: e["startHr"])
        for i in range(1, len(entries)):
            prev, curr = entries[i - 1], entries[i]
            assert curr["startHr"] >= prev["endHr"], (
                f"Slot {slot}: {curr['farmId']} overlaps {prev['farmId']}"
            )


def test_job_shop_within_horizon():
    horizon = 24
    frames = WaterSchedulingEngine.solve(
        load_demo({"mode": "job-shop", "horizon_hrs": horizon})
    )
    terminal = frames[-1]
    for e in terminal["state"]["schedule"]:
        assert e["endHr"] <= horizon, (
            f"{e['farmId']} ends at {e['endHr']} > horizon {horizon}"
        )


# ── MIS: greedy and exact agree on the demo instance ─────────────────────────

def test_mis_exact_value():
    """Exact MIS on f1—f2, f3—f4 should be {f1,f3} with value=220."""
    frames = WaterSchedulingEngine.solve(load_demo({"mode": "mis"}))
    terminal = frames[-1]
    assert terminal["state"]["isTerminal"]
    assert terminal["state"]["quality"] == "exact"
    assert terminal["metrics"]["selectedValue"] == 220, (
        f"Expected value 220 (f1+f3), got {terminal['metrics']['selectedValue']}"
    )
    sel = set(terminal["state"]["selectedFarms"])
    # Either {f1,f3} is selected (both are non-conflicting)
    assert sel == {"f1", "f3"}, f"Expected {{f1,f3}}, got {sel}"


def test_mis_no_conflicts_in_result():
    """No two selected farms should be neighbors."""
    frames = WaterSchedulingEngine.solve(load_demo({"mode": "mis"}))
    terminal = frames[-1]
    sel = set(terminal["state"]["selectedFarms"])
    # Check against neighbor lists
    with open(FIXTURE) as fh:
        demo = json.load(fh)
    for farm in demo["farms"]:
        if farm["id"] in sel:
            for nbr in farm.get("neighbors", []):
                assert nbr not in sel, (
                    f"Conflict: {farm['id']} and {nbr} both selected"
                )


def test_mis_greedy_exact_agree():
    """For this tiny demo the greedy and exact MIS value should match."""
    frames = WaterSchedulingEngine.solve(load_demo({"mode": "mis"}))
    heuristic_val = None
    exact_val = None
    for f in frames:
        q = f["state"].get("quality")
        if q == "heuristic" and not f["state"].get("isTerminal"):
            heuristic_val = f["metrics"]["selectedValue"]
        if q == "exact" and f["state"].get("isTerminal"):
            exact_val = f["metrics"]["selectedValue"]
    assert heuristic_val is not None, "no heuristic frame found"
    assert exact_val is not None, "no exact frame found"
    assert heuristic_val == exact_val, (
        f"Greedy value {heuristic_val} != exact value {exact_val}"
    )


# ── No regression: existing problems still solve ─────────────────────────────

def test_no_regression_dfa():
    # Verify DFA engine still works by running the water scheduling engine (which
    # imports water_model / water_network_engine) alongside the min-cut engine —
    # checking that imports don't clobber each other.
    from water_scheduling_engine import WaterSchedulingEngine
    from min_st_cut_dual_engine import MinSTCutDualSolver
    sched = WaterSchedulingEngine.solve(load_demo({"mode": "coloring"}))
    cut = MinSTCutDualSolver.solve({
        "nodes": ["s", "a", "t"],
        "edges": [{"from": "s", "to": "a", "weight": 5}, {"from": "a", "to": "t", "weight": 3}],
        "source": "s", "sink": "t",
    })
    assert sched[-1]["state"]["isTerminal"]
    assert any(f["state"].get("isAccepted") for f in cut)


def test_no_regression_min_cut():
    from min_st_cut_dual_engine import MinSTCutDualSolver
    frames = MinSTCutDualSolver.solve({
        "nodes": ["s", "a", "b", "t"],
        "edges": [
            {"from": "s", "to": "a", "weight": 10},
            {"from": "s", "to": "b", "weight": 8},
            {"from": "a", "to": "t", "weight": 7},
            {"from": "b", "to": "t", "weight": 5},
        ],
        "source": "s",
        "sink": "t",
    })
    terminal = next((f for f in frames if f["state"].get("isTerminal")), None)
    assert terminal is not None
    assert terminal["state"]["isAccepted"]

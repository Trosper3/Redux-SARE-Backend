"""
Tests for FarmSelectionEngine: B&B correctness, network feasibility, pruning (B.1).

Run from Backend/:  python -m pytest tests/ -v
  or directly:      python tests/test_farm_selection.py
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from farm_selection_engine import FarmSelectionEngine, _check_network_feasible, _frac_bound
from water_model import parse_water_network, water_to_flow_inputs
from water_network_engine import _run_max_flow

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "water_demo.json")


def load_network_data():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def load_network():
    return parse_water_network(load_network_data())


# ── Brute-force reference ─────────────────────────────────────────────────────

def brute_force_optimal(network, capacity, mode="fractional"):
    """
    Try all 2^n subsets; return (best_value, best_farm_ids).
    For mode='network', validate each subset with a feasibility max-flow.
    For mode='fractional', validate only that total demand <= capacity.
    """
    farms = network.farms
    best_val = 0
    best_ids = []
    for r in range(len(farms) + 1):
        for combo in itertools.combinations(farms, r):
            total_dem = sum(f.demand for f in combo)
            if total_dem > capacity:
                continue
            if mode == "network":
                if not _check_network_feasible(list(combo), network):
                    continue
            val = sum(f.value for f in combo)
            if val > best_val:
                best_val = val
                best_ids = sorted(f.id for f in combo)
    return best_val, best_ids


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_terminal(frames):
    return frames[-1]


def get_selected_farms(frames):
    return sorted(get_terminal(frames)["state"]["selectedFarms"])


def get_optimal_value(frames):
    return get_terminal(frames)["state"]["optimalValue"]


def get_explored(frames):
    return get_terminal(frames)["metrics"]["explored"]


def get_pruned(frames):
    return get_terminal(frames)["metrics"]["pruned"]


def get_capacity(frames):
    return get_terminal(frames)["state"]["capacity"]


# ── B.1 Acceptance criteria ───────────────────────────────────────────────────

def test_fractional_bb_matches_brute_force():
    """B&B (fractional bound) optimal value matches brute-force on demo network."""
    data = load_network_data()
    network = parse_water_network(data)
    fi = water_to_flow_inputs(network)
    capacity, _, _ = _run_max_flow(fi["nodes"], fi["edges"], fi["source"], fi["sink"])

    frames = FarmSelectionEngine.solve({**data, "bound_mode": "fractional"})
    terminal = get_terminal(frames)

    assert terminal["state"]["isTerminal"], "Last frame must be terminal"
    assert terminal["state"]["isAccepted"], "Terminal frame must be accepted"

    bb_val = get_optimal_value(frames)
    bf_val, _ = brute_force_optimal(network, capacity, mode="fractional")
    assert bb_val == bf_val, f"B&B value {bb_val} ≠ brute-force {bf_val}"


def test_network_bb_matches_brute_force():
    """B&B (network bound) optimal value matches brute-force with feasibility check."""
    data = load_network_data()
    network = parse_water_network(data)
    fi = water_to_flow_inputs(network)
    capacity, _, _ = _run_max_flow(fi["nodes"], fi["edges"], fi["source"], fi["sink"])

    frames = FarmSelectionEngine.solve({**data, "bound_mode": "network"})

    bb_val = get_optimal_value(frames)
    bf_val, bf_ids = brute_force_optimal(network, capacity, mode="network")
    assert bb_val == bf_val, f"Network B&B value {bb_val} ≠ brute-force {bf_val}"

    bb_ids = get_selected_farms(frames)
    # Values must match; farm-ID sets may differ if there are ties (any optimal is valid)
    # Check that the B&B set is also network-feasible and matches the optimal value
    bb_farms = [f for f in network.farms if f.id in bb_ids]
    assert _check_network_feasible(bb_farms, network), \
        f"B&B selected set {bb_ids} is not network-feasible"
    assert sum(f.value for f in bb_farms) == bf_val, \
        f"B&B set {bb_ids} value {sum(f.value for f in bb_farms)} ≠ optimal {bf_val}"


def test_selected_set_passes_network_feasibility():
    """The B&B (network mode) result passes a direct feasibility max-flow."""
    data = load_network_data()
    network = parse_water_network(data)

    frames = FarmSelectionEngine.solve({**data, "bound_mode": "network"})
    selected_ids = get_selected_farms(frames)
    selected_farms = [f for f in network.farms if f.id in selected_ids]

    assert _check_network_feasible(selected_farms, network), \
        f"Selected farms {selected_ids} fail network feasibility"


def test_pruned_count_greater_than_zero():
    """At least one B&B node must be pruned (proves the bound fires)."""
    data = load_network_data()
    frames = FarmSelectionEngine.solve({**data, "bound_mode": "fractional"})
    pruned = get_pruned(frames)
    assert pruned > 0, f"Expected pruned > 0, got {pruned}"


def test_bb_tree_nodes_present_in_frames():
    """Every non-opening frame after the start should carry tree nodes."""
    data = load_network_data()
    frames = FarmSelectionEngine.solve({**data, "bound_mode": "fractional"})
    # Opening frame has empty tree; terminal frame must have nodes
    terminal = get_terminal(frames)
    assert len(terminal["state"]["tree"]) > 0, "Terminal frame must carry B&B tree nodes"


def test_fractional_bound_computes_correctly():
    """Unit-test _frac_bound on a tiny hand-computed example."""
    from water_model import WaterFarm
    farms = [
        WaterFarm(id="a", nodeId="n_a", demand=3, value=30, durationHrs=1),
        WaterFarm(id="b", nodeId="n_b", demand=5, value=40, durationHrs=1),
        WaterFarm(id="c", nodeId="n_c", demand=4, value=20, durationHrs=1),
    ]
    # Sorted by ratio: a(10), b(8), c(5)
    farms.sort(key=lambda f: f.value / f.demand, reverse=True)
    capacity = 8

    # Include a (demand 3, value 30, remaining 5).
    # Include b fully (demand 5, value 40). remaining = 0.
    # Bound = 30 + 40 = 70.
    bound = _frac_bound(farms, 0, 0, 0, capacity)
    assert bound == 70.0, f"Expected 70.0, got {bound}"

    # Fractional at depth 2 (after a included): remaining = 8-3 = 5.
    # b fits (5 == 5), bound += 40. remaining = 0. c: nothing left.
    bound2 = _frac_bound(farms, 1, 3, 30, capacity)
    assert bound2 == 70.0, f"Expected 70.0, got {bound2}"


def test_pre_computed_max_flow_used():
    """Passing max_flow=5 limits capacity and only small subsets can be selected."""
    data = load_network_data()
    network = parse_water_network(data)

    frames = FarmSelectionEngine.solve({**data, "max_flow": 5, "bound_mode": "fractional"})
    cap = get_capacity(frames)
    assert cap == 5, f"Expected capacity 5, got {cap}"

    selected = get_selected_farms(frames)
    selected_farms = [f for f in network.farms if f.id in selected]
    total_dem = sum(f.demand for f in selected_farms)
    assert total_dem <= 5, f"Selected demand {total_dem} exceeds capped capacity 5"


def test_no_regression_existing_engines():
    """Spot-check: min-st-cut-dual engine still runs without errors."""
    import min_st_cut_dual_engine
    result = min_st_cut_dual_engine.MinSTCutDualSolver.solve({
        "nodes": ["s", "a", "b", "t"],
        "edges": [
            {"from": "s", "to": "a", "weight": 10},
            {"from": "a", "to": "t", "weight": 10},
            {"from": "s", "to": "b", "weight": 10},
            {"from": "b", "to": "t", "weight": 10},
        ],
        "source": "s",
        "sink": "t",
    })
    assert isinstance(result, list) and len(result) > 0, "min-st-cut-dual must return frames"


# ── I.1: FPTAS within ε (direct call, bypasses FPTAS_THRESHOLD) ──────────────

def test_fptas_within_epsilon():
    """
    I.1 acceptance: FPTAS result is within (1-ε) of brute-force optimal on a
    small synthetic instance.  Calls _fptas_solve directly so the 10,000-farm
    threshold does not block the test.
    """
    from farm_selection_engine import _fptas_solve
    from water_model import WaterFarm, WaterNode, WaterPipe, WaterNetwork

    farms = [
        WaterFarm(id="a", nodeId="n_a", demand=2, value=30, durationHrs=1),
        WaterFarm(id="b", nodeId="n_b", demand=3, value=50, durationHrs=1),
        WaterFarm(id="c", nodeId="n_c", demand=2, value=25, durationHrs=1),
        WaterFarm(id="d", nodeId="n_d", demand=4, value=60, durationHrs=1),
        WaterFarm(id="e", nodeId="n_e", demand=1, value=15, durationHrs=1),
    ]
    nodes = [WaterNode(id="aq", kind="aquifer")] + [
        WaterNode(id=f"n_{f.id}", kind="farm") for f in farms
    ]
    pipes = [WaterPipe(from_="aq", to=f"n_{f.id}", capacity=f.demand) for f in farms]
    network = WaterNetwork(nodes=nodes, pipes=pipes, farms=farms, source="aq", sink="virtual_sink")

    capacity = 6
    epsilon = 0.001

    # Brute-force optimum (demand-feasible subsets only)
    best = 0
    for r in range(len(farms) + 1):
        for combo in itertools.combinations(farms, r):
            if sum(f.demand for f in combo) <= capacity:
                best = max(best, sum(f.value for f in combo))

    frames = _fptas_solve(network, capacity, epsilon=epsilon)
    terminal = frames[-1]
    fptas_val = terminal["metrics"]["approximate_value"]

    assert fptas_val >= best * (1 - epsilon), (
        f"FPTAS value {fptas_val} is more than {epsilon * 100:.1f}% below optimal {best}"
    )

    # Selected set must respect capacity
    selected_ids = set(terminal["state"]["selectedFarms"])
    used = sum(f.demand for f in farms if f.id in selected_ids)
    assert used <= capacity, f"FPTAS selected set uses {used} > capacity {capacity}"


if __name__ == "__main__":
    test_fractional_bound_computes_correctly();    print("✓ frac_bound unit test")
    test_fractional_bb_matches_brute_force();      print("✓ fractional B&B == brute force")
    test_network_bb_matches_brute_force();         print("✓ network B&B == brute force")
    test_selected_set_passes_network_feasibility();print("✓ selected set is network-feasible")
    test_pruned_count_greater_than_zero();         print("✓ pruned > 0")
    test_bb_tree_nodes_present_in_frames();        print("✓ tree nodes in terminal frame")
    test_pre_computed_max_flow_used();             print("✓ pre-computed max_flow cap respected")
    test_no_regression_existing_engines();         print("✓ min-st-cut-dual no regression")
    test_fptas_within_epsilon();                   print("✓ FPTAS within epsilon of optimal")
    print("All B.1/I.1 tests passed.")

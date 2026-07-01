"""
Tests for WaterNetworkEngine: max-flow correctness, slackUp, and pipe-upgrade what-if (A.1 + A.2).

Run from Backend/:  python -m pytest tests/ -v
  or directly:      python tests/test_water_engine.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from water_model import parse_water_network, water_to_flow_inputs
from water_network_engine import WaterNetworkEngine, _build_residual, _run_max_flow

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "water_demo.json")


def load_flow_inputs():
    with open(FIXTURE, encoding="utf-8") as f:
        data = json.load(f)
    return water_to_flow_inputs(parse_water_network(data))


# ── A.1: max-flow correctness ─────────────────────────────────────────────────

def test_max_flow_equals_cut_capacity():
    fi = load_flow_inputs()
    steps = WaterNetworkEngine.solve(fi)
    terminal = steps[-1]
    assert terminal["state"]["isTerminal"]
    assert terminal["state"]["isAccepted"]
    cut = terminal["state"]["cutCapacity"]
    mf  = terminal["metrics"]["max_flow"]
    assert cut == mf, f"cut_capacity {cut} ≠ max_flow {mf}"


def test_expected_max_flow_value():
    # water_demo: aquifer→j_north cap 7 is choke; north demand 4+5=9 → 7 flows north
    # south: cap 10, demand 6+3=9 → 9 flows south → total = 16
    fi = load_flow_inputs()
    steps = WaterNetworkEngine.solve(fi)
    mf = steps[-1]["metrics"]["max_flow"]
    assert mf == 16, f"Expected max_flow 16, got {mf}"


# ── A.1: slackUp brute-force check ───────────────────────────────────────────

def _brute_force_max_flow(fi, u, v, delta):
    edges = [
        {**e, "weight": e["weight"] + delta} if e["from"] == u and e["to"] == v else e
        for e in fi["edges"]
    ]
    flow, _, _ = _run_max_flow(fi["nodes"], edges, fi["source"], fi["sink"])
    return flow


def test_slack_up_brute_force():
    fi = load_flow_inputs()
    steps = WaterNetworkEngine.solve(fi)
    terminal = steps[-1]["state"]

    # Universal bottleneck edge exists
    universal = terminal.get("universalEdges", [])
    assert universal, "Expected at least one universal bottleneck edge on demo network"

    slack = terminal.get("slackUp")
    assert slack is not None and slack > 0, "Expected slackUp > 0 for the choke edge"

    u, v = universal[0]["from"], universal[0]["to"]
    base_flow = 16

    # Adding slackUp units should increase flow by exactly slackUp
    flow_after = _brute_force_max_flow(fi, u, v, slack)
    assert flow_after == base_flow + slack, (
        f"Brute force: flow after +{slack} on {u}→{v} = {flow_after}, "
        f"expected {base_flow + slack}"
    )

    # Adding slackUp+1 should NOT increase flow further (demands saturated)
    flow_extra = _brute_force_max_flow(fi, u, v, slack + 1)
    assert flow_extra == flow_after, (
        f"Expected no additional flow at slackUp+1, got {flow_extra} > {flow_after}"
    )


# ── A visualization: edgeFlows in terminal state ──────────────────────────────

def test_edge_flows_in_terminal_state():
    fi = load_flow_inputs()
    steps = WaterNetworkEngine.solve(fi)
    terminal = steps[-1]["state"]

    flows = terminal.get("edgeFlows")
    assert flows is not None, "Terminal state must have edgeFlows dict"
    assert isinstance(flows, dict), "edgeFlows must be a dict"

    # Spot-check: every original edge should have a flow entry
    for e in fi["edges"]:
        key = f"{e['from']}|{e['to']}"
        assert key in flows, f"edgeFlows missing key {key}"
        assert 0 <= flows[key] <= e["weight"], (
            f"edgeFlows[{key}]={flows[key]} out of range [0, {e['weight']}]"
        )

    # Conservation: total flow into the sink must equal max_flow
    mf = terminal["cutCapacity"]
    sink = fi["sink"]
    inflow = sum(flows.get(f"{e['from']}|{sink}", 0) for e in fi["edges"] if e["to"] == sink)
    assert inflow == mf, f"Sink inflow {inflow} ≠ max_flow {mf}"


# ── A.2: pipe-upgrade what-if ─────────────────────────────────────────────────

def test_upgrade_choke_edge_raises_flow_and_moves_bottleneck():
    fi = load_flow_inputs()
    # aquifer→j_north (cap 7) is the choke; upgrade by 3
    result = WaterNetworkEngine.solve_upgrade({
        **fi,
        "transformation": {"id": "upgrade-pipe", "pipe": ["aquifer", "j_north"], "delta": 3},
    })

    assert "before" in result and "after" in result and "delta" in result
    d = result["delta"]
    assert d["maxFlowAfter"] > d["maxFlowBefore"], "Upgrade choke → flow should increase"
    assert d["bottleneckMoved"], "Bottleneck should shift after upgrading the choke"
    assert d["warmStart"] is True, "Solve must use warm start, not cold recompute"


def test_upgrade_non_binding_edge_leaves_flow_unchanged():
    fi = load_flow_inputs()
    # aquifer→j_south (cap 10) is NOT in the min-cut; upgrading it should not change max flow
    result = WaterNetworkEngine.solve_upgrade({
        **fi,
        "transformation": {"id": "upgrade-pipe", "pipe": ["aquifer", "j_south"], "delta": 5},
    })

    d = result["delta"]
    assert d["maxFlowBefore"] == d["maxFlowAfter"], (
        f"Upgrading non-binding edge should not change flow: "
        f"{d['maxFlowBefore']} vs {d['maxFlowAfter']}"
    )


if __name__ == "__main__":
    test_max_flow_equals_cut_capacity();  print("✓ max_flow == cut_capacity")
    test_expected_max_flow_value();       print("✓ max_flow == 16")
    test_slack_up_brute_force();          print("✓ slackUp brute-force match")
    test_upgrade_choke_edge_raises_flow_and_moves_bottleneck();
    print("✓ upgrade choke → flow up, bottleneck moves")
    test_upgrade_non_binding_edge_leaves_flow_unchanged();
    print("✓ upgrade non-binding → flow unchanged")
    print("All engine tests passed.")

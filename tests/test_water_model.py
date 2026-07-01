"""
Tests for WaterNetwork data model and engine field-name adapter (Task 0.1).

Run from Backend/:  python -m pytest tests/ -v
  or directly:      python tests/test_water_model.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import min_st_cut_dual_engine
from water_model import parse_water_network, water_to_flow_inputs

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "water_demo.json")


def load():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def test_super_sink_edges_generated():
    data = load()
    network = parse_water_network(data)
    fi = water_to_flow_inputs(network)

    assert network.sink in fi["nodes"], f"Sink '{network.sink}' missing from nodes"

    sink_edges = [e for e in fi["edges"] if e["to"] == network.sink]
    assert len(sink_edges) == len(network.farms), (
        f"Expected {len(network.farms)} super-sink edges, got {len(sink_edges)}"
    )

    demands = {f.nodeId: f.demand for f in network.farms}
    for e in sink_edges:
        assert e["weight"] == demands[e["from"]], (
            f"Edge {e['from']}→sink weight {e['weight']} ≠ demand {demands[e['from']]}"
        )


def test_round_trip_through_dual_engine():
    data = load()
    network = parse_water_network(data)
    fi = water_to_flow_inputs(network)

    steps = min_st_cut_dual_engine.MinSTCutDualSolver.solve(fi)

    assert len(steps) > 0
    terminal = steps[-1]
    assert terminal["state"]["isTerminal"] is True
    assert terminal["state"]["isAccepted"] is True

    cut_cap = terminal["state"].get("cutCapacity")
    max_flow = terminal["metrics"].get("cut_capacity") or terminal["state"].get("cutCapacity")
    assert cut_cap is not None, "Terminal frame missing cutCapacity"
    assert max_flow == cut_cap, f"max_flow {max_flow} ≠ cut_capacity {cut_cap}"


def test_super_sink_node_not_in_original_nodes():
    data = load()
    network = parse_water_network(data)
    original_ids = {n.id for n in network.nodes}
    assert network.sink not in original_ids, (
        "Virtual sink should NOT appear in WaterNetwork.nodes — adapter creates it"
    )


def test_legacy_single_source_still_solves_correctly():
    """Legacy aquifer+farm fixture must continue to solve to max_flow=16 unchanged."""
    data = load()
    network = parse_water_network(data)
    fi = water_to_flow_inputs(network)

    # No virtual_source node should appear for a single-source network
    from water_model import VIRTUAL_SOURCE_ID
    assert VIRTUAL_SOURCE_ID not in fi["nodes"], (
        "Single-source network should NOT get a virtual super-source node"
    )
    assert fi["source"] == "aquifer", f"Expected source 'aquifer', got {fi['source']}"

    steps = min_st_cut_dual_engine.MinSTCutDualSolver.solve(fi)
    terminal = steps[-1]
    assert terminal["state"]["isTerminal"] is True
    assert terminal["state"]["isAccepted"] is True
    mf = terminal["metrics"].get("cut_capacity") or terminal["state"].get("cutCapacity")
    assert mf == 16, f"Legacy single-source max_flow expected 16, got {mf}"


def test_multi_source_max_flow_equals_cut_capacity():
    """
    Two-source network: virtual super-source routes to both sources;
    max_flow must equal cut_capacity (max-flow min-cut theorem).

    Network:
        aquifer_1 → pipe_1 (cap 5) → farm_A_node (demand 4) → virtual_sink
        aquifer_2 → pipe_2 (cap 8) → farm_B_node (demand 7) → virtual_sink
    Expected max_flow = min(5,4) + min(8,7) = 4 + 7 = 11
    """
    from water_model import VIRTUAL_SOURCE_ID
    from water_network_engine import WaterNetworkEngine

    data = {
        "nodes": [
            {"id": "aquifer_1",   "kind": "source",   "x": 0,   "y": 100},
            {"id": "aquifer_2",   "kind": "source",   "x": 0,   "y": 300},
            {"id": "farm_A_node", "kind": "junction", "x": 300, "y": 100},
            {"id": "farm_B_node", "kind": "junction", "x": 300, "y": 300},
        ],
        "pipes": [
            {"from": "aquifer_1", "to": "farm_A_node", "capacity": 5},
            {"from": "aquifer_2", "to": "farm_B_node", "capacity": 8},
        ],
        "farms": [
            {"id": "fA", "nodeId": "farm_A_node", "demand": 4, "value": 100, "durationHrs": 2, "neighbors": []},
            {"id": "fB", "nodeId": "farm_B_node", "demand": 7, "value": 150, "durationHrs": 3, "neighbors": []},
        ],
        "sources": ["aquifer_1", "aquifer_2"],
        "sink": "virtual_sink",
    }

    network = parse_water_network(data)
    assert network.source == VIRTUAL_SOURCE_ID, (
        f"Multi-source network should have source='{VIRTUAL_SOURCE_ID}', got {network.source!r}"
    )
    assert len(network.sources) == 2

    fi = water_to_flow_inputs(network)
    assert VIRTUAL_SOURCE_ID in fi["nodes"], "virtual_source must appear in node list"
    assert fi["source"] == VIRTUAL_SOURCE_ID

    steps = WaterNetworkEngine.solve(fi)
    terminal = steps[-1]
    assert terminal["state"]["isTerminal"] is True
    assert terminal["state"]["isAccepted"] is True

    mf  = terminal["metrics"]["max_flow"]
    cut = terminal["state"]["cutCapacity"]
    assert mf == cut, f"max_flow {mf} ≠ cut_capacity {cut}"
    assert mf == 11, f"Expected max_flow 11 for this two-source network, got {mf}"


def test_multi_source_legacy_kind_aquifer_accepted():
    """Nodes with legacy kind 'aquifer' in a sources list are still accepted."""
    from water_model import VIRTUAL_SOURCE_ID, _normalize_kind
    assert _normalize_kind("aquifer") == "source"
    assert _normalize_kind("farm") == "sink"
    assert _normalize_kind("junction") == "junction"
    assert _normalize_kind("source") == "source"
    assert _normalize_kind("sink") == "sink"

    data = {
        "nodes": [
            {"id": "aq1", "kind": "aquifer",  "x": 0, "y": 0},
            {"id": "aq2", "kind": "aquifer",  "x": 0, "y": 200},
            {"id": "fn1", "kind": "farm",     "x": 300, "y": 100},
        ],
        "pipes": [
            {"from": "aq1", "to": "fn1", "capacity": 3},
            {"from": "aq2", "to": "fn1", "capacity": 4},
        ],
        "farms": [
            {"id": "f1", "nodeId": "fn1", "demand": 6, "value": 100, "durationHrs": 2, "neighbors": []},
        ],
        "sources": ["aq1", "aq2"],
        "sink": "virtual_sink",
    }
    network = parse_water_network(data)
    assert network.source == VIRTUAL_SOURCE_ID
    # kinds normalized
    kinds = {n.id: n.kind for n in network.nodes}
    assert kinds["aq1"] == "source"
    assert kinds["aq2"] == "source"
    assert kinds["fn1"] == "sink"

    from water_network_engine import WaterNetworkEngine
    fi = water_to_flow_inputs(network)
    steps = WaterNetworkEngine.solve(fi)
    terminal = steps[-1]
    assert terminal["state"]["isTerminal"] is True
    mf  = terminal["metrics"]["max_flow"]
    cut = terminal["state"]["cutCapacity"]
    assert mf == cut, f"max_flow {mf} ≠ cut_capacity {cut}"
    # aq1→fn1 (cap 3) + aq2→fn1 (cap 4) = 7, but demand = 6 → max_flow = 6
    assert mf == 6, f"Expected max_flow 6, got {mf}"


if __name__ == "__main__":
    test_super_sink_edges_generated()
    print("✓ test_super_sink_edges_generated")
    test_round_trip_through_dual_engine()
    print("✓ test_round_trip_through_dual_engine")
    test_super_sink_node_not_in_original_nodes()
    print("✓ test_super_sink_node_not_in_original_nodes")
    test_legacy_single_source_still_solves_correctly()
    print("✓ test_legacy_single_source_still_solves_correctly")
    test_multi_source_max_flow_equals_cut_capacity()
    print("✓ test_multi_source_max_flow_equals_cut_capacity")
    test_multi_source_legacy_kind_aquifer_accepted()
    print("✓ test_multi_source_legacy_kind_aquifer_accepted")
    print("All model tests passed.")

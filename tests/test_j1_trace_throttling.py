"""
Tests for J.1 — trace throttling via detail='result'.

Verifies:
  1. detail='full' (default) still returns multiple frames (backward-compat).
  2. detail='result' returns exactly one frame, and that frame is isTerminal.
  3. The terminal frame carries the same key metrics as the full-run terminal.
  4. solve_upgrade with detail='result' returns single-frame before/after lists.
  5. FarmSelectionEngine and WaterSchedulingEngine honour detail='result'.

Run from Backend/: python -m pytest tests/test_j1_trace_throttling.py -v
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from farm_selection_engine import FarmSelectionEngine
from water_network_engine import WaterNetworkEngine
from water_scheduling_engine import WaterSchedulingEngine

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "water_demo.json")


def _load():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def test_full_returns_multiple_frames():
    data = _load()
    steps = WaterNetworkEngine.solve(data, detail="full")
    assert len(steps) > 1, "detail='full' must return more than one frame"


def test_result_returns_single_terminal_frame():
    data = _load()
    steps = WaterNetworkEngine.solve(data, detail="result")
    assert len(steps) == 1, "detail='result' must return exactly one frame"
    assert steps[0]["state"].get("isTerminal") is True


def test_result_terminal_matches_full_terminal():
    data = _load()
    full_steps = WaterNetworkEngine.solve(data, detail="full")
    result_steps = WaterNetworkEngine.solve(data, detail="result")
    full_terminal = full_steps[-1]
    result_terminal = result_steps[0]
    # Key metrics must agree
    assert result_terminal["metrics"].get("cutCapacity") == full_terminal["metrics"].get("cutCapacity")
    assert result_terminal["state"].get("isAccepted") == full_terminal["state"].get("isAccepted")


def test_default_detail_is_full():
    data = _load()
    steps_default = WaterNetworkEngine.solve(data)
    steps_full = WaterNetworkEngine.solve(data, detail="full")
    assert len(steps_default) == len(steps_full), "default detail must equal detail='full'"


# ── solve_upgrade ─────────────────────────────────────────────────────────────

def test_upgrade_result_trims_before_after():
    data = _load()
    payload = {**data, "transformation": {"id": "upgrade-pipe", "pipe": ["aquifer", "j_north"], "delta": 2}}
    result = WaterNetworkEngine.solve_upgrade(payload, detail="result")
    assert len(result["before"]) == 1, "before must be single frame with detail='result'"
    assert len(result["after"]) == 1, "after must be single frame with detail='result'"
    assert result["before"][0]["state"].get("isTerminal") is True
    assert result["after"][0]["state"].get("isTerminal") is True


def test_upgrade_full_has_multiple_frames():
    data = _load()
    payload = {**data, "transformation": {"id": "upgrade-pipe", "pipe": ["aquifer", "j_north"], "delta": 2}}
    result = WaterNetworkEngine.solve_upgrade(payload, detail="full")
    assert len(result["before"]) > 1
    assert len(result["after"]) > 1


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def test_farm_selection_result_single_terminal():
    data = _load()
    steps = FarmSelectionEngine.solve(data, detail="result")
    assert len(steps) == 1
    assert steps[0]["state"].get("isTerminal") is True


def test_farm_selection_result_matches_full_optimal():
    data = _load()
    full_steps = FarmSelectionEngine.solve(data, detail="full")
    result_steps = FarmSelectionEngine.solve(data, detail="result")
    assert result_steps[0]["metrics"].get("optimal") == full_steps[-1]["metrics"].get("optimal")


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def test_scheduling_result_single_terminal_coloring():
    data = {**_load(), "selected_farms": ["f1", "f2", "f3", "f4"], "mode": "coloring"}
    steps = WaterSchedulingEngine.solve(data, detail="result")
    assert len(steps) == 1
    assert steps[0]["state"].get("isTerminal") is True


def test_scheduling_result_single_terminal_bin_packing():
    data = {**_load(), "selected_farms": ["f1", "f2", "f3", "f4"], "mode": "bin-packing", "daily_quota": 10}
    steps = WaterSchedulingEngine.solve(data, detail="result")
    assert len(steps) == 1
    assert steps[0]["state"].get("isTerminal") is True


def test_scheduling_full_has_multiple_frames():
    data = {**_load(), "selected_farms": ["f1", "f2", "f3", "f4"], "mode": "coloring"}
    steps = WaterSchedulingEngine.solve(data, detail="full")
    assert len(steps) > 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

"""
batch_baseline.py — Load and expose the committed overnight-batch schedule.

The workspace opens on the baseline; user edits create a scenario delta,
never overwriting the baseline.  The baseline is read-only from the frontend.

Usage:
    baseline = BatchBaseline.load()   # returns the full baseline dict
    BatchBaseline.diff(baseline, scenario)  # returns delta summary
"""

import json
import os
from typing import Any, Dict, List, Optional

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "data", "baseline_schedule.json")


class BatchBaseline:
    @staticmethod
    def load(path: str = _FIXTURE_PATH) -> Dict[str, Any]:
        """Load the committed baseline from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def diff(
        baseline: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute a delta summary between the baseline committed schedule and a
        user-defined scenario returned by the water pipeline.

        baseline  — the dict returned by BatchBaseline.load()
        scenario  — {
            max_flow: int,
            selected_farms: list[str],
            total_value: int,
            schedule: list[{farmId, startHr, endHr, slot}],
            pipes: list[{from, to, capacity}]    # current network state
        }

        Returns a delta dict suitable for the frontend DiffPanel.
        """
        committed = baseline.get("committed", {})

        base_flow   = committed.get("max_flow", 0)
        scen_flow   = scenario.get("max_flow", 0)

        base_farms  = set(committed.get("selected_farms", []))
        scen_farms  = set(scenario.get("selected_farms", []))

        base_value  = committed.get("total_value", 0)
        scen_value  = scenario.get("total_value", 0)

        # Pipe capacity changes relative to the baseline network snapshot
        base_pipes: Dict[str, int] = {
            f"{p['from']}→{p['to']}": p["capacity"]
            for p in baseline.get("network_snapshot", {}).get("pipes", [])
        }
        scen_pipes: Dict[str, int] = {
            f"{p['from']}→{p['to']}": p["capacity"]
            for p in scenario.get("pipes", [])
        }
        pipe_changes: List[Dict[str, Any]] = []
        for key, base_cap in base_pipes.items():
            scen_cap = scen_pipes.get(key, base_cap)
            if scen_cap != base_cap:
                pipe_changes.append({
                    "pipe": key,
                    "baseline_capacity": base_cap,
                    "scenario_capacity": scen_cap,
                    "delta": scen_cap - base_cap,
                })

        return {
            "delta_flow":   scen_flow - base_flow,
            "delta_value":  scen_value - base_value,
            "baseline_flow":  base_flow,
            "scenario_flow":  scen_flow,
            "baseline_value": base_value,
            "scenario_value": scen_value,
            "farms_added":   sorted(scen_farms - base_farms),
            "farms_removed": sorted(base_farms - scen_farms),
            "pipe_changes":  pipe_changes,
            "committed_at":  committed.get("committed_at"),
        }

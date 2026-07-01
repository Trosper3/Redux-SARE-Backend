"""
Drought scenario presets for the Water Allocation Lab.

These are synthetic, illustrative networks based on the demo topology.
They are NOT official tribal data — stakeholder sign-off required before
incorporating any figures derived from real Fort Hall Reservation records.

Three scenarios:
  normal   — full aquifer recharge, baseline growing season
  moderate — aquifer-output capacity reduced ~30 % (common dry-year)
  severe   — aquifer-output capacity reduced ~60 % (rationing scenario)

Distribution pipes (junction→farm) are unchanged across scenarios;
only the aquifer-output pipes scale, modelling reduced well/spring yield.
"""

from typing import Any, Dict, List

# ── Shared topology (nodes, farms, connectivity) ──────────────────────────────

_NODES = [
    {"id": "aquifer",  "kind": "aquifer",  "x": 100, "y": 300},
    {"id": "j_north",  "kind": "junction", "x": 300, "y": 150},
    {"id": "j_south",  "kind": "junction", "x": 300, "y": 450},
    {"id": "farm_1",   "kind": "farm",     "x": 550, "y": 75 },
    {"id": "farm_2",   "kind": "farm",     "x": 550, "y": 225},
    {"id": "farm_3",   "kind": "farm",     "x": 550, "y": 375},
    {"id": "farm_4",   "kind": "farm",     "x": 550, "y": 525},
]

_FARMS = [
    {"id": "f1", "nodeId": "farm_1", "demand": 4, "value": 100, "durationHrs": 4, "neighbors": ["f2"]},
    {"id": "f2", "nodeId": "farm_2", "demand": 5, "value": 80,  "durationHrs": 3, "neighbors": ["f1"]},
    {"id": "f3", "nodeId": "farm_3", "demand": 6, "value": 120, "durationHrs": 5, "neighbors": ["f4"]},
    {"id": "f4", "nodeId": "farm_4", "demand": 3, "value": 90,  "durationHrs": 4, "neighbors": ["f3"]},
]

_DIST_PIPES = [
    {"from": "j_north", "to": "farm_1", "capacity": 5},
    {"from": "j_north", "to": "farm_2", "capacity": 7},
    {"from": "j_south", "to": "farm_3", "capacity": 6},
    {"from": "j_south", "to": "farm_4", "capacity": 4},
]

# ── Preset definitions ────────────────────────────────────────────────────────

PRESETS: List[Dict[str, Any]] = [
    {
        "id": "normal",
        "label": "Normal",
        "severity": 0,
        "description": "Full aquifer recharge — typical growing season.",
        "note": "Illustrative synthetic data only. Not official tribal records.",
        "network": {
            "nodes": _NODES,
            "pipes": [
                {"from": "aquifer", "to": "j_north", "capacity": 7},
                {"from": "aquifer", "to": "j_south", "capacity": 10},
                *_DIST_PIPES,
            ],
            "farms": _FARMS,
            "source": "aquifer",
            "sink": "virtual_sink",
        },
    },
    {
        "id": "moderate",
        "label": "Moderate Drought",
        "severity": 1,
        "description": "Aquifer recharge ~30 % below normal — common in dry years.",
        "note": "Illustrative synthetic data only. Not official tribal records.",
        "network": {
            "nodes": _NODES,
            "pipes": [
                {"from": "aquifer", "to": "j_north", "capacity": 5},   # 7 → 5
                {"from": "aquifer", "to": "j_south", "capacity": 7},   # 10 → 7
                *_DIST_PIPES,
            ],
            "farms": _FARMS,
            "source": "aquifer",
            "sink": "virtual_sink",
        },
    },
    {
        "id": "severe",
        "label": "Severe Drought",
        "severity": 2,
        "description": "Aquifer recharge ~60 % below normal — rationing scenario.",
        "note": "Illustrative synthetic data only. Not official tribal records.",
        "network": {
            "nodes": _NODES,
            "pipes": [
                {"from": "aquifer", "to": "j_north", "capacity": 3},   # 7 → 3
                {"from": "aquifer", "to": "j_south", "capacity": 4},   # 10 → 4
                *_DIST_PIPES,
            ],
            "farms": _FARMS,
            "source": "aquifer",
            "sink": "virtual_sink",
        },
    },
]


def get_preset_list() -> List[Dict[str, Any]]:
    """Return summary list (id, label, severity, description) — omits the heavy network payload."""
    return [
        {"id": p["id"], "label": p["label"], "severity": p["severity"], "description": p["description"]}
        for p in PRESETS
    ]


def get_preset(preset_id: str) -> Dict[str, Any] | None:
    """Return the full preset (including network) for a given id."""
    return next((p for p in PRESETS if p["id"] == preset_id), None)

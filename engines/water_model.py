"""
WaterNetwork: canonical shared data model for the water-allocation lab.

All three phases (max-flow, B&B selection, scheduling) work from this object.

Super-sink wiring:
    Phase 1 max-flow needs a single sink node.  The adapter synthesises a
    virtual "sink" node (id = WaterNetwork.sink) and for every farm adds:
        farm.nodeId → sink  with  capacity = farm.demand
    This represents each farm's maximum intake as an outgoing capacity edge.
    The virtual sink node id is stored in WaterNetwork.sink and is NOT
    included in WaterNetwork.nodes — the adapter creates it dynamically.

Super-source wiring (multi-source):
    When there are multiple source nodes, the adapter synthesises a virtual
    super-source (VIRTUAL_SOURCE_ID = "virtual_source") with edges to each
    real source node at large capacity (_INF_CAPACITY), then runs max-flow
    from virtual_source to the virtual sink.  Single-source inputs are
    back-compat and receive no virtual super-source node.

Node kind normalization:
    Accepts legacy kinds 'aquifer' (→ 'source') and 'farm' (→ 'sink') as
    well as the new semantic values 'source', 'junction', 'sink'.  After
    parsing, all WaterNode.kind values carry the normalized form.

Engine field-name adapter:
    min_st_cut_dual_engine (and water_network_engine) expect:
        nodes:  List[str]
        edges:  List[{"from": str, "to": str, "weight": int}]
        source: str
        sink:   str
    water_to_flow_inputs() performs this mapping.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


VIRTUAL_SOURCE_ID = "virtual_source"
_INF_CAPACITY = 10 ** 6

# Legacy → normalized kind map
_KIND_ALIASES: Dict[str, str] = {"aquifer": "source", "farm": "sink"}


def _normalize_kind(k: str) -> str:
    return _KIND_ALIASES.get(k, k)


@dataclass
class WaterNode:
    id: str
    kind: str   # 'source' | 'junction' | 'sink'  (normalized; legacy 'aquifer'/'farm' mapped in)
    x: float = 0.0
    y: float = 0.0


@dataclass
class WaterPipe:
    from_: str
    to: str
    capacity: int


@dataclass
class WaterFarm:
    id: str
    nodeId: str
    demand: int
    value: int
    durationHrs: float
    neighbors: List[str] = field(default_factory=list)


@dataclass
class WaterNetwork:
    nodes: List[WaterNode]
    pipes: List[WaterPipe]
    farms: List[WaterFarm]
    source: str        # effective flow source: VIRTUAL_SOURCE_ID if multi-source, else the single source id
    sink: str          # virtual super-sink id — synthesised by adapter, NOT in nodes
    sources: List[str] = field(default_factory=list)  # all real source node ids


def parse_water_network(data: Dict[str, Any]) -> "WaterNetwork":
    nodes = [
        WaterNode(
            id=n["id"],
            kind=_normalize_kind(n.get("kind", "junction")),
            x=float(n.get("x", 0)),
            y=float(n.get("y", 0)),
        )
        for n in data.get("nodes", [])
    ]
    pipes = [
        WaterPipe(from_=p["from"], to=p["to"], capacity=int(p["capacity"]))
        for p in data.get("pipes", [])
    ]
    farms = [
        WaterFarm(
            id=f["id"],
            nodeId=f["nodeId"],
            demand=int(f["demand"]),
            value=int(f["value"]),
            durationHrs=float(f["durationHrs"]),
            neighbors=list(f.get("neighbors", [])),
        )
        for f in data.get("farms", [])
    ]

    # Multi-source: prefer `sources` list; fall back to singular `source`
    raw_sources: List[str] = list(data.get("sources") or [])
    if not raw_sources and "source" in data:
        raw_sources = [data["source"]]

    effective_source = (
        VIRTUAL_SOURCE_ID if len(raw_sources) > 1 else (raw_sources[0] if raw_sources else "")
    )

    return WaterNetwork(
        nodes=nodes,
        pipes=pipes,
        farms=farms,
        source=effective_source,
        sink=data["sink"],
        sources=raw_sources,
    )


def water_to_flow_inputs(network: "WaterNetwork") -> Dict[str, Any]:
    """
    Adapter: WaterNetwork → min_st_cut_dual_engine / water_network_engine input.

    When there are multiple source nodes, prepends the virtual super-source
    node and adds fan-out edges to each real source.
    Always appends the virtual super-sink node and fan-in edges for every farm.
    Pipe capacities become edge weights.
    """
    node_ids: List[str] = [n.id for n in network.nodes]

    multi_source = len(network.sources) > 1
    if multi_source:
        node_ids = [VIRTUAL_SOURCE_ID] + node_ids

    node_ids = node_ids + [network.sink]

    # Original pipe edges
    edges: List[Dict[str, Any]] = [
        {"from": p.from_, "to": p.to, "weight": p.capacity}
        for p in network.pipes
    ]

    # Super-source fan-out: virtual_source → each real source at large capacity
    if multi_source:
        for src in network.sources:
            edges.append({"from": VIRTUAL_SOURCE_ID, "to": src, "weight": _INF_CAPACITY})

    # Super-sink fan-in: each farm's node → virtual sink, capacity = demand
    for farm in network.farms:
        edges.append({"from": farm.nodeId, "to": network.sink, "weight": farm.demand})

    return {
        "nodes": node_ids,
        "edges": edges,
        "source": network.source,
        "sink": network.sink,
        "algorithm": "dual-bfs",
    }

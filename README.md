# SARE 2026 — Backend

FastAPI-based solver backend that powers all algorithm workspaces and the Water Allocation Lab.

---

## Tech Stack

| Technology | Role |
|---|---|
| Python 3.11+ | Core language |
| FastAPI | Async web framework |
| Pydantic | Request / response validation |
| pytest | Test suite |
| pybind11 | C++ solver bindings (optional, `python_binding/`) |

---

## Features

### Unified Problem API

All workspaces share a single request shape:

```json
{
  "problem_id": "knapsack",
  "inputs": { ... },
  "transformation": "none",
  "mode": "dp",
  "detail": "full"
}
```

`detail: "full"` returns every animation frame; `detail: "result"` returns only the terminal frame, reducing payload size for large instances.

Responses are arrays of **step frames** — each frame carries the full algorithm state, highlighted edges/nodes, and metrics — which the frontend replays as an animation.

### Problem Coverage

**Automata & Formal Languages**
- DFA simulation, NFA simulation with epsilon-closure
- DPDA / NPDA pushdown automata
- NFA → DFA subset construction
- Regex → Thompson NFA construction

**Computational Geometry**
- Convex hull: Graham scan and gift wrapping

**Dynamic Programming**
- 0/1 Knapsack, subset-sum, edit distance

**Graph**
- TSP, graph coloring, maximum clique, SCC, vertex cover, dominating set
- Maximum independent set

**Network Flow**
- Min-st-cut (Edmonds-Karp + dual-BFS residual visualization)
- Optimal pump scheduling

**NP-Complete / NP-Hard**
- 3-SAT, bin packing (FFD + exact B&B)

**Water Allocation Lab** — see section below.

---

### Water Allocation Lab (Three-Phase Pipeline)

Domain-specific solver chain that reduces aquifer distribution to classical CS problems:

#### Phase 1 — `water_network_engine.py`

Max-flow / min-cut analysis of the pipe network.

- **Algorithm**: Edmonds-Karp (O(VE²))
- **Warm-start re-solve**: reuses previous flow state when a single pipe capacity changes
- **Min-slack computation**: identifies bottleneck upgrade priority
- **Outputs**: max_flow value, cut edges, per-node source/sink coloring, residual graph frames

#### Phase 2 — `farm_selection_engine.py`

Economic farm selection under the flow capacity found in Phase 1.

- **Algorithm**: Branch & Bound with two bound modes:
  - *Fractional*: fast knapsack upper bound
  - *Network-feasible*: exact max-flow validation at each B&B leaf
- **FPTAS fallback**: ε = 0.001 approximation kicks in for > 10,000 farms
- **Timeout**: 10 s wall-clock → greedy fallback with quality badge
- **Outputs**: B&B tree frames, incumbent farm set, total value, quality flag

#### Phase 3 — `water_scheduling_engine.py`

Assigns selected farms to delivery windows. Four scheduling modes:

| Mode | Algorithm | CS Reduction |
|---|---|---|
| `coloring` | Welsh-Powell + exact chromatic | Conflict scheduling → Graph Coloring |
| `bin-packing` | FFD + exact B&B | Quota limits → Bin Packing |
| `job-shop` | Earliest-start greedy | Time-window assignment → Job Shop |
| `mis` | Maximum independent set | Dispatch high-value non-conflicting set → MIS |

- Timeout: 5 s → heuristic fallback
- **Outputs**: slot/time assignments, quality badge (heuristic / approximate / exact)

#### Supporting Modules

| File | Purpose |
|---|---|
| `water_model.py` | `WaterNetwork` dataclass; super-sink wiring; JSON adapters |
| `batch_baseline.py` | Load committed baseline; compute delta flow/farm/pipe diffs |
| `drought_presets.py` | Three preset networks: normal, moderate (−30%), severe (−60%) |

---

## Project Structure

```
Backend/
├── main.py                  # FastAPI app; all problem routes (~1600 lines)
├── engines/                 # Modular solver engines
│   ├── water_network_engine.py
│   ├── farm_selection_engine.py
│   ├── water_scheduling_engine.py
│   ├── water_model.py
│   ├── automata_utils.py
│   ├── convexHull_engine.py
│   ├── knapsack_engine.py
│   ├── subset_sum_engine.py
│   ├── regex_engine.py
│   ├── tsp_engine.py
│   ├── graph_coloring_engine.py
│   ├── clique_engine.py
│   ├── sat_engine.py
│   ├── min_st_cut_dual_engine.py
│   ├── optimal_pump_scheduling_engine.py
│   └── ...
├── data/                    # Problem specs (JSON) loaded at startup
│   ├── water-allocation.json
│   ├── baseline_schedule.json
│   └── ...
├── tests/                   # pytest test suite
│   ├── test_water_engine.py
│   ├── test_farm_selection.py
│   ├── test_water_scheduling.py
│   ├── test_water_model.py
│   └── test_j1_trace_throttling.py
├── python_binding/          # C++ solver bindings via pybind11
├── batch_baseline.py
├── drought_presets.py
└── tools/                   # Utilities and inspectors
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/problems` | Lists all problems (scanned from `data/*.json`) |
| `POST /api/solve` | Solve a problem; returns step-frame array |
| `POST /api/water/solve` | Water allocation three-phase pipeline |
| `GET /api/water/drought-presets` | Returns the three drought preset networks |
| `POST /api/water/baseline-diff` | Computes delta between scenario and committed baseline |

CORS is open to all origins for development.

---

## Getting Started

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install fastapi pydantic uvicorn

# Start the server
uvicorn Backend.main:app --reload --port 8000
```

The server runs at `http://localhost:8000`.

---

## Running Tests

```bash
pytest Backend/tests/
```

Tests cover max-flow correctness, B&B vs. brute-force equivalence, scheduling mode correctness, data model round-trips, and trace throttling behavior.

---

## Performance Notes

- **Trace throttling**: use `detail: "result"` for large instances to return only the final frame and avoid large JSON payloads.
- **Warm-start max-flow**: pipe upgrade re-solves reuse the previous flow state.
- **Timeout fallbacks**: farm selection (10 s) and scheduling (5 s) fall back to greedy/FPTAS with explicit quality labels — no silent degradation.
- **FPTAS**: farm selection approximation ratio guaranteed at ε = 0.001 for very large instances.

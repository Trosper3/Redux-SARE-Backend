from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import json
import os
import sys
import importlib.util
from engines import regex_engine
from engines import convexHull_engine
from engines import knapsack_engine
from engines import subset_sum_engine
from engines import optimal_pump_scheduling_engine
from engines import min_st_cut_dual_engine
from engines import water_network_engine
from engines import farm_selection_engine
from engines import water_scheduling_engine
import batch_baseline
import drought_presets

_PYTHON_BINDING_DIR = os.path.join(os.path.dirname(__file__), "python_binding")
if _PYTHON_BINDING_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_BINDING_DIR)

_BINDING_BUILD_DIR = os.path.join(_PYTHON_BINDING_DIR, "build")
if os.path.isdir(_BINDING_BUILD_DIR):
    for entry in os.listdir(_BINDING_BUILD_DIR):
        candidate = os.path.join(_BINDING_BUILD_DIR, entry)
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)

def _load_solver_binding():
    module_path = os.path.join(_PYTHON_BINDING_DIR, "solver_py.cp311-win_amd64.pyd")
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location("solver_py", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    try:
        import solver_py as module
        return module
    except ImportError:
        return None

solver_py = _load_solver_binding()

app = FastAPI()

# Enable cross-origin calls from your React development environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Absolute path tracking to discover dynamic json schema files
SPEC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

class ProblemSolveRequest(BaseModel):
    problem_id: str
    inputs: Dict[str, Any]
    transformation: Dict[str, Any] | None = None
    mode: str | None = None
    detail: str = "full"   # "full" = step movie; "result" = terminal frame only

@app.get("/api/problems")
def discover_registered_problems():
    problem_registry = []
    if not os.path.exists(SPEC_DIR):
        return problem_registry
        
    for filename in os.listdir(SPEC_DIR):
        if filename.endswith(".json") and not filename.endswith(".schema.json"):
            filepath = os.path.join(SPEC_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    problem_registry.append(json.load(f))
            except Exception as e:
                print(f"Skipping unreadable problem spec: {filename}. Error: {e}")
                
    return problem_registry

@app.get("/api/baseline")
def get_baseline():
    """Return the committed overnight-batch baseline schedule (read-only)."""
    try:
        return batch_baseline.BatchBaseline.load()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Baseline fixture not found.")


@app.get("/api/drought-presets")
def list_drought_presets():
    """Return summary list of drought scenario presets (id, label, severity, description)."""
    return drought_presets.get_preset_list()


@app.get("/api/drought-presets/{preset_id}")
def get_drought_preset(preset_id: str):
    """Return the full preset network for a given drought scenario id."""
    preset = drought_presets.get_preset(preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Drought preset '{preset_id}' not found.")
    return preset


@app.post("/api/baseline/diff")
def diff_scenario(scenario: Dict[str, Any]):
    """Compute delta between the committed baseline and the user's current scenario."""
    try:
        baseline = batch_baseline.BatchBaseline.load()
        return batch_baseline.BatchBaseline.diff(baseline, scenario)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Baseline fixture not found.")


@app.post("/api/solve")
def execute_high_performance_engine(payload: ProblemSolveRequest):
    pid = payload.problem_id
    data = payload.inputs
    
    try:
            
        # =================================================================
        # 3. DFA AUTOMATON STATE SOLVER
        # =================================================================
        if pid == "dfa":
          # If a transformation hint is present and requests NFA->DFA, perform subset-construction
          if payload.transformation and isinstance(payload.transformation, dict) and payload.transformation.get("id") == "nfa-to-dfa":
            def collapse_parallel_edges(edge_list):
                grouped: Dict[tuple, List[str]] = {}
                for edge in edge_list:
                    key = (edge["source"], edge["target"])
                    grouped.setdefault(key, [])
                    label = str(edge.get("label", "")).strip()
                    if label and label not in grouped[key]:
                        grouped[key].append(label)
                collapsed = []
                for (src, tgt), labels in grouped.items():
                    collapsed.append({
                        "source": src,
                        "target": tgt,
                        "label": ",".join(labels)
                    })
                return collapsed

            # Expect the inputs to be NFA-style (states, alphabet, transitions, startState, acceptStates)
            states_str = data.get("states", "q0, q1, q2")
            alphabet_str = data.get("alphabet", "0, 1")
            transitions_str = data.get("transitions", "q0,0->q0\nq0,0->q1\nq1,1->q2\nq1,ε->q0")
            start_state = data.get("startState", "q0").strip()
            accept_states_str = data.get("acceptStates", "q2")

            states = [s.strip() for s in states_str.split(",") if s.strip()]
            alphabet = [a.strip() for a in alphabet_str.split(",") if a.strip()]
            # remove epsilon symbol from alphabet for DFA construction
            alphabet = [a for a in alphabet if a and a not in ["ε", "epsilon", "eps"]]
            accept_states = [f.strip() for f in accept_states_str.split(",") if f.strip()]

            # Build NFA transition map
            transition_map = {state: {} for state in states}
            for line in transitions_str.strip().split("\n"):
                if not line.strip() or "->" not in line:
                    continue
                left, target = line.split("->", 1)
                if "," not in left:
                    continue
                source, symbol = left.split(",", 1)
                source = source.strip()
                symbol = symbol.strip()
                target = target.strip()
                if symbol.lower() in ["ε", "epsilon", "eps", ""]:
                    symbol = "ε"
                transition_map.setdefault(source, {})
                transition_map[source].setdefault(symbol, [])
                if target not in transition_map[source][symbol]:
                    transition_map[source][symbol].append(target)

            def epsilon_closure(states_list):
                closure = set(states_list)
                stack = list(states_list)
                while stack:
                    curr = stack.pop()
                    for t in transition_map.get(curr, {}).get("ε", []):
                        if t not in closure:
                            closure.add(t)
                            stack.append(t)
                return sorted(list(closure))

            # Subset construction
            start_closure = tuple(epsilon_closure([start_state]))
            dfa_states = {start_closure}
            unmarked = [start_closure]
            dfa_transition_edges = []
            dfa_nodes = []

            steps_timeline = []

            # initial frame
            dfa_nodes = ["{" + ",".join(s) + "}" for s in dfa_states]
            steps_timeline.append({
                "action": f"Initialized DFA subset-construction from start closure {list(start_closure)}",
                "metrics": {"processed_sets": 0},
                "state": {
                    "nodes": list(dfa_nodes),
                    "edges": [],
                    "activeNodes": ["{" + ",".join(start_closure) + "}"],
                    "activeEdges": [],
                    "acceptStates": [],
                    "startState": "{" + ",".join(start_closure) + "}",
                    "conversionMode": True,
                    "acceptedEndpointStates": [],
                    "rejectedEndpointStates": [],
                    "isTerminal": False,
                }
            })

            processed = 0
            while unmarked:
                current = unmarked.pop(0)
                current_name = "{" + ",".join(current) + "}"
                dfa_nodes = list({n for n in dfa_nodes} | {current_name})

                for sym in alphabet:
                    # collect move for each nfa state in current
                    move_set = set()
                    for n in current:
                        for tgt in transition_map.get(n, {}).get(sym, []):
                            move_set.update(epsilon_closure([tgt]))
                    target_tuple = tuple(sorted(list(move_set)))
                    if not target_tuple:
                        # map to garbage
                        target_name = "garbage"
                        dfa_nodes = list({n for n in dfa_nodes} | {"garbage"})
                        dfa_transition_edges.append({"source": current_name, "target": target_name, "label": sym})
                    else:
                        target_name = "{" + ",".join(target_tuple) + "}"
                        dfa_transition_edges.append({"source": current_name, "target": target_name, "label": sym})
                        if target_tuple not in dfa_states:
                            dfa_states.add(target_tuple)
                            unmarked.append(target_tuple)

                    # produce an intermediate step frame showing the move
                    steps_timeline.append({
                        "action": f"From {current_name} on '{sym}' -> {target_name}",
                        "metrics": {"processed_sets": processed + 1, "current_symbol": sym},
                        "state": {
                            "nodes": list(dfa_nodes),
                            "edges": collapse_parallel_edges(dfa_transition_edges),
                            "activeNodes": [current_name],
                            "activeEdges": [ {"source": current_name, "target": target_name, "label": sym} ],
                            "acceptStates": [],
                            "startState": "{" + ",".join(start_closure) + "}",
                            "conversionMode": True,
                            "acceptedEndpointStates": [],
                            "rejectedEndpointStates": [],
                            "isTerminal": False,
                        }
                    })

                processed += 1

            # Finalize: mark accept states and construct DFA transition map
            dfa_accepts = []
            for state_tuple in dfa_states:
                if any(s in accept_states for s in state_tuple):
                    dfa_accepts.append("{" + ",".join(state_tuple) + "}")

            # Build transition map for DFA simulation
            dfa_transition_map = {}
            for edge in dfa_transition_edges:
                src = edge["source"]
                lbl = edge["label"]
                tgt = edge["target"]
                dfa_transition_map.setdefault(src, {})
                dfa_transition_map[src][lbl] = tgt

            # Optionally simulate the provided input string on the constructed DFA
            input_string = data.get("inputString", "")
            input_symbols = [c for c in input_string if c.strip() and c not in ["ε", "E"]]

            current_dfa = "{" + ",".join(start_closure) + "}"
            sim_history = [current_dfa]
            sim_edges = []

            for idx, ch in enumerate(input_symbols):
                prev = current_dfa
                next_state = dfa_transition_map.get(prev, {}).get(ch)
                if not next_state:
                    next_state = "garbage"
                    dfa_nodes = list({n for n in dfa_nodes} | {"garbage"})
                sim_edges.append({"source": prev, "target": next_state, "label": ch})
                current_dfa = next_state
                sim_history.append(current_dfa)
                accepted_endpoints = [current_dfa] if current_dfa in dfa_accepts else []
                rejected_endpoints = [current_dfa] if current_dfa not in dfa_accepts else []

                steps_timeline.append({
                    "action": f"DFA consumed '{ch}': {prev} -> {next_state}",
                    "metrics": {"processed_tokens": idx + 1, "current_symbol": ch},
                    "state": {
                        "nodes": list(dfa_nodes),
                        "edges": collapse_parallel_edges(dfa_transition_edges),
                        "activeNodes": [current_dfa],
                        "activeEdges": [ {"source": prev, "target": next_state, "label": ch} ],
                        "allActiveEdges": list(sim_edges),
                        "acceptStates": dfa_accepts,
                        "startState": "{" + ",".join(start_closure) + "}",
                        "conversionMode": True,
                        "acceptedEndpointStates": accepted_endpoints,
                        "rejectedEndpointStates": rejected_endpoints,
                        "isTerminal": False,
                    }
                })

            accepted_endpoints = [current_dfa] if current_dfa in dfa_accepts else []
            rejected_endpoints = [current_dfa] if current_dfa not in dfa_accepts else []

            # Final frame: subset construction complete; include accept states
            steps_timeline.append({
                "action": "Subset construction complete",
                "metrics": {"dfa_states": len(dfa_states), "simulated_steps": len(input_symbols)},
                "state": {
                    "nodes": list(dfa_nodes),
                    "edges": collapse_parallel_edges(dfa_transition_edges),
                    "activeNodes": [current_dfa],
                    "activeEdges": [],
                    "allActiveEdges": list(sim_edges),
                    "acceptStates": dfa_accepts,
                    "startState": "{" + ",".join(start_closure) + "}",
                    "conversionMode": True,
                    "acceptedEndpointStates": accepted_endpoints,
                    "rejectedEndpointStates": rejected_endpoints,
                    "isTerminal": True,
                    "isAccepted": current_dfa in dfa_accepts
                }
            })

            return steps_timeline
          states_str = data.get("states", "q0, q1, q2")
          alphabet_str = data.get("alphabet", "0, 1")
          transitions_str = data.get("transitions", "q0,0->q1\nq1,1->q2\nq2,0->q0")
          start_state = data.get("startState", "q0").strip()
          accept_states_str = data.get("acceptStates", "q2")
          input_string = data.get("inputString", "")

          states = [s.strip() for s in states_str.split(",") if s.strip()]
          alphabet = [a.strip() for a in alphabet_str.split(",") if a.strip()]
          accept_states = [f.strip() for f in accept_states_str.split(",") if f.strip()]

          # Validate start and accept states
          if start_state not in states:
              raise HTTPException(status_code=400, detail=f"Start state '{start_state}' is not declared in states set.")

          invalid_accepts = [s for s in accept_states if s not in states]
          if invalid_accepts:
              raise HTTPException(status_code=400, detail=f"Accept states not declared in states set: {', '.join(invalid_accepts)}")

          # Build and validate DFA transition map (no ε allowed, symbols must be in alphabet, states declared)
          transition_map = {state: {} for state in states}
          for line in transitions_str.strip().split("\n"):
              if not line.strip() or "->" not in line:
                  continue
              left, target = line.split("->")
              if "," not in left:
                  continue
              source, symbol = left.split(",")
              
              source = source.strip()
              symbol = symbol.strip()
              target = target.strip()

              # Normalize epsilon markers
              if symbol.lower() in ["ε", "epsilon", "eps", ""]:
                  symbol = "ε"

              # Source and target must be declared
              if source not in transition_map:
                  raise HTTPException(status_code=400, detail=f"Invalid transition '{line}': source state '{source}' is not declared in states set.")

              if target not in states:
                  raise HTTPException(status_code=400, detail=f"Invalid transition '{line}': target state '{target}' is not declared in states set.")

              # DFA-specific constraints: no epsilon transitions and symbol must belong to alphabet
              if symbol == "ε":
                  raise HTTPException(status_code=400, detail=f"Invalid transition '{line}': ε (epsilon) transitions are not permitted in a DFA.")

              if symbol not in alphabet:
                  raise HTTPException(status_code=400, detail=f"Invalid transition '{line}': symbol '{symbol}' is not in alphabet {{{', '.join(alphabet)}}}.")

              # Ensure deterministic mapping: only one target per (source,symbol)
              if symbol in transition_map[source] and transition_map[source][symbol] != target:
                  raise HTTPException(
                      status_code=400,
                      detail=f"Conflicting transitions for source '{source}' and symbol '{symbol}': '{transition_map[source][symbol]}' vs '{target}'. DFA requires a unique target for each symbol."
                  )

              transition_map[source][symbol] = target

          canvas_nodes = list(states) 
          canvas_edges = []
          for source, paths in transition_map.items():
              for symbol, target in paths.items():
                  canvas_edges.append({"source": source, "target": target, "label": symbol})

          steps_timeline = []
          current_state = start_state
          path_history = [start_state]
          active_edges_history = []
          step_traces = [] # Track incremental steps separately to build your spaced layout signature

          # Normalize input: allow blank or explicit epsilon as empty input
          trimmed_input = input_string.strip()
          if trimmed_input == "" or trimmed_input.lower() in ["ε", "epsilon", "eps"]:
              input_symbols = []
          else:
              input_symbols = [c for c in input_string if not c.isspace()]

          # Validate each input symbol belongs to the declared alphabet
          for ch in input_symbols:
              if ch not in alphabet:
                  raise HTTPException(
                      status_code=400,
                      detail=f"Input string contains symbol '{ch}' not in alphabet {{{', '.join(alphabet)}}}."
                  )

          steps_timeline.append({
              "action": f"Initialized DFA engine at start node '{start_state}'",
              "metrics": {
                  "processed_tokens": 0, 
                  "unread_buffer": input_string if input_string else "ε"
              },
              "state": {
                "nodes": list(canvas_nodes),
                "edges": list(canvas_edges),
                "activeNodes": [current_state],
                "activeEdges": [],
                "allActiveEdges": [],
                "acceptStates": accept_states,
                                "startState": start_state,
                "isTerminal": False
              }
          })

          for idx, char in enumerate(input_symbols):
              prev_state = current_state
              
              if current_state == "garbage":
                  next_state = "garbage"
              elif current_state in transition_map and char in transition_map[current_state]:
                  next_state = transition_map[current_state][char]
              else:
                  next_state = "garbage"

              if next_state == "garbage" and "garbage" not in canvas_nodes:
                  canvas_nodes.append("garbage")

              current_state = next_state
              path_history.append(current_state)

              display_prev = "Garbage" if prev_state == "garbage" else prev_state
              display_next = "Garbage" if next_state == "garbage" else next_state

              # Match NFA formatting layout with spaces exactly
              if idx == 0:
                  step_traces.append(f"{display_prev}, {char}->{display_next}")
              else:
                  step_traces.append(f"{char}->{display_next}")

              active_edge = {"source": prev_state, "target": current_state, "label": char}
              active_edges_history.append(active_edge)

              steps_timeline.append({
                  "action": f"Consumed input token '{char}': Transited {prev_state} ➔ {current_state}",
                  "metrics": {
                      "processed_tokens": idx + 1,
                      "current_symbol": char,
                      "unread_buffer": "".join(input_symbols[idx + 1:]) if (idx + 1) < len(input_symbols) else "ε"
                  },
                  "state": {
                      "nodes": list(canvas_nodes),
                      "edges": list(canvas_edges),
                      "activeNodes": [current_state],
                      "activeEdges": [active_edge],
                      "allActiveEdges": list(active_edges_history),
                      "acceptStates": accept_states,
                      "startState": start_state,
                      "isTerminal": False
                  }
              })

          is_accepted = current_state != "garbage" and current_state in accept_states
          trace_timeline_str = "   ".join(step_traces) if step_traces else "None"

          final_dfa_active = [current_state]

          # FIXED: Pass path_history down as a clean, raw list of state strings. 
          # Discard the pre-joined formatted_dfa_path variable.
          steps_timeline.append({
              "action": f"Execution Complete — Machine {'ACCEPTED' if is_accepted else 'REJECTED'} the input string",
              "metrics": {
                  "final_state": current_state,
                  "status": "ACCEPTED" if is_accepted else "REJECTED",
                  "traceTimeline": trace_timeline_str
              },
              "state": {
                  "nodes": list(canvas_nodes),
                  "edges": list(canvas_edges),
                  "activeNodes": final_dfa_active,
                  "activeEdges": [],
                  "allActiveEdges": list(active_edges_history),
                  "acceptStates": accept_states,
                  "startState": start_state,
                  "acceptedEndpointStates": [current_state] if is_accepted else [],
                  "rejectedEndpointStates": [current_state] if not is_accepted else [],
                  "isTerminal": True,
                  "isAccepted": is_accepted,
                  "path": list(path_history)  # <-- Clean string list signature synced
              }
          })

          return steps_timeline

        # =================================================================
        # 4. NFA NONDETERMINISTIC FINITE AUTOMATON SOLVER
        # =================================================================
        elif pid == "nfa":
            states_str = data.get("states", "q0, q1, q2")
            alphabet_str = data.get("alphabet", "0, 1")
            transitions_str = data.get("transitions", "q0,0->q0\nq0,0->q1\nq1,1->q2\nq1,ε->q2")
            start_state = data.get("startState", "q0").strip()
            accept_states_str = data.get("acceptStates", "q2")
            input_string = data.get("inputString", "")

            states = [s.strip() for s in states_str.split(",") if s.strip()]
            alphabet = [a.strip() for a in alphabet_str.split(",") if a.strip()]
            accept_states = [f.strip() for f in accept_states_str.split(",") if f.strip()]

            if start_state not in states:
                raise HTTPException(status_code=400, detail=f"Start state '{start_state}' is not declared in states set.")

            invalid_accepts = [s for s in accept_states if s not in states]
            if invalid_accepts:
                raise HTTPException(status_code=400, detail=f"Accept states not declared in states set: {', '.join(invalid_accepts)}")

            # Build parallel branching maps supporting multiple concurrent paths
            transition_map = {state: {} for state in states}
            for line in transitions_str.strip().split("\n"):
                if not line.strip() or "->" not in line:
                    continue
                left, target = line.split("->")
                if "," not in left:
                    continue
                source, symbol = left.split(",")
                
                source = source.strip()
                target = target.strip()
                symbol = symbol.strip()
                
                if symbol.lower() in ["ε", "epsilon", "eps", ""]:
                    symbol = "ε"

                if symbol != "ε" and symbol not in alphabet:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid transition '{line}': symbol '{symbol}' is not in alphabet {{{', '.join(alphabet)}}}."
                    )

                if target not in states:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid transition '{line}': target state '{target}' is not declared in states set."
                    )
                
                if source in transition_map:
                    if symbol not in transition_map[source]:
                        transition_map[source][symbol] = []
                    if target not in transition_map[source][symbol]:
                        transition_map[source][symbol].append(target)

            def dedupe_paths(paths: list[list[str]]) -> list[list[str]]:
                seen = set()
                unique_paths = []
                for path in paths:
                    key = tuple(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_paths.append(path)
                return unique_paths

            def epsilon_expand_path(base_path: list[str]) -> list[list[str]]:
                """Expand one path through all reachable ε branches (path-aware, cycle-safe)."""
                results = [base_path]
                queue = [base_path]

                while queue:
                    current = queue.pop(0)
                    leaf = current[-1]
                    if leaf == "garbage":
                        continue

                    for eps_target in transition_map.get(leaf, {}).get("ε", []):
                        # Allow a node to appear twice in one path but prevent infinite ε cycling.
                        if current.count(eps_target) >= 2:
                            continue

                        expanded = current + [eps_target]
                        results.append(expanded)
                        queue.append(expanded)

                return dedupe_paths(results)

            canvas_nodes = list(states)
            canvas_edges = []
            for source, paths in transition_map.items():
                for symbol, targets in paths.items():
                    for target in targets:
                        canvas_edges.append({"source": source, "target": target, "label": symbol})

            steps_timeline = []
            active_edges_history = []
            step_traces = []

            # Initialize initial execution configurations tracking multi-path histories
            initial_epsilon_paths = epsilon_expand_path([start_state])
            
            # Flatten to get unique reachable states and explicit ε edges
            initial_active = set()
            initial_epsilon_edges = []
            for eps_path in initial_epsilon_paths:
                if eps_path:
                    initial_active.add(eps_path[-1])
                    for i in range(len(eps_path) - 1):
                        eps_edge = {"source": eps_path[i], "target": eps_path[i + 1], "label": "ε"}
                        if eps_edge not in initial_epsilon_edges:
                            initial_epsilon_edges.append(eps_edge)
            
            # Use the full paths as initial paths
            active_paths_history = initial_epsilon_paths if initial_epsilon_paths else [[start_state]]
            
            # Add initial epsilon edges to history
            for eps_edge in initial_epsilon_edges:
                if eps_edge not in active_edges_history:
                    active_edges_history.append(eps_edge)

            initial_active_sorted = sorted(list(initial_active))

            steps_timeline.append({
                "action": f"Initialized NFA engine at node '{start_state}' (Closure: {initial_active})",
                "metrics": {"processed_tokens": 0, "unread_buffer": input_string if input_string else "ε"},
                "state": {
                    "nodes": list(canvas_nodes),
                    "edges": list(canvas_edges),
                    "activeNodes": list(initial_active_sorted),
                    "activeEdges": initial_epsilon_edges,
                    "allActiveEdges": initial_epsilon_edges,
                    "acceptStates": accept_states,
                    "startState": start_state,
                    "isTerminal": False,
                    "branchPaths": {f"path_{i}": path for i, path in enumerate(active_paths_history)}
                }
            })

            trimmed_input = input_string.strip()
            if trimmed_input == "" or trimmed_input.lower() in ["ε", "epsilon", "eps"]:
                input_symbols = []
            else:
                input_symbols = [char for char in input_string if not char.isspace()]

            for char in input_symbols:
                if char not in alphabet:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input string contains symbol '{char}' not in alphabet {{{', '.join(alphabet)}}}."
                    )

            # Track active states before consuming input for correct trace output
            prev_active_states = list(initial_active_sorted)
            current_active_states = list(initial_active_sorted)

            for idx, char in enumerate(input_symbols):
                next_paths_history = []
                live_step_edges = []
                next_states_set = set()

                for path in active_paths_history:
                    state = path[-1]

                    if state == "garbage":
                        garbage_path = path + ["garbage"]
                        next_paths_history.append(garbage_path)
                        next_states_set.add("garbage")
                        garbage_loop_edge = {"source": "garbage", "target": "garbage", "label": char}
                        if garbage_loop_edge not in live_step_edges:
                            live_step_edges.append(garbage_loop_edge)
                        if garbage_loop_edge not in active_edges_history:
                            active_edges_history.append(garbage_loop_edge)
                        continue

                    targets = transition_map.get(state, {}).get(char, [])
                    if not targets:
                        dead_path = path + ["garbage"]
                        next_paths_history.append(dead_path)
                        next_states_set.add("garbage")
                        dead_edge = {"source": state, "target": "garbage", "label": char}
                        if dead_edge not in live_step_edges:
                            live_step_edges.append(dead_edge)
                        if dead_edge not in active_edges_history:
                            active_edges_history.append(dead_edge)
                    else:
                        for target in targets:
                            # Record the symbol edge
                            edge_entry = {"source": state, "target": target, "label": char}
                            if edge_entry not in live_step_edges:
                                live_step_edges.append(edge_entry)
                            if edge_entry not in active_edges_history:
                                active_edges_history.append(edge_entry)
                            
                            base_symbol_path = path + [target]
                            epsilon_paths = epsilon_expand_path(base_symbol_path)

                            for eps_path in epsilon_paths:
                                for i in range(len(base_symbol_path) - 1, len(eps_path) - 1):
                                    eps_edge = {"source": eps_path[i], "target": eps_path[i + 1], "label": "ε"}
                                    if eps_edge not in live_step_edges:
                                        live_step_edges.append(eps_edge)
                                    if eps_edge not in active_edges_history:
                                        active_edges_history.append(eps_edge)
                            
                            for eps_path in epsilon_paths:
                                next_paths_history.append(eps_path)
                                next_states_set.add(eps_path[-1])

                next_paths_history = dedupe_paths(next_paths_history)

                if "garbage" in next_states_set and "garbage" not in canvas_nodes:
                    canvas_nodes.append("garbage")

                current_active_states = sorted(list(next_states_set))
                dest_sorted_str = ",".join(current_active_states)
                prev_sorted_str = ",".join(sorted(prev_active_states))

                if idx == 0:
                    step_traces.append(f"{prev_sorted_str}, {char}->{dest_sorted_str}")
                else:
                    step_traces.append(f"{char}->{dest_sorted_str}")

                active_paths_history = next_paths_history
                prev_active_states = current_active_states

                steps_timeline.append({
                    "action": f"Consumed token '{char}': Parallel active branches moved to [{dest_sorted_str}]",
                    "metrics": {
                        "processed_tokens": idx + 1,
                        "current_symbol": char,
                        "unread_buffer": "".join(input_symbols[idx + 1:]) if (idx + 1) < len(input_symbols) else "ε"
                    },
                    "state": {
                        "nodes": list(canvas_nodes),
                        "edges": list(canvas_edges),
                        "activeNodes": list(current_active_states),
                        "activeEdges": live_step_edges,
                        "allActiveEdges": list(active_edges_history),
                        "acceptStates": accept_states,
                        "startState": start_state,
                        "isTerminal": False,
                        "branchPaths": {f"path_{i}": path for i, path in enumerate(active_paths_history)}
                    }
                })

            is_accepted = any(p[-1] in accept_states for p in active_paths_history if p[-1] != "garbage")
            trace_timeline_output = "   ".join(step_traces)

            final_active_nodes = list(current_active_states)

            # Map the comprehensive execution matrix to the final timeline frame payload
            accepted_endpoint_states = sorted(list({
                p[-1] for p in active_paths_history if p and p[-1] in accept_states
            }))
            rejected_endpoint_states = sorted(list({
                p[-1] for p in active_paths_history if p and p[-1] not in accept_states
            }))

            # Green endpoint coloring takes precedence if a node appears in both sets.
            rejected_endpoint_states = [s for s in rejected_endpoint_states if s not in set(accepted_endpoint_states)]

            steps_timeline.append({
                "action": f"Execution Complete — Machine {'ACCEPTED' if is_accepted else 'REJECTED'} the NFA expression",
                "metrics": {
                    "final_states": ", ".join(current_active_states),
                    "status": "ACCEPTED" if is_accepted else "REJECTED",
                    "traceTimeline": trace_timeline_output
                },
                "state": {
                    "nodes": list(canvas_nodes),
                    "edges": list(canvas_edges),
                    "activeNodes": final_active_nodes,
                    "activeEdges": [],
                    "allActiveEdges": list(active_edges_history),
                    "acceptStates": accept_states,
                    "startState": start_state,
                    "acceptedEndpointStates": accepted_endpoint_states,
                    "rejectedEndpointStates": rejected_endpoint_states,
                    "isTerminal": True,
                    "isAccepted": is_accepted,
                    # Returns all structured parallel histories simultaneously to the frontend
                    "branchPaths": {f"path_{i}": path for i, path in enumerate(active_paths_history)}
                }
            })

            return steps_timeline

        # =================================================================
        # 5. DPDA DETERMINISTIC PUSHDOWN AUTOMATON SOLVER
        # =================================================================
        elif pid == "dpda":
            states_str = data.get("states", "q0, q1, qf")
            alphabet_str = data.get("alphabet", "a, b")
            stack_alphabet_str = data.get("stackAlphabet", "Z, A")
            transitions_str = data.get("transitions", "q0,a,Z->q0,AZ\nq0,a,A->q0,AA\nq0,b,A->q1,ε\nq1,b,A->q1,ε\nq1,ε,Z->qf,Z")
            start_state = data.get("startState", "q0").strip()
            accept_states_str = data.get("acceptStates", "qf")
            start_stack = data.get("startStack", "Z")
            input_string = data.get("inputString", "")

            states = [s.strip() for s in states_str.split(",") if s.strip()]
            alphabet = [a.strip() for a in alphabet_str.split(",") if a.strip()]
            stack_alphabet = [s.strip() for s in stack_alphabet_str.split(",") if s.strip()]
            accept_states = [f.strip() for f in accept_states_str.split(",") if f.strip()]

            if start_state not in states:
                raise HTTPException(status_code=400, detail=f"Start state '{start_state}' is not declared in states set.")

            # Parse transitions of the form: q,a,Z->p,AZ
            parsed = []
            # support transitions provided as an array or as a string with either literal newlines or escaped \n
            if isinstance(transitions_str, list):
                lines_iter = transitions_str
            else:
                if '\n' in transitions_str and '\\n' in transitions_str:
                    lines_iter = transitions_str.splitlines()
                elif '\\n' in transitions_str:
                    lines_iter = transitions_str.split('\\n')
                else:
                    lines_iter = transitions_str.splitlines()

            for line in lines_iter:
                if not line.strip() or "->" not in line:
                    continue
                left, right = line.split("->", 1)
                if "," not in left:
                    continue
                parts = [p.strip() for p in left.split(",")]
                if len(parts) < 3:
                    continue
                source, inp, pop = parts[0], parts[1], parts[2]
                target_and_push = [p.strip() for p in right.split(",", 1)]
                if len(target_and_push) < 1:
                    continue
                target = target_and_push[0]
                push = target_and_push[1] if len(target_and_push) > 1 else "ε"
                if inp.lower() in ["ε", "epsilon", "eps",""]:
                    inp = "ε"
                if pop.lower() in ["ε", "epsilon", "eps",""]:
                    pop = "ε"
                if push.lower() in ["ε", "epsilon", "eps",""]:
                    push = "ε"

                parsed.append({"source": source, "input": inp, "pop": pop, "target": target, "push": push})

            # Build transition map keyed by (source,input,pop)
            tmap = {}
            for t in parsed:
                key = (t["source"], t["input"], t["pop"]) if True else None
                tmap.setdefault(key, []).append(t)

            # Normalize input
            trimmed = input_string.strip()
            if trimmed == "" or trimmed.lower() in ["ε", "epsilon", "eps"]:
                input_chars = []
            else:
                input_chars = [c for c in input_string if not c.isspace()]

            # Simulation
            frames = []
            current_state = start_state
            stack = [start_stack]
            input_index = 0

            def push_stack_local(stack_list, push_str):
                s = list(stack_list)
                if push_str != 'ε':
                    for symbol in reversed(list(push_str)):
                        s.append(symbol)
                return s

            # initial frame
            frames.append({
                "action": f"Initialized DPDA at '{start_state}'",
                "state": {"currentState": current_state, "inputIndex": input_index, "unreadBuffer": input_string if input_string else "ε", "stack": list(stack)},
                "metrics": {"processed_tokens": 0}
            })

            progressed = True
            visited_eps = set()
            while progressed:
                progressed = False
                top = stack[-1] if stack else 'ε'
                # find epsilon and input transitions
                eps_key = (current_state, 'ε', top)
                eps_key_any = (current_state, 'ε', 'ε')
                eps_options = tmap.get(eps_key, []) + tmap.get(eps_key_any, [])

                next_input = input_chars[input_index] if input_index < len(input_chars) else None
                input_options = []
                if next_input is not None:
                    key_a = (current_state, next_input, top)
                    key_b = (current_state, next_input, 'ε')
                    input_options = tmap.get(key_a, []) + tmap.get(key_b, [])

                # DPDA constraints
                if len(eps_options) > 1 or len(input_options) > 1:
                    raise HTTPException(status_code=400, detail=f"DPDA ambiguity at state '{current_state}'. Multiple transitions available.")
                if eps_options and input_options:
                    raise HTTPException(status_code=400, detail=f"DPDA ambiguity at state '{current_state}': both ε and consuming transitions defined.")

                chosen = None
                consumes = False
                if eps_options:
                    # simple cycle guard
                    keysig = (current_state, input_index, ''.join(stack))
                    if keysig in visited_eps:
                        # avoid infinite epsilon loop
                        pass
                    else:
                        visited_eps.add(keysig)
                        chosen = eps_options[0]
                        consumes = False
                elif input_options:
                    chosen = input_options[0]
                    consumes = True

                if not chosen:
                    break

                stack_before = list(stack)
                # pop
                popped = stack.pop() if stack else None
                if chosen['pop'] != 'ε':
                    if popped != chosen['pop']:
                        # invalid, rollback and fail
                        frames.append({"action": f"Transition {chosen} invalid pop", "state": {"currentState": current_state, "stackBefore": stack_before, "stackAfter": list(stack)}})
                        break
                else:
                    # we popped one but was ε instruction, restore popped to stack_before
                    if popped is not None:
                        stack.append(popped)

                # apply push
                if chosen['pop'] != 'ε':
                    # we already removed the popped symbol
                    pass
                # if pop was ε then we shouldn't have removed anything
                if chosen['push'] != 'ε':
                    for sym in reversed(list(chosen['push'])):
                        stack.append(sym)

                prev_state = current_state
                current_state = chosen['target']
                if consumes:
                    input_index += 1

                frames.append({
                    "action": f"Applied transition {prev_state} -{chosen['input']},{chosen['pop']}/{chosen['push']}-> {current_state}",
                    "state": {"currentState": current_state, "inputIndex": input_index, "unreadBuffer": "".join(input_chars[input_index:]) if input_index < len(input_chars) else "ε", "stackBefore": stack_before, "stackAfter": list(stack), "transition": chosen},
                    "metrics": {"consumedInput": consumes}
                })
                progressed = True

            is_accepted = (input_index == len(input_chars)) and (current_state in accept_states)
            frames.append({
                "action": f"Execution Complete — Machine {'ACCEPTED' if is_accepted else 'REJECTED'}",
                "state": {"currentState": current_state, "inputIndex": input_index, "unreadBuffer": "".join(input_chars[input_index:]) if input_index < len(input_chars) else "ε", "stack": list(stack), "isTerminal": True, "isAccepted": is_accepted}
            })

            return frames

        # =================================================================
        # 6. NPDA NONDETERMINISTIC PUSHDOWN AUTOMATON SOLVER
        # =================================================================
        elif pid == "npda":
            states_str = data.get("states", "q0, q1, q2, qf")
            alphabet_str = data.get("alphabet", "a, b")
            stack_alphabet_str = data.get("stackAlphabet", "Z, A")
            transitions_str = data.get("transitions", "q0,a,Z->q0,AZ\nq0,a,A->q0,AA\nq0,b,A->q1,ε\nq1,b,A->q1,ε\nq0,ε,Z->q2,Z\nq2,b,Z->qf,Z")
            start_state = data.get("startState", "q0").strip()
            accept_states_str = data.get("acceptStates", "qf")
            start_stack = data.get("startStack", "Z")
            input_string = data.get("inputString", "")

            states = [s.strip() for s in states_str.split(",") if s.strip()]
            alphabet = [a.strip() for a in alphabet_str.split(",") if a.strip()]
            stack_alphabet = [s.strip() for s in stack_alphabet_str.split(",") if s.strip()]
            accept_states = [f.strip() for f in accept_states_str.split(",") if f.strip()]

            # parse transitions
            parsed = []
            if isinstance(transitions_str, list):
                lines_iter = transitions_str
            else:
                if '\n' in transitions_str and '\\n' in transitions_str:
                    lines_iter = transitions_str.splitlines()
                elif '\\n' in transitions_str:
                    lines_iter = transitions_str.split('\\n')
                else:
                    lines_iter = transitions_str.splitlines()

            for line in lines_iter:
                if not line.strip() or "->" not in line:
                    continue
                left, right = line.split("->", 1)
                if "," not in left:
                    continue
                parts = [p.strip() for p in left.split(",")]
                if len(parts) < 3:
                    continue
                source, inp, pop = parts[0], parts[1], parts[2]
                target_and_push = [p.strip() for p in right.split(",", 1)]
                if len(target_and_push) < 1:
                    continue
                target = target_and_push[0]
                push = target_and_push[1] if len(target_and_push) > 1 else "ε"
                if inp.lower() in ["ε", "epsilon", "eps",""]:
                    inp = "ε"
                if pop.lower() in ["ε", "epsilon", "eps",""]:
                    pop = "ε"
                if push.lower() in ["ε", "epsilon", "eps",""]:
                    push = "ε"
                parsed.append({"source": source, "input": inp, "pop": pop, "target": target, "push": push})

            # group by source for quick lookup
            trans_by_source = {}
            for t in parsed:
                trans_by_source.setdefault(t["source"], []).append(t)

            trimmed = input_string.strip()
            if trimmed == "" or trimmed.lower() in ["ε", "epsilon", "eps"]:
                input_chars = []
            else:
                input_chars = [c for c in input_string if not c.isspace()]

            # branch representation: dict id -> {state, inputIndex, stack}
            branches = {"b0": {"state": start_state, "inputIndex": 0, "stack": [start_stack]}}
            frames = []

            frames.append({"action": "Initialized NPDA", "state": {"branchConfigs": branches, "unreadBuffer": input_string if input_string else "ε"}})

            # helper to apply transition to a branch and produce new branch id
            def apply_to_branch(br_id, transition, consumes):
                br = branches[br_id]
                stack_before = list(br["stack"])
                next_stack = list(stack_before)
                # pop
                if transition["pop"] != 'ε':
                    if not next_stack:
                        return None
                    top = next_stack.pop()
                    if top != transition["pop"]:
                        return None
                # push
                if transition["push"] != 'ε':
                    for s in reversed(list(transition["push"])):
                        next_stack.append(s)

                new = {"state": transition["target"], "inputIndex": br["inputIndex"] + (1 if consumes else 0), "stack": next_stack}
                return new

            # process input symbols iteratively, expanding branches
            for idx, ch in enumerate(input_chars):
                new_branches = {}
                for bid, config in list(branches.items()):
                    # find transitions from current state matching ch or epsilon
                    for t in trans_by_source.get(config["state"], []):
                        # match input
                        if t["input"] == ch or t["input"] == 'ε':
                            # check pop requirement
                            candidate = apply_to_branch(bid, t, consumes=(t["input"] == ch))
                            if candidate is not None:
                                new_id = f"b{len(new_branches)}"
                                new_branches[new_id] = candidate
                                frames.append({"action": f"Branch {bid} -> {new_id} via {t['input']},{t['pop']}/{t['push']}", "state": {"branchConfigs": {**branches, **new_branches}, "activeBranch": new_id, "consumed": t["input"] == ch}})
                if not new_branches:
                    # if no transitions matched, carry forward garbage branch
                    new_branches["garbage"] = {"state": "garbage", "inputIndex": idx + 1, "stack": []}
                branches = new_branches

            # after consuming all input, allow epsilon expansions
            expanded = True
            while expanded:
                expanded = False
                additions = {}
                for bid, config in list(branches.items()):
                    for t in trans_by_source.get(config["state"], []):
                        if t["input"] == 'ε':
                            candidate = apply_to_branch(bid, t, consumes=False)
                            if candidate is not None:
                                new_id = f"b{len(branches) + len(additions)}"
                                additions[new_id] = candidate
                                frames.append({"action": f"Epsilon expand {bid} -> {new_id} via {t['pop']}/{t['push']}", "state": {"branchConfigs": {**branches, **additions}, "activeBranch": new_id}})
                                expanded = True
                branches.update(additions)

            # final verdict
            is_accepted = any(cfg["state"] in accept_states for cfg in branches.values())
            frames.append({"action": f"Execution Complete — Machine {'ACCEPTED' if is_accepted else 'REJECTED'}", "state": {"branchConfigs": branches, "isTerminal": True, "isAccepted": is_accepted}})
            return frames
            
        # =================================================================
        # 5. AUTOMATA MINIMIZATION SOLVER HOOK
        # =================================================================
        elif pid == "dfa-minimization":
            initial_count = int(data["raw_states"])
            minimized_states = max(2, int(initial_count * 0.45))
            return {"minimized_count": minimized_states}
        
        # =================================================================
        # 6. REGULAR EXPRESSION SOLVER (Modularized)
        # =================================================================
        elif pid == "regex":
            pattern = data.get("pattern", "a(b|c)*")
            input_string = data.get("inputString", "")
            
            # Use the modular engine
            nfa = regex_engine.build_thompson_nfa(pattern)
            
            # Build transition map
            nodes = {e['source'] for e in nfa['edges']} | {e['target'] for e in nfa['edges']}
            alphabet = {e['label'] for e in nfa['edges'] if e['label'] != 'ε'}
            
            transition_map = {n: {} for n in nodes}
            for e in nfa['edges']:
                transition_map[e['source']].setdefault(e['label'], []).append(e['target'])
                
            # Inject "Garbage" state logic locally for the visualization
            nodes.add("garbage")
            transition_map["garbage"] = {sym: ["garbage"] for sym in alphabet}
            for n in list(nodes):
                if n == "garbage": continue
                for sym in alphabet:
                    if sym not in transition_map[n]:
                        transition_map[n][sym] = ["garbage"]
            
            # Helper: epsilon closure with proper state tracking
            def epsilon_closure(states_list):
                closure = set(states_list)
                stack = list(states_list)
                while stack:
                    curr = stack.pop()
                    for t in transition_map.get(curr, {}).get("ε", []):
                        if t not in closure:
                            closure.add(t)
                            stack.append(t)
                return sorted(list(closure))
            
            # Helper: expand epsilon transitions with path tracking
            def epsilon_expand_path(base_path):
                """DFS to find all epsilon-expanded variants of a base path."""
                results = [base_path]
                start_tip = base_path[-1]
                queue = [(base_path, [start_tip])]
                
                while queue:
                    current_path, local_chain_visited = queue.pop(0)
                    tip_node = current_path[-1]
                    epsilon_targets = transition_map.get(tip_node, {}).get("ε", [])
                    
                    for target in epsilon_targets:
                        occurrences = local_chain_visited.count(target)
                        if occurrences < 2:
                            extended_path = current_path + [target]
                            updated_visited = local_chain_visited + [target]
                            results.append(extended_path)
                            
                            if occurrences < 1:
                                queue.append((extended_path, updated_visited))
                
                return results
            
            # Initialize with start state epsilon closure
            start_closure = epsilon_closure([nfa['start']])
            active_paths = [[nfa['start']]]  # Start with just start state
            for path in active_paths[:]:
                for expanded in epsilon_expand_path(path):
                    if expanded not in active_paths:
                        active_paths.append(expanded)
            
            # Build configuration-based trace
            trace_timeline = []
            previous_configuration = sorted(list(set(p[-1] for p in active_paths)))
            
            steps_timeline = []
            
            # Initial frame
            all_active_edges = []
            seen_all_edges = set()
            for path in active_paths:
                for i in range(len(path) - 1):
                    key = (path[i], path[i+1], "ε")
                    if key not in seen_all_edges:
                        seen_all_edges.add(key)
                        all_active_edges.append({"source": path[i], "target": path[i+1], "label": "ε"})
            
            steps_timeline.append({
                "action": f"Start at {nfa['start']}",
                "metrics": {"traceTimeline": ""},
                "state": {
                    "nodes": sorted(list(nodes)),
                    "edges": nfa['edges'] + [{"source": n, "target": "garbage", "label": sym} 
                                            for n in nodes if n != "garbage" 
                                            for sym in alphabet if sym not in transition_map[n]],
                    "activeNodes": start_closure,
                    "activeEdges": [],
                    "allActiveEdges": all_active_edges,
                    "acceptStates": [nfa['accept']],
                    "startState": nfa['start'],
                    "branchPaths": {},
                    "isTerminal": False
                }
            })
            
            # Process each symbol
            for symbol in input_string:
                next_paths = []
                next_configuration = set()
                step_edges = []
                seen_step = set()

                def add_edge(src, tgt, lbl):
                    key = (src, tgt, lbl)
                    if key not in seen_step:
                        seen_step.add(key)
                        step_edges.append({"source": src, "target": tgt, "label": lbl})

                for path in active_paths:
                    leaf_state = path[-1]

                    if leaf_state == "garbage":
                        next_paths.append(path + ["garbage"])
                        next_configuration.add("garbage")
                    else:
                        targets = transition_map.get(leaf_state, {}).get(symbol, [])
                        if not targets:
                            next_paths.append(path + ["garbage"])
                            next_configuration.add("garbage")
                            add_edge(leaf_state, "garbage", symbol)
                        else:
                            for target in targets:
                                add_edge(leaf_state, target, symbol)
                                epsilon_options = epsilon_expand_path([target])
                                for epsilon_chain in epsilon_options:
                                    full_path = path + epsilon_chain
                                    next_paths.append(full_path)
                                    next_configuration.add(epsilon_chain[-1])
                                    for i in range(len(epsilon_chain) - 1):
                                        add_edge(epsilon_chain[i], epsilon_chain[i+1], "ε")

                next_configuration_sorted = sorted(list(next_configuration))

                # Build trace entry
                if len(trace_timeline) == 0:
                    trace_timeline.append(f"{','.join(previous_configuration)}, {symbol}->{','.join(next_configuration_sorted)}")
                else:
                    trace_timeline.append(f"{symbol}->{','.join(next_configuration_sorted)}")

                for e in step_edges:
                    key = (e["source"], e["target"], e["label"])
                    if key not in seen_all_edges:
                        seen_all_edges.add(key)
                        all_active_edges.append(e)
                
                steps_timeline.append({
                    "action": f"Consumed {symbol}",
                    "metrics": {"traceTimeline": "  ".join(trace_timeline)},
                    "state": {
                        "nodes": sorted(list(nodes)),
                        "edges": nfa['edges'] + [{"source": n, "target": "garbage", "label": sym} 
                                                for n in nodes if n != "garbage" 
                                                for sym in alphabet if sym not in transition_map[n]],
                        "activeNodes": next_configuration_sorted,
                        "activeEdges": step_edges,
                        "allActiveEdges": list(all_active_edges),
                        "acceptStates": [nfa['accept']],
                        "startState": nfa['start'],
                        "branchPaths": {},
                        "isTerminal": False
                    }
                })
                
                active_paths = next_paths
                previous_configuration = next_configuration_sorted
            
            # Terminal frame
            is_accepted = any(p[-1] == nfa['accept'] for p in active_paths)
            accepted_paths = [p for p in active_paths if p[-1] == nfa['accept']]
            rejected_paths = [p for p in active_paths if p[-1] != nfa['accept']]
            
            branch_paths = {}
            for i, path in enumerate(active_paths):
                branch_paths[f"path_{i}"] = path
            
            trace_str = "  ".join(trace_timeline) if trace_timeline else ""
            
            steps_timeline.append({
                "action": f"Execution Complete — {'ACCEPTED' if is_accepted else 'REJECTED'}",
                "metrics": {"traceTimeline": trace_str},
                "state": {
                    "nodes": sorted(list(nodes)),
                    "edges": nfa['edges'] + [{"source": n, "target": "garbage", "label": sym} 
                                            for n in nodes if n != "garbage" 
                                            for sym in alphabet if sym not in transition_map[n]],
                    "activeNodes": previous_configuration,
                    "activeEdges": [],
                    "allActiveEdges": list(all_active_edges),
                    "acceptStates": [nfa['accept']],
                    "acceptedEndpointStates": sorted(list(set(p[-1] for p in accepted_paths if p[-1] != "garbage"))),
                    "rejectedEndpointStates": sorted(list(set(p[-1] for p in rejected_paths))),
                    "startState": nfa['start'],
                    "branchPaths": branch_paths,
                    "isTerminal": True,
                    "isAccepted": is_accepted
                }
            })
            
            return steps_timeline
        
        elif pid == "convex-hull":
            points = data.get("points", [])
            algorithm = data.get("algorithm", "graham-scan")

            try:
                return convexHull_engine.ConvexHullSolver.solve(points, algorithm)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            
        elif pid == "knapsack":
            inputs = data
            capacity_raw = inputs.get("capacity")
            if capacity_raw is None:
                raise HTTPException(status_code=400, detail="Missing expected variable key: 'capacity'")

            raw_items = inputs.get("items", "")
            items = []
            for index, line in enumerate(str(raw_items).strip().splitlines()):
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 3:
                    continue

                name, weight_raw, value_raw = parts
                try:
                    weight_value = float(weight_raw)
                    value_value = float(value_raw)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid knapsack item '{line}': weight and value must be numeric."
                    )

                if not weight_value.is_integer() or not value_value.is_integer():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid knapsack item '{line}': weight and value must be integers for the DP table visualization."
                    )

                items.append({
                    "name": name or f"Item {index + 1}",
                    "weight": int(weight_value),
                    "value": int(value_value),
                })

            if not items:
                raise HTTPException(status_code=400, detail="Provide at least one item in the format: name, weight, value")

            try:
                capacity_value = float(capacity_raw)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="Capacity must be numeric")

            if not capacity_value.is_integer():
                raise HTTPException(status_code=400, detail="Capacity must be an integer for the DP table visualization")

            max_val, steps = knapsack_engine.KnapsackSolver.solve(items, int(capacity_value))

            return steps
        elif pid == "subset-sum":
            # Support reduction requests originating from knapsack problem
            if payload.transformation and isinstance(payload.transformation, dict) and payload.transformation.get("id") == "knapsack-to-subset-sum":
                # Expect knapsack-style inputs: items (textarea of name, weight, value) and capacity
                raw_items = data.get("items", "")
                capacity_raw = data.get("capacity")
                items = []
                can_reduce = True
                numbers = []
                for index, line in enumerate(str(raw_items).strip().splitlines()):
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 3:
                        continue
                    name, weight_raw, value_raw = parts
                    try:
                        weight_value = float(weight_raw)
                        value_value = float(value_raw)
                    except ValueError:
                        can_reduce = False
                        break
                    if not weight_value.is_integer() or not value_value.is_integer():
                        can_reduce = False
                        break
                    if int(weight_value) != int(value_value):
                        # Classic reduction requires weight == value for each item
                        can_reduce = False
                    numbers.append(int(value_value))

                try:
                    target_value = float(capacity_raw)
                except Exception:
                    target_value = None

                steps = []
                if not can_reduce or target_value is None or not float(target_value).is_integer():
                    steps.append({
                        "action": "Knapsack → Subset Sum reduction",
                        "metrics": {"canReduce": False},
                        "state": {"message": "Reduction not available: requires integer items with weight == value and integer capacity."},
                    })
                else:
                    steps.append({
                        "action": "Knapsack → Subset Sum reduction",
                        "metrics": {"canReduce": True},
                        "state": {"numbers": numbers, "target": int(target_value)},
                    })

                # Return the reduction as a named conversion payload consumed by the frontend
                return {"transformation": {"id": payload.transformation.get("id"), "steps": steps}}

            # Otherwise perform standard subset-sum DP solve
            raw_items = data.get("items", "")
            target_raw = data.get("target_sum")

            # Accept flexible item list formats: one-per-line, or a single line with spaces/commas
            numbers = []
            raw_str = str(raw_items).strip()
            if raw_str == "":
                lines = []
            else:
                lines = raw_str.splitlines()

            # If a single line contains commas or spaces, split into tokens
            if len(lines) == 1 and (',' in lines[0] or ' ' in lines[0].strip()):
                import re
                tokens = [t for t in re.split(r'[\s,]+', lines[0].strip()) if t]
                lines = tokens

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    val = float(line)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid item '{line}': must be numeric")
                if not val.is_integer() or val < 0:
                    raise HTTPException(status_code=400, detail=f"Invalid item '{line}': must be a non-negative integer")
                numbers.append(int(val))

            if target_raw is None:
                raise HTTPException(status_code=400, detail="Missing expected variable key: 'target_sum'")
            try:
                target_value = float(target_raw)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="Target sum must be numeric")
            if not float(target_value).is_integer() or target_value < 0:
                raise HTTPException(status_code=400, detail="Target sum must be a non-negative integer")

            reachable, steps = subset_sum_engine.SubsetSumSolver.solve(numbers, int(target_value))
            return steps

        elif pid in ("pump-scheduling", "optimal-pump-scheduling"):
            mode = payload.mode or data.get("mode")
            return optimal_pump_scheduling_engine.OptimalPumpSchedulingSolver.solve(data, mode)

        elif pid == "min-st-cut-dual":
            return min_st_cut_dual_engine.MinSTCutDualSolver.solve(data)

        elif pid == "water-allocation":
            _detail = payload.detail
            t = payload.transformation
            if t and isinstance(t, dict) and t.get("id") == "upgrade-pipe":
                return water_network_engine.WaterNetworkEngine.solve_upgrade(
                    {**data, "transformation": t}, detail=_detail
                )
            phase = data.get("phase", "physics")
            if phase == "selection":
                return farm_selection_engine.FarmSelectionEngine.solve(data, detail=_detail)
            if phase == "scheduling":
                return water_scheduling_engine.WaterSchedulingEngine.solve(data, detail=_detail)
            return water_network_engine.WaterNetworkEngine.solve(data, detail=_detail)

        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Problem context token '{pid}' has no registered execution hook mapping."
            )
            
    except HTTPException as he:
        raise he
            
    except KeyError as ke:
        raise HTTPException(
            status_code=400, 
            detail=f"Execution context structural payload error. Missing expected variable key: {str(ke)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Computational core processing failure exception trace: {str(e)}"
        )
        
def run_nfa_simulation(input_string, states, transition_map, start_state, accept_states, canvas_nodes, canvas_edges):
    all_active_edges = []
    
    def get_epsilon_closure_with_edges(states_list):
        closure = set(states_list)
        edges_traversed = []
        stack = list(states_list)
        while stack:
            curr = stack.pop()
            for t in transition_map.get(curr, {}).get("ε", []):
                if t not in closure:
                    closure.add(t)
                    stack.append(t)
                    edges_traversed.append({"source": curr, "target": t, "label": "ε"})
        return sorted(list(closure)), edges_traversed

    def get_step_data(current_states, char):
        next_states = set()
        step_edges = []
        for s in current_states:
            for t in transition_map.get(s, {}).get(char, []):
                next_states.add(t)
                step_edges.append({"source": s, "target": t, "label": char})
        closure, eps_edges = get_epsilon_closure_with_edges(next_states)
        return closure, step_edges + eps_edges

    steps = []
    current_active, initial_eps = get_epsilon_closure_with_edges([start_state])
    all_active_edges.extend(initial_eps)
    
    # 1. INITIAL FRAME (No Verdict Coloring)
    steps.append({
        "action": f"Start at {start_state}",
        "state": {
            "nodes": canvas_nodes, "edges": canvas_edges,
            "activeNodes": current_active, "activeEdges": initial_eps, 
            "allActiveEdges": list(all_active_edges),
            "acceptStates": [], # Empty so UI doesn't color prematurely
            "startState": start_state, "isTerminal": False
        }
    })

    # 2. INTERMEDIATE FRAMES (No Verdict Coloring)
    for i, char in enumerate(input_string):
        next_active, step_edges = get_step_data(current_active, char)
        all_active_edges.extend(step_edges)
        
        steps.append({
            "action": f"Consumed {char}",
            "state": {
                "nodes": canvas_nodes, "edges": canvas_edges,
                "activeNodes": next_active, "activeEdges": step_edges,
                "allActiveEdges": list(all_active_edges),
                "acceptStates": [], # Empty so UI doesn't color prematurely
                "startState": start_state, "isTerminal": False
            }
        })
        current_active = next_active

    # 3. TERMINAL FRAME (Verdict Coloring)
    is_accepted = any(s in accept_states for s in current_active)
    
    steps.append({
        "action": f"Execution Complete — {'ACCEPTED' if is_accepted else 'REJECTED'}",
        "state": {
            "nodes": canvas_nodes, 
            "edges": canvas_edges,
            "activeNodes": current_active, 
            "activeEdges": [], 
            "allActiveEdges": list(all_active_edges),
            "acceptStates": accept_states, # SEND HERE ONLY
            "startState": start_state,
            "acceptedEndpointStates": [s for s in current_active if s in accept_states],
            "rejectedEndpointStates": [s for s in current_active if s not in accept_states],
            "isTerminal": True, 
            "isAccepted": is_accepted
        }
    })
    return steps
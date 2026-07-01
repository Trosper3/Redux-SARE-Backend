# automata_utils.py

def build_graph_data(states, transition_map):
    """Generates canvas nodes and edges consistent with your NFA solver."""
    canvas_nodes = list(states)
    canvas_edges = []
    for source, paths in transition_map.items():
        for symbol, targets in paths.items():
            for target in targets:
                canvas_edges.append({"source": source, "target": target, "label": symbol})
    return canvas_nodes, canvas_edges

def run_nfa_simulation(input_string, states, transition_map, start_state, accept_states, canvas_nodes, canvas_edges):
    def epsilon_closure(states_list):
        closure = set(states_list)
        stack = list(states_list)
        while stack:
            curr = stack.pop()
            for t in transition_map.get(curr, {}).get("ε", []):
                if t not in closure:
                    closure.add(t); stack.append(t)
        return sorted(list(closure))

    def get_step_data(current_states, char):
        next_states = set()
        step_edges = []
        for s in current_states:
            for t in transition_map.get(s, {}).get(char, []):
                next_states.add(t)
                step_edges.append({"source": s, "target": t, "label": char})
        
        # Expand epsilon closure after symbol transition
        closure, eps_edges = epsilon_closure_with_edges(next_states)
        return closure, step_edges + eps_edges

    def epsilon_closure_with_edges(states_list):
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

    steps = []
    current_active, initial_eps = epsilon_closure_with_edges([start_state])
    all_active_edges = list(initial_eps)
    
    steps.append({
        "action": f"Start at {start_state}",
        "state": {
            "nodes": canvas_nodes, "edges": canvas_edges,
            "activeNodes": current_active, "activeEdges": initial_eps, 
            "allActiveEdges": all_active_edges,
            "acceptStates": [], # Empty during traversal
            "startState": start_state, "isTerminal": False
        }
    })

    for char in input_string:
        next_active, step_edges = get_step_data(current_active, char)
        all_active_edges.extend(step_edges)
        
        steps.append({
            "action": f"Consumed {char}",
            "state": {
                "nodes": canvas_nodes, "edges": canvas_edges,
                "activeNodes": next_active, "activeEdges": step_edges,
                "allActiveEdges": list(all_active_edges),
                "acceptStates": [], # Empty during traversal
                "startState": start_state, "isTerminal": False
            }
        })
        current_active = next_active

    # Terminal Frame
    is_accepted = any(s in accept_states for s in current_active)
    steps.append({
        "action": f"Execution Complete — {'ACCEPTED' if is_accepted else 'REJECTED'}",
        "state": {
            "nodes": canvas_nodes, "edges": canvas_edges,
            "activeNodes": current_active, "activeEdges": [], 
            "allActiveEdges": list(all_active_edges),
            "acceptStates": accept_states, 
            "startState": start_state,
            "acceptedEndpointStates": [s for s in current_active if s in accept_states],
            "rejectedEndpointStates": [s for s in current_active if s not in accept_states],
            "isTerminal": True, "isAccepted": is_accepted
        }
    })
    return steps
import itertools
import math
import re
from typing import Any, Dict, List


class SATSolver:
    @staticmethod
    def _normalize_formula(formula: Any) -> str:
        if formula is None:
            return ""
        return str(formula)

    @staticmethod
    def _normalize_literal(literal: str) -> str:
        token = literal.strip()
        if not token:
            return ""
        token = token.replace("¬", "!").replace("~", "!")
        token = re.sub(r"^NOT\s+", "!", token, flags=re.IGNORECASE)
        token = re.sub(r"^!+", "!", token)
        return token

    @staticmethod
    def _parse_formula(formula: Any) -> List[List[str]]:
        raw = SATSolver._normalize_formula(formula)
        if not raw.strip():
            return []

        raw = raw.replace("(", " ").replace(")", " ")
        raw = raw.replace("∧", " AND ").replace("&&", " AND ")
        clause_tokens = re.split(r"\bAND\b", raw, flags=re.IGNORECASE)
        clauses: List[List[str]] = []

        for clause_text in clause_tokens:
          clause = clause_text.strip()
          if not clause:
              continue
          clause = clause.replace("∨", " OR ").replace("||", " OR ")
          clause = clause.replace("|", " OR ").replace(",", " OR ")
          literals = []
          for part in re.split(r"\bOR\b", clause, flags=re.IGNORECASE):
              literal = SATSolver._normalize_literal(part)
              if literal:
                  literals.append(literal)
          if literals:
              clauses.append(literals)

        return clauses

    @staticmethod
    def _extract_variables(clauses: List[List[str]]) -> List[str]:
        variables = []
        seen = set()
        for clause in clauses:
            for literal in clause:
                variable = literal[1:] if literal.startswith("!") else literal
                if variable and variable not in seen:
                    seen.add(variable)
                    variables.append(variable)
        return variables

    @staticmethod
    def _resolve_bruteforce_variables(clauses: List[List[str]], declared_count: int) -> List[str]:
        variables = []
        seen = set()

        for index in range(max(0, declared_count)):
            variable = f"x{index + 1}"
            if variable not in seen:
                seen.add(variable)
                variables.append(variable)

        for variable in SATSolver._extract_variables(clauses):
            if variable not in seen:
                seen.add(variable)
                variables.append(variable)

        return variables

    @staticmethod
    def _evaluate_literal(literal: str, assignment: Dict[str, bool]) -> bool:
        negated = literal.startswith("!")
        variable = literal[1:] if negated else literal
        value = bool(assignment.get(variable, False))
        return not value if negated else value

    @staticmethod
    def _evaluate_clauses(clauses: List[List[str]], assignment: Dict[str, bool]) -> List[bool]:
        return [any(SATSolver._evaluate_literal(literal, assignment) for literal in clause) for clause in clauses]

    @staticmethod
    def _build_state(
        variables: List[str],
        clauses: List[List[str]],
        assignment: Dict[str, bool],
        clause_results: List[bool],
        *,
        current_clause_index: int | None = None,
        current_literal: str | None = None,
        is_terminal: bool = False,
        is_accepted: bool = False,
        witness: Dict[str, bool] | None = None,
    ) -> Dict[str, Any]:
        return {
            "variables": variables,
            "clauses": clauses,
            "assignment": assignment,
            "clauseResults": clause_results,
            "currentClauseIndex": current_clause_index,
            "currentLiteral": current_literal,
            "witness": witness,
            "isTerminal": is_terminal,
            "isAccepted": is_accepted,
        }

    @staticmethod
    def _build_clique_reduction(clauses: List[List[str]]) -> Dict[str, Any]:
        if len(clauses) == 0:
            return {
                "canReduce": False,
                "message": "Reduction cannot be made because there are no clauses.",
                "vertices": [],
                "edges": [],
                "requiredCliqueSize": 0,
            }

        if any(len(clause) == 0 for clause in clauses):
            return {
                "canReduce": False,
                "message": "Reduction cannot be made because one or more clauses are empty.",
                "vertices": [],
                "edges": [],
                "requiredCliqueSize": len(clauses),
            }

        vertices: List[Dict[str, Any]] = []
        for clause_index, clause in enumerate(clauses):
            for literal_index, literal in enumerate(clause):
                vertices.append({
                    "id": f"c{clause_index}l{literal_index}",
                    "clauseIndex": clause_index,
                    "literal": literal,
                    "x": 100 + clause_index * 180,
                    "y": 120 + literal_index * 55,
                })

        def compatible(left: str, right: str) -> bool:
            left_negated = left.startswith("!")
            right_negated = right.startswith("!")
            left_variable = left[1:] if left_negated else left
            right_variable = right[1:] if right_negated else right
            return left_variable != right_variable or left_negated == right_negated

        edges: List[Dict[str, str]] = []
        for left_index, left in enumerate(vertices):
            for right in vertices[left_index + 1 :]:
                if left["clauseIndex"] == right["clauseIndex"]:
                    continue
                if compatible(str(left["literal"]), str(right["literal"])):
                    edges.append({"source": left["id"], "target": right["id"]})

        return {
            "canReduce": True,
            "message": f"The formula is satisfiable if and only if this graph has a clique of size {len(clauses)}.",
            "vertices": vertices,
            "edges": edges,
            "requiredCliqueSize": len(clauses),
        }

    @staticmethod
    def solve(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        formula = inputs.get("formula", "")
        clauses = SATSolver._parse_formula(formula)
        inferred_variables = SATSolver._extract_variables(clauses)
        declared_count_raw = inputs.get("num_variables", len(inferred_variables))
        try:
            declared_count = int(declared_count_raw)
        except (TypeError, ValueError):
            declared_count = len(inferred_variables)
        algorithm = str(inputs.get("algorithm") or "sat-bruteforce").strip().lower()

        if algorithm not in {"sat-bruteforce", "sat-dpll"}:
            raise ValueError(f"Unknown SAT algorithm: {algorithm}")

        reduction = SATSolver._build_clique_reduction(clauses)
        steps: List[Dict[str, Any]] = []
        variables = inferred_variables if algorithm == "sat-dpll" else SATSolver._resolve_bruteforce_variables(clauses, declared_count)

        steps.append({
            "action": f"Initialized SAT search using {algorithm.replace('sat-', '').replace('-', ' ')}",
            "metrics": {
                "algorithm": algorithm,
                "variables": len(variables),
                "declaredVariables": declared_count,
                "clauses": len(clauses),
            },
            "state": SATSolver._build_state(variables, clauses, {}, [False for _ in clauses], is_terminal=False),
        })

        witness: Dict[str, bool] | None = None

        if algorithm == "sat-bruteforce":
            total_assignments = 1 << len(variables)
            sat_count = 0

            for mask in range(total_assignments):
                assignment = {variable: bool(mask & (1 << index)) for index, variable in enumerate(variables)}
                clause_results = SATSolver._evaluate_clauses(clauses, assignment)
                is_sat = all(clause_results)
                if is_sat:
                    sat_count += 1
                    if witness is None:
                        witness = assignment.copy()

                steps.append({
                    "action": f"Evaluated assignment #{mask + 1}: " + ", ".join(f"{variable}={'T' if value else 'F'}" for variable, value in assignment.items()),
                    "metrics": {
                        "algorithm": algorithm,
                        "assignment_index": mask + 1,
                        "satisfies_formula": is_sat,
                        "satisfied_clauses": sum(1 for value in clause_results if value),
                    },
                    "state": SATSolver._build_state(
                        variables,
                        clauses,
                        assignment,
                        clause_results,
                        witness=witness,
                        is_terminal=False,
                        is_accepted=is_sat,
                    ),
                })

            steps.append({
                "action": "SAT brute-force search complete",
                "metrics": {
                    "algorithm": algorithm,
                    "satisfying_assignments": sat_count,
                    "witness": witness or {},
                },
                "state": SATSolver._build_state(
                    variables,
                    clauses,
                    witness or {},
                    SATSolver._evaluate_clauses(clauses, witness or {}),
                    witness=witness,
                    is_terminal=True,
                    is_accepted=witness is not None,
                ),
            })

        else:
            assignment: Dict[str, bool] = {}

            def dpll(clause_set: List[List[str]], depth: int) -> bool:
                nonlocal witness

                if not clause_set:
                    witness = assignment.copy()
                    steps.append({
                        "action": f"Found satisfying assignment at depth {depth}",
                        "metrics": {"algorithm": algorithm, "depth": depth, "assignment": assignment.copy()},
                        "state": SATSolver._build_state(variables, clauses, assignment.copy(), SATSolver._evaluate_clauses(clauses, assignment), witness=witness, is_terminal=False, is_accepted=True),
                    })
                    return True

                if any(len(clause) == 0 for clause in clause_set):
                    steps.append({
                        "action": f"Conflict detected at depth {depth}",
                        "metrics": {"algorithm": algorithm, "depth": depth},
                        "state": SATSolver._build_state(variables, clauses, assignment.copy(), SATSolver._evaluate_clauses(clauses, assignment), witness=witness, is_terminal=False, is_accepted=False),
                    })
                    return False

                unit_literal = next((clause[0] for clause in clause_set if len(clause) == 1), None)
                if unit_literal:
                    variable = unit_literal[1:] if unit_literal.startswith("!") else unit_literal
                    value = not unit_literal.startswith("!")
                    assignment[variable] = value
                    steps.append({
                        "action": f"Unit propagate {variable} = {'T' if value else 'F'}",
                        "metrics": {"algorithm": algorithm, "depth": depth, "variable": variable, "value": value},
                        "state": SATSolver._build_state(variables, clauses, assignment.copy(), SATSolver._evaluate_clauses(clauses, assignment), current_literal=unit_literal, witness=witness, is_terminal=False),
                    })
                    reduced = []
                    for clause in clause_set:
                        if unit_literal in clause:
                            continue
                        reduced_clause = [literal for literal in clause if literal != ("!" + variable if value else variable)]
                        reduced.append(reduced_clause)
                    if dpll(reduced, depth + 1):
                        return True
                    assignment.pop(variable, None)
                    return False

                chosen_literal = clause_set[0][0]
                variable = chosen_literal[1:] if chosen_literal.startswith("!") else chosen_literal
                for value in [True, False]:
                    assignment[variable] = value
                    steps.append({
                        "action": f"Branch on {variable} = {'T' if value else 'F'}",
                        "metrics": {"algorithm": algorithm, "depth": depth, "variable": variable, "value": value},
                        "state": SATSolver._build_state(variables, clauses, assignment.copy(), SATSolver._evaluate_clauses(clauses, assignment), current_literal=chosen_literal, witness=witness, is_terminal=False),
                    })

                    reduced = []
                    for clause in clause_set:
                        if (value and variable in clause) or (not value and f"!{variable}" in clause):
                            continue
                        reduced_clause = [literal for literal in clause if literal != (f"!{variable}" if value else variable)]
                        reduced.append(reduced_clause)

                    if dpll(reduced, depth + 1):
                        return True

                    assignment.pop(variable, None)
                    steps.append({
                        "action": f"Backtrack on {variable}",
                        "metrics": {"algorithm": algorithm, "depth": depth, "variable": variable},
                        "state": SATSolver._build_state(variables, clauses, assignment.copy(), SATSolver._evaluate_clauses(clauses, assignment), witness=witness, is_terminal=False),
                    })

                return False

            dpll([clause[:] for clause in clauses], 0)

            steps.append({
                "action": "SAT DPLL search complete",
                "metrics": {
                    "algorithm": algorithm,
                    "witness": witness or {},
                },
                "state": SATSolver._build_state(
                    variables,
                    clauses,
                    witness or {},
                    SATSolver._evaluate_clauses(clauses, witness or {}),
                    witness=witness,
                    is_terminal=True,
                    is_accepted=witness is not None,
                ),
            })

        response: Dict[str, Any] = {"steps": steps}
        if reduction:
            response["selectedTransformation"] = {
                "id": "sat-to-clique",
                "steps": [
                    {
                        "action": "SAT to Clique reduction",
                        "metrics": {"canReduce": reduction["canReduce"]},
                        "state": reduction,
                    },
                    {
                        "action": "Reduction ready",
                        "metrics": {"requiredCliqueSize": reduction["requiredCliqueSize"]},
                        "state": reduction,
                    },
                ],
            }

        return response
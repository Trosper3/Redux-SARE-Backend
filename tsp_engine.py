import math
import re
from itertools import permutations
from typing import Any, Dict, List, Tuple


class TSPSolver:
    @staticmethod
    def _normalize_cities(cities: Any) -> List[Dict[str, Any]]:
        if isinstance(cities, str):
            raw_lines = cities.splitlines()
        elif isinstance(cities, (list, tuple)):
            raw_lines = list(cities)
        else:
            raw_lines = []

        normalized: List[Dict[str, Any]] = []
        seen = set()

        for index, raw_city in enumerate(raw_lines):
            if isinstance(raw_city, dict):
                name = str(raw_city.get("name") or f"C{index + 1}").strip()
                x_raw = raw_city.get("x")
                y_raw = raw_city.get("y")
            else:
                text = str(raw_city).strip()
                if not text:
                    continue
                parts = [part.strip() for part in text.split(",")]
                if len(parts) < 3:
                    continue
                name, x_raw, y_raw = parts[0], parts[1], parts[2]

            if not name or name in seen:
                continue

            try:
                x_val = float(x_raw)
                y_val = float(y_raw)
            except (TypeError, ValueError):
                continue

            seen.add(name)
            normalized.append({"id": name, "x": x_val, "y": y_val})

        return normalized

    @staticmethod
    def _distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
        return math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))

    @staticmethod
    def _build_edges(cities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        edges: List[Dict[str, str]] = []
        for index, left in enumerate(cities):
            for right in cities[index + 1 :]:
                distance = TSPSolver._distance(left, right)
                edges.append({
                    "source": left["id"],
                    "target": right["id"],
                    "label": f"{distance:.1f}",
                })
        return edges

    @staticmethod
    def _tour_distance(cities: List[Dict[str, Any]], tour: List[int]) -> float:
        return sum(TSPSolver._distance(cities[tour[index]], cities[tour[index + 1]]) for index in range(len(tour) - 1))

    @staticmethod
    def _route_edges(cities: List[Dict[str, Any]], tour: List[int]) -> List[Dict[str, str]]:
        route_edges: List[Dict[str, str]] = []
        for index in range(len(tour) - 1):
            left = cities[tour[index]]
            right = cities[tour[index + 1]]
            route_edges.append({
                "source": left["id"],
                "target": right["id"],
                "label": f"{TSPSolver._distance(left, right):.1f}",
            })
        return route_edges

    @staticmethod
    def _build_state(
        cities: List[Dict[str, Any]],
        edges: List[Dict[str, str]],
        tour: List[int],
        best_tour: List[int],
        *,
        current_edge: List[str] | None = None,
        is_terminal: bool = False,
        is_accepted: bool = False,
        best_distance: float | None = None,
        current_distance: float | None = None,
    ) -> Dict[str, Any]:
        route_nodes = [cities[index]["id"] for index in tour]
        best_nodes = [cities[index]["id"] for index in best_tour]
        current_route_edges = TSPSolver._route_edges(cities, tour) if len(tour) > 1 else []
        best_route_edges = TSPSolver._route_edges(cities, best_tour) if len(best_tour) > 1 else []

        return {
            "points": cities,
            "edges": edges,
            "activeNodes": route_nodes,
            "activeEdges": current_route_edges,
            "allActiveEdges": best_route_edges,
            "currentRoute": route_nodes,
            "bestRoute": best_nodes,
            "candidateEdge": current_edge,
            "currentDistance": current_distance,
            "bestDistance": best_distance,
            "isTerminal": is_terminal,
            "isAccepted": is_accepted,
        }

    @staticmethod
    def solve(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        cities = TSPSolver._normalize_cities(inputs.get("cities", []))
        if len(cities) < 3:
            raise ValueError("Provide at least 3 cities to solve TSP.")

        algorithm = str(inputs.get("algorithm") or "tsp-nearest-neighbor").strip().lower()
        if algorithm == "tsp-brute-force" and len(cities) > 8:
            raise ValueError("Brute force TSP is limited to 8 cities so the step trace stays readable.")
        if algorithm not in {"tsp-nearest-neighbor", "tsp-brute-force"}:
            raise ValueError(f"Unknown TSP algorithm: {algorithm}")

        start_city_name = str(inputs.get("startCity") or cities[0]["id"]).strip()
        start_index = next((index for index, city in enumerate(cities) if city["id"] == start_city_name), 0)
        edges = TSPSolver._build_edges(cities)

        steps: List[Dict[str, Any]] = []
        steps.append({
            "action": f"Initialized TSP search using {algorithm.replace('tsp-', '').replace('-', ' ')}",
            "metrics": {
                "cities": len(cities),
                "algorithm": algorithm,
                "start_city": cities[start_index]["id"],
            },
            "state": TSPSolver._build_state(cities, edges, [start_index], [start_index], current_edge=None, best_distance=0.0, current_distance=0.0),
        })

        if algorithm == "tsp-nearest-neighbor":
            unvisited = set(range(len(cities)))
            unvisited.remove(start_index)
            tour = [start_index]
            current_index = start_index
            total_distance = 0.0
            step_count = 0

            while unvisited:
                nearest_index = min(unvisited, key=lambda candidate: TSPSolver._distance(cities[current_index], cities[candidate]))
                segment_distance = TSPSolver._distance(cities[current_index], cities[nearest_index])
                total_distance += segment_distance
                tour.append(nearest_index)
                unvisited.remove(nearest_index)

                route_edges = TSPSolver._route_edges(cities, tour)
                steps.append({
                    "action": f"Step {step_count + 1}: visit {cities[nearest_index]['id']} from {cities[current_index]['id']}",
                    "metrics": {
                        "algorithm": algorithm,
                        "step": step_count + 1,
                        "current_city": cities[current_index]["id"],
                        "next_city": cities[nearest_index]["id"],
                        "segment_distance": round(segment_distance, 1),
                        "tour_distance": round(total_distance, 1),
                        "visited_count": len(tour),
                    },
                    "state": TSPSolver._build_state(
                        cities,
                        edges,
                        tour[:],
                        tour[:],
                        current_edge=[cities[current_index]["id"], cities[nearest_index]["id"]],
                        current_distance=total_distance,
                        best_distance=total_distance,
                    ),
                    "highlights": {
                        "elements": [cities[index]["id"] for index in tour],
                        "connections": [[edge["source"], edge["target"]] for edge in route_edges],
                    },
                })

                current_index = nearest_index
                step_count += 1

            return_distance = TSPSolver._distance(cities[current_index], cities[start_index])
            total_distance += return_distance
            tour.append(start_index)

            final_route_edges = TSPSolver._route_edges(cities, tour)
            steps.append({
                "action": f"Return to {cities[start_index]['id']} to close the tour",
                "metrics": {
                    "algorithm": algorithm,
                    "tour_distance": round(total_distance, 1),
                    "visited_count": len(cities),
                    "tour": [cities[index]["id"] for index in tour],
                },
                "state": TSPSolver._build_state(
                    cities,
                    edges,
                    tour[:],
                    tour[:],
                    current_edge=[cities[current_index]["id"], cities[start_index]["id"]],
                    is_terminal=True,
                    is_accepted=True,
                    current_distance=total_distance,
                    best_distance=total_distance,
                ),
                "highlights": {
                    "elements": [city["id"] for city in cities],
                    "connections": [[edge["source"], edge["target"]] for edge in final_route_edges],
                },
            })

            return steps

        other_indices = [index for index in range(len(cities)) if index != start_index]
        permutations_evaluated = 0
        best_tour = [start_index]
        best_length = math.inf

        for permutation in permutations(other_indices):
            permutations_evaluated += 1
            tour = [start_index, *permutation, start_index]
            current_length = TSPSolver._tour_distance(cities, tour)

            route_edges = TSPSolver._route_edges(cities, tour)
            steps.append({
                "action": f"Evaluating tour {permutations_evaluated}: {' → '.join(cities[index]['id'] for index in tour)}",
                "metrics": {
                    "algorithm": algorithm,
                    "permutation": permutations_evaluated,
                    "current_length": round(current_length, 1),
                    "best_length": round(best_length, 1) if math.isfinite(best_length) else None,
                },
                "state": TSPSolver._build_state(
                    cities,
                    edges,
                    tour[:],
                    best_tour[:],
                    current_edge=[cities[tour[-2]]["id"], cities[tour[-1]]["id"]],
                    current_distance=current_length,
                    best_distance=best_length if math.isfinite(best_length) else None,
                ),
                "highlights": {
                    "elements": [cities[index]["id"] for index in tour[:-1]],
                    "connections": [[edge["source"], edge["target"]] for edge in route_edges],
                },
            })

            if current_length < best_length:
                best_length = current_length
                best_tour = tour[:]
                steps.append({
                    "action": f"Found new best tour: {' → '.join(cities[index]['id'] for index in best_tour)}",
                    "metrics": {
                        "algorithm": algorithm,
                        "best_length": round(best_length, 1),
                        "permutation": permutations_evaluated,
                        "tour": [cities[index]["id"] for index in best_tour],
                    },
                    "state": TSPSolver._build_state(
                        cities,
                        edges,
                        best_tour[:],
                        best_tour[:],
                        current_edge=[cities[best_tour[-2]]["id"], cities[best_tour[-1]]["id"]],
                        current_distance=best_length,
                        best_distance=best_length,
                    ),
                    "highlights": {
                        "elements": [cities[index]["id"] for index in best_tour[:-1]],
                        "connections": [[edge["source"], edge["target"]] for edge in route_edges],
                    },
                })

        steps.append({
            "action": f"TSP search complete using {algorithm.replace('tsp-', '').replace('-', ' ')}",
            "metrics": {
                "algorithm": algorithm,
                "best_length": round(best_length, 1),
                "permutations_evaluated": permutations_evaluated,
                "tour": [cities[index]["id"] for index in best_tour],
            },
            "state": TSPSolver._build_state(
                cities,
                edges,
                best_tour[:],
                best_tour[:],
                current_edge=None,
                is_terminal=True,
                is_accepted=True,
                current_distance=best_length,
                best_distance=best_length,
            ),
            "highlights": {
                "elements": [city["id"] for city in cities],
                "connections": [[edge["source"], edge["target"]] for edge in TSPSolver._route_edges(cities, best_tour)],
            },
        })

        return steps
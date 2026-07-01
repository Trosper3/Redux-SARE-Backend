"""Convex hull solver utilities for the backend API.

The backend returns standardized step frames that the generic frontend
workspace can render directly in the geometry canvas.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Point = Dict[str, Any]
Frame = Dict[str, Any]


class ConvexHullSolver:
    @staticmethod
    def normalize_points(points: Any) -> List[Point]:
        normalized: List[Point] = []

        raw_items: List[Any]
        if isinstance(points, str):
            raw_items = [line.strip() for line in points.splitlines() if line.strip()]
        elif isinstance(points, list):
            raw_items = list(points)
        else:
            return normalized

        for index, raw in enumerate(raw_items):
            identifier = f"P{index + 1}"
            x = y = None

            if isinstance(raw, dict):
                identifier = str(raw.get("id") or raw.get("label") or identifier)
                x = raw.get("x")
                y = raw.get("y")
            elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                x, y = raw[0], raw[1]
            elif isinstance(raw, str):
                tokens = [token.strip() for token in raw.replace(",", " ").split() if token.strip()]
                if len(tokens) >= 3 and not ConvexHullSolver._is_number(tokens[0]):
                    identifier = tokens[0]
                    x, y = tokens[1], tokens[2]
                elif len(tokens) >= 2:
                    x, y = tokens[0], tokens[1]

            if x is None or y is None:
                continue

            try:
                x_value = float(x)
                y_value = float(y)
            except (TypeError, ValueError):
                continue

            normalized.append({"id": identifier, "x": x_value, "y": y_value, "label": identifier})

        return normalized

    @staticmethod
    def solve(points: Any, algorithm: str = "graham-scan") -> List[Frame]:
        normalized_points = ConvexHullSolver.normalize_points(points)
        normalized_algorithm = (algorithm or "graham-scan").strip().lower()

        if normalized_algorithm in {"brute", "brute-force", "bruteforce", "brute_force"}:
            normalized_algorithm = "brute-force"
        elif normalized_algorithm in {"graham", "graham-scan", "grahamscan"}:
            normalized_algorithm = "graham-scan"
        elif normalized_algorithm in {"divide", "divide-conquer", "divide-and-conquer", "divide_conquer", "divideconquer"}:
            normalized_algorithm = "divide-conquer"

        if len(normalized_points) < 3:
            return [
                ConvexHullSolver._frame(
                    "Not enough points for a hull",
                    normalized_points,
                    hull=normalized_points,
                    metrics={"iterations": 0, "quality": 1, "algorithm": normalized_algorithm, "pointCount": len(normalized_points)},
                )
            ]

        if normalized_algorithm == "brute-force":
            return ConvexHullSolver._brute_force(normalized_points)
        if normalized_algorithm == "graham-scan":
            return ConvexHullSolver._graham_scan(normalized_points)
        if normalized_algorithm == "divide-conquer":
            return ConvexHullSolver._divide_and_conquer(normalized_points)

        raise ValueError(f"Unknown convex hull algorithm: {algorithm}")

    @staticmethod
    def _is_number(value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _cross(o: Point, a: Point, b: Point) -> float:
        return (a["x"] - o["x"]) * (b["y"] - o["y"]) - (a["y"] - o["y"]) * (b["x"] - o["x"])

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    @staticmethod
    def _point_ids(points: Sequence[Point]) -> List[str]:
        return [str(point["id"]) for point in points]

    @staticmethod
    def _edge(source: str, target: str, kind: str = "hull", label: str | None = None) -> Dict[str, Any]:
        edge: Dict[str, Any] = {"source": source, "target": target, "kind": kind}
        if label is not None:
            edge["label"] = label
        return edge

    @staticmethod
    def _frame(
        action: str,
        points: Sequence[Point],
        *,
        edges: Sequence[Dict[str, Any]] | None = None,
        hull: Sequence[Point] | None = None,
        candidate_edge: Sequence[str] | None = None,
        stack: Sequence[str] | None = None,
        sorted_order: Sequence[str] | None = None,
        phase: str | None = None,
        depth: int | None = None,
        metrics: Dict[str, Any] | None = None,
        highlights_nodes: Iterable[str] | None = None,
        highlights_edges: Iterable[Sequence[str]] | None = None,
    ) -> Frame:
        state: Dict[str, Any] = {
            "points": list(points),
            "edges": list(edges or []),
        }

        if hull is not None:
            state["hull"] = ConvexHullSolver._point_ids(hull)
        if candidate_edge is not None:
            state["candidateEdge"] = list(candidate_edge)
        if stack is not None:
            state["stack"] = list(stack)
        if sorted_order is not None:
            state["sortedOrder"] = list(sorted_order)
        if phase is not None:
            state["phase"] = phase
        if depth is not None:
            state["depth"] = depth

        frame: Frame = {"action": action, "state": state}

        if metrics is not None:
            frame["metrics"] = metrics

        highlights: Dict[str, Any] = {}
        if highlights_nodes is not None:
            highlights["elements"] = list(dict.fromkeys(str(node) for node in highlights_nodes))
        if highlights_edges is not None:
            highlights["connections"] = [list(edge) for edge in highlights_edges]
        if highlights:
            frame["highlights"] = highlights

        return frame

    @staticmethod
    def _unique_by_coordinate(points: Sequence[Point]) -> List[Point]:
        unique: List[Point] = []
        seen = set()
        for point in sorted(points, key=lambda p: (p["x"], p["y"], p["id"])):
            key = (round(point["x"], 12), round(point["y"], 12))
            if key in seen:
                continue
            seen.add(key)
            unique.append(point)
        return unique

    @staticmethod
    def _monotonic_chain(points: Sequence[Point]) -> List[Point]:
        ordered = ConvexHullSolver._unique_by_coordinate(points)
        if len(ordered) <= 1:
            return list(ordered)

        lower: List[Point] = []
        for point in ordered:
            while len(lower) >= 2 and ConvexHullSolver._cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)

        upper: List[Point] = []
        for point in reversed(ordered):
            while len(upper) >= 2 and ConvexHullSolver._cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)

        return lower[:-1] + upper[:-1]

    @staticmethod
    def _brute_force(points: Sequence[Point]) -> List[Frame]:
        frames: List[Frame] = []
        edges: List[Dict[str, Any]] = []
        edge_keys = set()
        iterations = 0

        frames.append(
            ConvexHullSolver._frame(
                "Starting brute force convex hull",
                points,
                metrics={"iterations": 0, "quality": 1, "algorithm": "brute-force", "pointCount": len(points)},
            )
        )

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1 = points[i]
                p2 = points[j]
                side = None
                is_hull_edge = True

                frames.append(
                    ConvexHullSolver._frame(
                        f"Checking candidate edge {p1['id']} - {p2['id']}",
                        points,
                        edges=edges,
                        candidate_edge=[str(p1["id"]), str(p2["id"])],
                        hull=ConvexHullSolver._monotonic_chain([point for point in points if (point["id"] in ConvexHullSolver._point_ids(points))]),
                        highlights_nodes=[p1["id"], p2["id"]],
                        highlights_edges=[[p1["id"], p2["id"]]],
                        metrics={"iterations": iterations, "quality": 1, "algorithm": "brute-force", "pointCount": len(points)},
                    )
                )

                for k in range(len(points)):
                    if k == i or k == j:
                        continue

                    iterations += 1
                    current = points[k]
                    cross_value = ConvexHullSolver._cross(p1, p2, current)

                    if cross_value == 0:
                        continue

                    current_side = 1 if cross_value > 0 else -1
                    if side is None:
                        side = current_side
                    elif side != current_side:
                        is_hull_edge = False
                        break

                    if len(points) <= 25:
                        frames.append(
                            ConvexHullSolver._frame(
                                f"Compare {p1['id']} - {p2['id']} against {current['id']}",
                                points,
                                edges=edges,
                                candidate_edge=[str(p1["id"]), str(p2["id"])],
                                highlights_nodes=[p1["id"], p2["id"], current["id"]],
                                highlights_edges=[[p1["id"], p2["id"]]],
                                metrics={"iterations": iterations, "quality": 1, "algorithm": "brute-force", "pointCount": len(points)},
                            )
                        )

                if is_hull_edge:
                    key = tuple(sorted((str(p1["id"]), str(p2["id"]))))
                    if key not in edge_keys:
                        edge_keys.add(key)
                        edges.append(ConvexHullSolver._edge(str(p1["id"]), str(p2["id"]), kind="hull"))
                        frames.append(
                            ConvexHullSolver._frame(
                                f"Found hull edge {p1['id']} - {p2['id']}",
                                points,
                                edges=edges,
                                hull=ConvexHullSolver._monotonic_chain(points),
                                highlights_nodes=[p1["id"], p2["id"]],
                                highlights_edges=[[p1["id"], p2["id"]]],
                                metrics={"iterations": iterations, "quality": 1, "algorithm": "brute-force", "pointCount": len(points)},
                            )
                        )

        hull = ConvexHullSolver._monotonic_chain(points)
        hull_edges = [
            ConvexHullSolver._edge(str(hull[index]["id"]), str(hull[(index + 1) % len(hull)]["id"]), kind="hull")
            for index in range(len(hull))
        ] if len(hull) >= 2 else []

        frames.append(
            ConvexHullSolver._frame(
                f"Brute force complete: {len(hull)} hull vertices found",
                points,
                edges=hull_edges,
                hull=hull,
                highlights_nodes=[point["id"] for point in hull],
                highlights_edges=[[hull[index]["id"], hull[(index + 1) % len(hull)]["id"]] for index in range(len(hull))] if len(hull) >= 2 else [],
                metrics={"iterations": iterations, "quality": 1, "algorithm": "brute-force", "pointCount": len(points)},
            )
        )

        return frames

    @staticmethod
    def _graham_scan(points: Sequence[Point]) -> List[Frame]:
        frames: List[Frame] = []
        iterations = 0

        start = min(points, key=lambda point: (point["y"], point["x"], point["id"]))
        sorted_points = sorted(
            points,
            key=lambda point: (
                0 if point["id"] == start["id"] else 1,
                math.atan2(point["y"] - start["y"], point["x"] - start["x"]),
                ConvexHullSolver._distance(start, point),
                point["id"],
            ),
        )

        frames.append(
            ConvexHullSolver._frame(
                "Starting Graham Scan",
                points,
                sorted_order=ConvexHullSolver._point_ids(sorted_points),
                highlights_nodes=[start["id"]],
                metrics={"iterations": 0, "quality": 1, "algorithm": "graham-scan", "pointCount": len(points)},
            )
        )
        frames.append(
            ConvexHullSolver._frame(
                f"Selected pivot {start['id']} (lowest point)",
                points,
                sorted_order=ConvexHullSolver._point_ids(sorted_points),
                highlights_nodes=[start["id"]],
                metrics={"iterations": 0, "quality": 1, "algorithm": "graham-scan", "pointCount": len(points)},
            )
        )

        hull: List[Point] = []
        for point in sorted_points:
            frames.append(
                ConvexHullSolver._frame(
                    f"Consider {point['id']} for the hull stack",
                    points,
                    stack=ConvexHullSolver._point_ids(hull),
                    candidate_edge=[str(hull[-1]["id"]), str(point["id"])] if hull else [str(point["id"]), str(point["id"])] ,
                    highlights_nodes=ConvexHullSolver._point_ids(hull) + [point["id"]],
                    metrics={"iterations": iterations, "quality": 1, "algorithm": "graham-scan", "pointCount": len(points)},
                )
            )

            while len(hull) >= 2:
                iterations += 1
                turn = ConvexHullSolver._cross(hull[-2], hull[-1], point)
                if turn <= 0:
                    removed = hull.pop()
                    frames.append(
                        ConvexHullSolver._frame(
                            f"Pop {removed['id']} because {hull[-1]['id']} -> {removed['id']} -> {point['id']} is not a left turn",
                            points,
                            stack=ConvexHullSolver._point_ids(hull),
                            candidate_edge=[str(hull[-1]["id"]), str(point["id"])] if hull else [str(point["id"]), str(point["id"])],
                            highlights_nodes=ConvexHullSolver._point_ids(hull) + [removed["id"], point["id"]],
                            metrics={"iterations": iterations, "quality": 1, "algorithm": "graham-scan", "pointCount": len(points)},
                        )
                    )
                else:
                    break

            hull.append(point)
            frames.append(
                ConvexHullSolver._frame(
                    f"Push {point['id']} onto the hull stack",
                    points,
                    stack=ConvexHullSolver._point_ids(hull),
                    hull=hull,
                    candidate_edge=[str(hull[-2]["id"]), str(hull[-1]["id"])] if len(hull) >= 2 else None,
                    highlights_nodes=ConvexHullSolver._point_ids(hull),
                    highlights_edges=[[hull[-2]["id"], hull[-1]["id"]]] if len(hull) >= 2 else [],
                    metrics={"iterations": iterations, "quality": 1, "algorithm": "graham-scan", "pointCount": len(points)},
                )
            )

        hull = ConvexHullSolver._monotonic_chain(hull)
        frames.append(
            ConvexHullSolver._frame(
                f"Graham Scan complete: {len(hull)} hull vertices found",
                points,
                hull=hull,
                edges=[ConvexHullSolver._edge(str(hull[index]["id"]), str(hull[(index + 1) % len(hull)]["id"]), kind="hull") for index in range(len(hull))] if len(hull) >= 2 else [],
                highlights_nodes=[point["id"] for point in hull],
                highlights_edges=[[hull[index]["id"], hull[(index + 1) % len(hull)]["id"]] for index in range(len(hull))] if len(hull) >= 2 else [],
                metrics={"iterations": iterations, "quality": 1, "algorithm": "graham-scan", "pointCount": len(points)},
            )
        )

        return frames

    @staticmethod
    def _divide_and_conquer(points: Sequence[Point]) -> List[Frame]:
        frames: List[Frame] = []
        iterations = 0
        ordered = sorted(points, key=lambda point: (point["x"], point["y"], point["id"]))

        def hull_edges(hull: Sequence[Point], kind: str = "hull") -> List[Dict[str, Any]]:
            if len(hull) < 2:
                return []
            return [
                ConvexHullSolver._edge(str(hull[index]["id"]), str(hull[(index + 1) % len(hull)]["id"]), kind=kind)
                for index in range(len(hull))
            ]

        def all_ids(subset: Sequence[Point]) -> List[str]:
            return ConvexHullSolver._point_ids(subset)

        def support_side(left_point: Point, right_point: Point, subset: Sequence[Point]) -> int | None:
            side: int | None = None
            for point in subset:
                cross_value = ConvexHullSolver._cross(left_point, right_point, point)
                if abs(cross_value) < 1e-9:
                    continue

                current_side = 1 if cross_value > 0 else -1
                if side is None:
                    side = current_side
                elif side != current_side:
                    return None

            return side

        def tangent_score(left_point: Point, right_point: Point) -> tuple[float, float, str, str]:
            return ((left_point["y"] + right_point["y"]) / 2.0, left_point["x"] + right_point["x"], str(left_point["id"]), str(right_point["id"]))

        frames.append(
            ConvexHullSolver._frame(
                "Starting Divide and Conquer",
                points,
                sorted_order=ConvexHullSolver._point_ids(ordered),
                metrics={"iterations": 0, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
            )
        )

        def merge_hulls(left_hull: Sequence[Point], right_hull: Sequence[Point], depth: int) -> List[Point]:
            nonlocal iterations

            left_hull = ConvexHullSolver._monotonic_chain(left_hull)
            right_hull = ConvexHullSolver._monotonic_chain(right_hull)

            if not left_hull:
                return list(right_hull)
            if not right_hull:
                return list(left_hull)

            merged_subset = list(left_hull) + list(right_hull)
            if len(left_hull) <= 2 or len(right_hull) <= 2:
                merged_small = ConvexHullSolver._monotonic_chain(merged_subset)
                frames.append(
                    ConvexHullSolver._frame(
                        f"Merge small hulls at depth {depth}",
                        points,
                        edges=hull_edges(left_hull) + hull_edges(right_hull),
                        hull=merged_small,
                        depth=depth,
                        highlights_nodes=[point["id"] for point in merged_small],
                        metrics={"iterations": iterations, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
                    )
                )
                return merged_small

            frames.append(
                ConvexHullSolver._frame(
                    f"Merge hulls at depth {depth}",
                    points,
                    edges=hull_edges(left_hull) + hull_edges(right_hull),
                    depth=depth,
                    highlights_nodes=all_ids(left_hull) + all_ids(right_hull),
                    metrics={"iterations": iterations, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
                )
            )

            valid_tangents: List[tuple[tuple[float, float, str, str], Point, Point, int]] = []
            for left_point in left_hull:
                for right_point in right_hull:
                    iterations += 1
                    frames.append(
                        ConvexHullSolver._frame(
                            f"Check tangent candidate {left_point['id']} - {right_point['id']}",
                            points,
                            edges=hull_edges(left_hull) + hull_edges(right_hull),
                            candidate_edge=[str(left_point["id"]), str(right_point["id"])],
                            depth=depth,
                            highlights_nodes=[left_point["id"], right_point["id"]],
                            highlights_edges=[[left_point["id"], right_point["id"]]],
                            metrics={"iterations": iterations, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
                        )
                    )

                    side = support_side(left_point, right_point, merged_subset)
                    if side is None:
                        continue

                    valid_tangents.append((tangent_score(left_point, right_point), left_point, right_point, side))

            if not valid_tangents:
                merged_fallback = ConvexHullSolver._monotonic_chain(merged_subset)
                frames.append(
                    ConvexHullSolver._frame(
                        f"Fallback merge at depth {depth}",
                        points,
                        hull=merged_fallback,
                        edges=hull_edges(merged_fallback),
                        depth=depth,
                        highlights_nodes=[point["id"] for point in merged_fallback],
                        metrics={"iterations": iterations, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
                    )
                )
                return merged_fallback

            upper_score, upper_left, upper_right, _ = max(valid_tangents, key=lambda item: item[0])
            lower_score, lower_left, lower_right, _ = min(valid_tangents, key=lambda item: item[0])

            frames.append(
                ConvexHullSolver._frame(
                    f"Upper tangent found at depth {depth}",
                    points,
                    edges=hull_edges(left_hull) + hull_edges(right_hull),
                    candidate_edge=[str(upper_left["id"]), str(upper_right["id"])],
                    depth=depth,
                    highlights_nodes=[upper_left["id"], upper_right["id"]],
                    highlights_edges=[[upper_left["id"], upper_right["id"]]],
                    metrics={"iterations": iterations, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
                )
            )
            frames.append(
                ConvexHullSolver._frame(
                    f"Lower tangent found at depth {depth}",
                    points,
                    edges=hull_edges(left_hull) + hull_edges(right_hull),
                    candidate_edge=[str(lower_left["id"]), str(lower_right["id"])],
                    depth=depth,
                    highlights_nodes=[lower_left["id"], lower_right["id"]],
                    highlights_edges=[[lower_left["id"], lower_right["id"]]],
                    metrics={"iterations": iterations, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
                )
            )

            merged_final = ConvexHullSolver._monotonic_chain(merged_subset)
            frames.append(
                ConvexHullSolver._frame(
                    f"Merged hull at depth {depth}: {len(merged_final)} vertices",
                    points,
                    hull=merged_final,
                    edges=hull_edges(merged_final),
                    depth=depth,
                    highlights_nodes=[point["id"] for point in merged_final],
                    highlights_edges=[[merged_final[index]["id"], merged_final[(index + 1) % len(merged_final)]["id"]] for index in range(len(merged_final))] if len(merged_final) >= 2 else [],
                    metrics={"iterations": iterations, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
                )
            )
            return merged_final

        def solve_recursively(subset: Sequence[Point], depth: int = 0) -> List[Point]:
            if len(subset) <= 3:
                base_hull = ConvexHullSolver._monotonic_chain(subset)
                frames.append(
                    ConvexHullSolver._frame(
                        f"Base case at depth {depth}: {len(base_hull)} hull vertices",
                        points,
                        edges=hull_edges(base_hull),
                        hull=base_hull,
                        depth=depth,
                        highlights_nodes=[point["id"] for point in base_hull],
                        metrics={"iterations": iterations, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
                    )
                )
                return base_hull

            mid = len(subset) // 2
            left = subset[:mid]
            right = subset[mid:]

            frames.append(
                ConvexHullSolver._frame(
                    f"Divide at depth {depth}: {len(left)} left, {len(right)} right",
                    points,
                    sorted_order=ConvexHullSolver._point_ids(subset),
                    depth=depth,
                    highlights_nodes=[point["id"] for point in left] + [point["id"] for point in right],
                    metrics={"iterations": iterations, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
                )
            )

            left_hull = solve_recursively(left, depth + 1)
            right_hull = solve_recursively(right, depth + 1)

            return merge_hulls(left_hull, right_hull, depth)

        final_hull = solve_recursively(ordered)
        frames.append(
            ConvexHullSolver._frame(
                f"Divide and Conquer complete: {len(final_hull)} hull vertices found",
                points,
                hull=final_hull,
                edges=[ConvexHullSolver._edge(str(final_hull[index]["id"]), str(final_hull[(index + 1) % len(final_hull)]["id"]), kind="hull") for index in range(len(final_hull))] if len(final_hull) >= 2 else [],
                highlights_nodes=[point["id"] for point in final_hull],
                highlights_edges=[[final_hull[index]["id"], final_hull[(index + 1) % len(final_hull)]["id"]] for index in range(len(final_hull))] if len(final_hull) >= 2 else [],
                metrics={"iterations": iterations, "quality": 1, "algorithm": "divide-conquer", "pointCount": len(points)},
            )
        )

        return frames
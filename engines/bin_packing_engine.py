from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


class BinPackingSolver:
    @staticmethod
    def _normalize_items(items: Any) -> List[Dict[str, int | str]]:
        normalized: List[Dict[str, int | str]] = []

        if isinstance(items, str):
            raw_lines = [line.strip() for line in items.splitlines() if line.strip()]
            for index, line in enumerate(raw_lines):
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 1:
                    name = f"Item {index + 1}"
                    weight_raw = parts[0]
                else:
                    name = parts[0] or f"Item {index + 1}"
                    weight_raw = parts[1]

                weight_value = int(float(weight_raw))
                normalized.append({"name": name, "weight": weight_value})
            return normalized

        if isinstance(items, (list, tuple)):
            for index, item in enumerate(items):
                if isinstance(item, dict):
                    name = str(item.get("name") or f"Item {index + 1}")
                    weight = int(item.get("weight", item.get("value", 0)))
                elif isinstance(item, (list, tuple)):
                    if len(item) >= 2:
                        name = str(item[0])
                        weight = int(item[1])
                    elif len(item) == 1:
                        name = f"Item {index + 1}"
                        weight = int(item[0])
                    else:
                        continue
                else:
                    name = f"Item {index + 1}"
                    weight = int(item)

                normalized.append({"name": name, "weight": weight})

        return normalized

    @staticmethod
    def solve(items: Any, capacity: int):
        capacity = int(capacity)
        normalized_items = BinPackingSolver._normalize_items(items)

        if capacity < 1:
            raise ValueError("Bin capacity must be at least 1.")
        if not normalized_items:
            raise ValueError("Provide at least one item.")

        for item in normalized_items:
            if int(item["weight"]) > capacity:
                raise ValueError(f"Item {item['name']} exceeds the bin capacity.")

        ordered_items = sorted(normalized_items, key=lambda item: (-int(item["weight"]), str(item["name"])))
        steps: List[Dict[str, Any]] = []
        heuristic_bins: List[Dict[str, Any]] = []
        for item in ordered_items:
            placed = False
            for bin_entry in heuristic_bins:
                if bin_entry["usedWeight"] + int(item["weight"]) <= capacity:
                    bin_entry["items"].append({"name": item["name"], "weight": int(item["weight"])})
                    bin_entry["usedWeight"] += int(item["weight"])
                    placed = True
                    break
            if not placed:
                heuristic_bins.append({"items": [{"name": item["name"], "weight": int(item["weight"])}], "usedWeight": int(item["weight"])})

        best_solution: List[Dict[str, Any]] | None = deepcopy(heuristic_bins)
        best_bin_count = len(heuristic_bins)

        def snapshot(action: str, active_item: str | None = None, active_bin: int | None = None, terminal: bool = False, accepted: bool = False, message: str | None = None):
            current_bins = [
                {
                    "index": index,
                    "items": deepcopy(bin_entry["items"]),
                    "usedWeight": bin_entry["usedWeight"],
                    "remaining": capacity - bin_entry["usedWeight"],
                }
                for index, bin_entry in enumerate(bins)
            ]
            steps.append({
                "action": action,
                "metrics": {
                    "item_count": len(ordered_items),
                    "capacity": capacity,
                    "used_bins": len(bins),
                    "best_bins": best_bin_count,
                },
                "state": {
                    "items": deepcopy(ordered_items),
                    "bins": current_bins,
                    "capacity": capacity,
                    "activeItem": active_item,
                    "activeBin": active_bin,
                    "bestBinCount": best_bin_count,
                    "message": message,
                    "isTerminal": terminal,
                    "isAccepted": accepted,
                },
            })

        bins: List[Dict[str, Any]] = []
        placement: Dict[str, int] = {}
        snapshot("Initialized bin packing search", message=f"Searching for the minimum number of bins with capacity {capacity}. Heuristic start uses {best_bin_count} bins.")

        def search(index: int):
            nonlocal best_solution, best_bin_count

            if index >= len(ordered_items):
                if len(bins) <= best_bin_count:
                    best_bin_count = len(bins)
                    best_solution = deepcopy(bins)
                    snapshot(
                        f"Found packing using {best_bin_count} bins",
                        accepted=True,
                        message="A complete packing was found."
                    )
                return

            if len(bins) >= best_bin_count:
                snapshot(
                    f"Pruned branch at {ordered_items[index]['name']}",
                    active_item=str(ordered_items[index]["name"]),
                    message=f"Pruned because current bin count {len(bins)} is no better than the best known {best_bin_count}."
                )
                return

            item = ordered_items[index]
            item_name = str(item["name"])
            item_weight = int(item["weight"])

            for bin_index, bin_entry in enumerate(bins):
                if bin_entry["usedWeight"] + item_weight > capacity:
                    snapshot(
                        f"Cannot place {item_name} in bin {bin_index + 1}",
                        active_item=item_name,
                        active_bin=bin_index,
                        message=f"Bin {bin_index + 1} does not have enough space for {item_name}."
                    )
                    continue

                bin_entry["items"].append({"name": item_name, "weight": item_weight})
                bin_entry["usedWeight"] += item_weight
                placement[item_name] = bin_index
                snapshot(
                    f"Place {item_name} in bin {bin_index + 1}",
                    active_item=item_name,
                    active_bin=bin_index,
                    message=f"Placed {item_name} into bin {bin_index + 1}."
                )

                search(index + 1)

                snapshot(
                    f"Remove {item_name} from bin {bin_index + 1}",
                    active_item=item_name,
                    active_bin=bin_index,
                    message=f"Backtracking from bin {bin_index + 1}."
                )
                bin_entry["items"].pop()
                bin_entry["usedWeight"] -= item_weight
                placement.pop(item_name, None)

            bins.append({"items": [{"name": item_name, "weight": item_weight}], "usedWeight": item_weight})
            placement[item_name] = len(bins) - 1
            snapshot(
                f"Open bin {len(bins)} for {item_name}",
                active_item=item_name,
                active_bin=len(bins) - 1,
                message=f"Opened a new bin for {item_name}."
            )

            search(index + 1)

            snapshot(
                f"Close bin {len(bins)} for {item_name}",
                active_item=item_name,
                active_bin=len(bins) - 1,
                message=f"Backtracking from the new bin opened for {item_name}."
            )
            bins.pop()
            placement.pop(item_name, None)

        search(0)

        final_bins = best_solution if best_solution is not None else []
        final_state = {
            "items": deepcopy(ordered_items),
            "bins": [
                {
                    "index": index,
                    "items": deepcopy(bin_entry["items"]),
                    "usedWeight": bin_entry["usedWeight"],
                    "remaining": capacity - bin_entry["usedWeight"],
                }
                for index, bin_entry in enumerate(final_bins)
            ],
            "capacity": capacity,
            "bestBinCount": best_bin_count if best_solution is not None else 0,
            "isTerminal": True,
            "isAccepted": best_solution is not None,
        }

        steps.append({
            "action": "Bin packing complete",
            "metrics": {
                "best_bins": best_bin_count if best_solution is not None else 0,
                "item_count": len(ordered_items),
                "capacity": capacity,
            },
            "state": final_state,
        })

        return final_bin_count(final_bins), steps


def final_bin_count(bins: List[Dict[str, Any]]) -> int:
    return len(bins)

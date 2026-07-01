class KnapsackSolver:
    @staticmethod
    def _normalize_items(items):
        normalized = []
        for index, item in enumerate(items):
            if isinstance(item, dict):
                name = str(item.get("name") or f"Item {index + 1}")
                weight = int(item.get("weight", 0))
                value = int(item.get("value", 0))
            elif isinstance(item, (list, tuple)):
                if len(item) == 3:
                    name, weight, value = item
                elif len(item) == 2:
                    name = f"Item {index + 1}"
                    weight, value = item
                else:
                    continue
                name = str(name)
                weight = int(weight)
                value = int(value)
            else:
                continue

            normalized.append((name, weight, value))

        return normalized

    @staticmethod
    def solve(items, capacity):
        normalized_items = KnapsackSolver._normalize_items(items)
        n = len(normalized_items)
        capacity = int(capacity)

        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        decision = [[False for _ in range(capacity + 1)] for _ in range(n + 1)]
        steps = []

        def build_rows(row_count):
            rows = []
            for row_index in range(row_count + 1):
                label = "Base" if row_index == 0 else normalized_items[row_index - 1][0]
                row = {"item": label}
                for cap in range(capacity + 1):
                    row[f"cap{cap}"] = dp[row_index][cap]
                rows.append(row)
            return rows

        steps.append({
            "action": "Initialized knapsack DP table",
            "metrics": {
                "item_count": n,
                "capacity": capacity,
                "best_value": 0,
            },
            "state": {
                "rows": build_rows(0),
                "dp_table": [row[:] for row in dp],
                "decision_matrix": [row[:] for row in decision],
                "active_pointer": [0, 0],
                "current_item": None,
                "current_weight": None,
                "current_value": None,
                "isTerminal": False,
            },
            "highlights": {"elements": ["0-cap0"]},
        })

        for i in range(1, n + 1):
            name, weight, value = normalized_items[i - 1]
            for w in range(capacity + 1):
                exclude_value = dp[i - 1][w]
                include_value = None
                take_item = False

                if weight <= w:
                    include_value = value + dp[i - 1][w - weight]
                    if include_value > exclude_value:
                        take_item = True

                dp[i][w] = include_value if take_item and include_value is not None else exclude_value
                decision[i][w] = take_item

                steps.append({
                    "action": f"Processed {name} at capacity {w}",
                    "metrics": {
                        "item": name,
                        "current_weight": weight,
                        "current_value": value,
                        "capacity": w,
                        "include_value": include_value,
                        "exclude_value": exclude_value,
                        "best_value": dp[i][w],
                    },
                    "state": {
                        "rows": build_rows(i),
                        "dp_table": [row[:] for row in dp],
                        "decision_matrix": [row[:] for row in decision],
                        "active_pointer": [i, w],
                        "current_item": name,
                        "current_weight": weight,
                        "current_value": value,
                        "capacity": capacity,
                        "isTerminal": False,
                    },
                    "highlights": {"elements": [f"{i}-cap{w}"]},
                })

        chosen_items = []
        chosen_weight = 0
        remaining_capacity = capacity

        for i in range(n, 0, -1):
            if decision[i][remaining_capacity]:
                name, weight, value = normalized_items[i - 1]
                chosen_items.append({"name": name, "weight": weight, "value": value})
                chosen_weight += weight
                remaining_capacity -= weight

        chosen_items.reverse()
        best_value = dp[n][capacity]

        steps.append({
            "action": "Knapsack computation complete",
            "metrics": {
                "best_value": best_value,
                "packed_weight": chosen_weight,
                "capacity": capacity,
                "selected_items": [item["name"] for item in chosen_items],
            },
            "state": {
                "rows": build_rows(n),
                "dp_table": [row[:] for row in dp],
                "decision_matrix": [row[:] for row in decision],
                "active_pointer": [n, capacity],
                "current_item": None,
                "current_weight": None,
                "current_value": None,
                "selected_items": chosen_items,
                "packed_weight": chosen_weight,
                "best_value": best_value,
                "capacity": capacity,
                "isTerminal": True,
                "isAccepted": best_value > 0,
            },
        })

        return best_value, steps
class SubsetSumSolver:
    @staticmethod
    def _normalize_items(items):
        normalized = []
        for index, item in enumerate(items):
            if isinstance(item, dict):
                value = int(item.get("value", 0))
            elif isinstance(item, (list, tuple)):
                if len(item) >= 1:
                    value = int(item[0])
                else:
                    continue
            else:
                try:
                    value = int(item)
                except Exception:
                    continue
            normalized.append(value)
        return normalized

    @staticmethod
    def solve(numbers, target):
        nums = SubsetSumSolver._normalize_items(numbers)
        n = len(nums)
        target = int(target)

        # dp[i][s] => whether sum s reachable using first i items
        dp = [[False for _ in range(target + 1)] for _ in range(n + 1)]
        choice = [[False for _ in range(target + 1)] for _ in range(n + 1)]
        dp[0][0] = True

        steps = []

        def build_rows(row_count):
            rows = []
            for i in range(row_count + 1):
                label = "Base (empty set)" if i == 0 else f"n{i} = {nums[i - 1]}"
                row = {"item": label}
                for s in range(target + 1):
                    row[f"s{s}"] = dp[i][s]
                rows.append(row)
            return rows

        steps.append({
            "action": "Initialized subset-sum DP table",
            "metrics": {"n": n, "target": target},
            "state": {
                "rows": build_rows(0),
                "dp_table": [row[:] for row in dp],
                "active_pointer": [0, 0],
                "isTerminal": False,
            }
        })

        for i in range(1, n + 1):
            val = nums[i - 1]
            dp[i][0] = True
            for s in range(1, target + 1):
                without = dp[i - 1][s]
                with_curr = dp[i - 1][s - val] if s >= val else False
                take = with_curr and not without
                dp[i][s] = without or with_curr
                choice[i][s] = take

                steps.append({
                    "action": f"Processed n{i} (value={val}) for sum {s}",
                    "metrics": {"item_index": i, "value": val, "sum": s, "reachable": dp[i][s]},
                    "state": {
                        "rows": build_rows(i),
                        "dp_table": [row[:] for row in dp],
                        "active_pointer": [i, s],
                        "isTerminal": False,
                    },
                    "highlights": {"elements": [f"{i}-s{s}"]},
                })

        # Reconstruct a witness subset if reachable
        reachable = dp[n][target]
        subset = []
        rem = target
        for i in range(n, 0, -1):
            if rem <= 0:
                break
            if choice[i][rem]:
                subset.append(nums[i - 1])
                rem -= nums[i - 1]

        subset.reverse()

        steps.append({
            "action": "Subset-sum computation complete",
            "metrics": {"reachable": reachable, "subset": subset},
            "state": {
                "rows": build_rows(n),
                "dp_table": [row[:] for row in dp],
                "isTerminal": True,
                "isAccepted": reachable,
                "selected_subset": subset,
            }
        })

        return reachable, steps

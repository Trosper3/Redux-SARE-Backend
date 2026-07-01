#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace py = pybind11;

struct IrrigationGraph {
    std::vector<std::vector<int>> adjacency_list;
    std::vector<std::vector<char>> adjacency_matrix;
};

static unsigned make_seed(int field_count, double conflict_density, double average_duration, double aqueduct_limit) {
    unsigned seed = 2166136261u;
    const auto mix = [&](unsigned value) {
        seed ^= value + 0x9e3779b9u + (seed << 6) + (seed >> 2);
    };
    mix(static_cast<unsigned>(field_count));
    mix(static_cast<unsigned>(conflict_density * 1000.0));
    mix(static_cast<unsigned>(average_duration * 100.0));
    mix(static_cast<unsigned>(aqueduct_limit));
    return seed;
}

static IrrigationGraph build_random_graph(int field_count, double conflict_density, unsigned seed) {
    IrrigationGraph graph;
    graph.adjacency_list.assign(field_count, {});
    graph.adjacency_matrix.assign(field_count, std::vector<char>(field_count, 0));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int source = 0; source < field_count; ++source) {
        for (int target = source + 1; target < field_count; ++target) {
            if (distribution(rng) < conflict_density) {
                graph.adjacency_list[source].push_back(target);
                graph.adjacency_list[target].push_back(source);
                graph.adjacency_matrix[source][target] = 1;
                graph.adjacency_matrix[target][source] = 1;
            }
        }
    }

    return graph;
}

static std::vector<int> make_degree_order(const IrrigationGraph& graph) {
    const int field_count = static_cast<int>(graph.adjacency_list.size());
    std::vector<int> order(field_count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int left, int right) {
        const auto left_degree = graph.adjacency_list[left].size();
        const auto right_degree = graph.adjacency_list[right].size();
        if (left_degree != right_degree) {
            return left_degree > right_degree;
        }
        return left < right;
    });
    return order;
}

static std::vector<int> dsatur_coloring(const IrrigationGraph& graph) {
    const int field_count = static_cast<int>(graph.adjacency_list.size());
    std::vector<int> color(field_count, -1);
    std::vector<std::set<int>> neighbor_colors(field_count);
    std::vector<int> degrees(field_count, 0);

    for (int field = 0; field < field_count; ++field) {
        degrees[field] = static_cast<int>(graph.adjacency_list[field].size());
    }

    int start = 0;
    for (int field = 1; field < field_count; ++field) {
        if (degrees[field] > degrees[start]) {
            start = field;
        }
    }

    color[start] = 0;
    for (int neighbor : graph.adjacency_list[start]) {
        neighbor_colors[neighbor].insert(0);
    }

    int colored = 1;
    while (colored < field_count) {
        int best = -1;
        int best_saturation = -1;
        int best_degree = -1;

        for (int field = 0; field < field_count; ++field) {
            if (color[field] != -1) {
                continue;
            }

            const int saturation = static_cast<int>(neighbor_colors[field].size());
            if (saturation > best_saturation || (saturation == best_saturation && degrees[field] > best_degree)) {
                best = field;
                best_saturation = saturation;
                best_degree = degrees[field];
            }
        }

        std::set<int> used = neighbor_colors[best];
        int next_color = 0;
        while (used.count(next_color) != 0) {
            ++next_color;
        }

        color[best] = next_color;
        for (int neighbor : graph.adjacency_list[best]) {
            neighbor_colors[neighbor].insert(next_color);
        }
        ++colored;
    }

    return color;
}

static int maximum_clique_rec(
    const IrrigationGraph& graph,
    std::vector<int> current,
    std::vector<int> candidates,
    std::vector<int> excluded,
    int best
) {
    if (candidates.empty() && excluded.empty()) {
        return std::max(best, static_cast<int>(current.size()));
    }

    int pivot = -1;
    std::size_t pivot_neighbors = 0;
    for (int node : candidates) {
        std::size_t count = 0;
        for (int other : candidates) {
            if (graph.adjacency_matrix[node][other] != 0) {
                ++count;
            }
        }
        if (count > pivot_neighbors) {
            pivot = node;
            pivot_neighbors = count;
        }
    }
    for (int node : excluded) {
        std::size_t count = 0;
        for (int other : candidates) {
            if (graph.adjacency_matrix[node][other] != 0) {
                ++count;
            }
        }
        if (count > pivot_neighbors) {
            pivot = node;
            pivot_neighbors = count;
        }
    }

    std::vector<int> branch_vertices;
    branch_vertices.reserve(candidates.size());
    for (int node : candidates) {
        if (pivot == -1 || graph.adjacency_matrix[pivot][node] == 0) {
            branch_vertices.push_back(node);
        }
    }

    for (int node : branch_vertices) {
        current.push_back(node);

        std::vector<int> next_candidates;
        std::vector<int> next_excluded;
        for (int other : candidates) {
            if (graph.adjacency_matrix[node][other] != 0) {
                next_candidates.push_back(other);
            }
        }
        for (int other : excluded) {
            if (graph.adjacency_matrix[node][other] != 0) {
                next_excluded.push_back(other);
            }
        }

        best = maximum_clique_rec(graph, current, next_candidates, next_excluded, best);
        current.pop_back();

        candidates.erase(std::remove(candidates.begin(), candidates.end(), node), candidates.end());
        excluded.push_back(node);
    }

    return best;
}

static int maximum_clique_size(const IrrigationGraph& graph) {
    const int field_count = static_cast<int>(graph.adjacency_list.size());
    std::vector<int> current;
    std::vector<int> candidates(field_count);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::vector<int> excluded;
    return maximum_clique_rec(graph, current, candidates, excluded, 0);
}

static void exact_coloring_rec(
    const IrrigationGraph& graph,
    const std::vector<int>& order,
    std::size_t position,
    std::vector<int>& assignment,
    int used_colors,
    int& best_colors,
    std::vector<int>& best_assignment
) {
    if (used_colors >= best_colors) {
        return;
    }

    if (position == order.size()) {
        best_colors = used_colors;
        best_assignment = assignment;
        return;
    }

    const int field = order[position];
    std::vector<char> banned(static_cast<std::size_t>(best_colors), 0);
    for (int neighbor : graph.adjacency_list[field]) {
        const int neighbor_color = assignment[neighbor];
        if (neighbor_color >= 0 && neighbor_color < best_colors) {
            banned[static_cast<std::size_t>(neighbor_color)] = 1;
        }
    }

    for (int color = 0; color < used_colors; ++color) {
        if (!banned[static_cast<std::size_t>(color)]) {
            assignment[field] = color;
            exact_coloring_rec(graph, order, position + 1, assignment, used_colors, best_colors, best_assignment);
            assignment[field] = -1;
        }
    }

    if (used_colors + 1 < best_colors && !banned[static_cast<std::size_t>(used_colors)]) {
        assignment[field] = used_colors;
        exact_coloring_rec(graph, order, position + 1, assignment, used_colors + 1, best_colors, best_assignment);
        assignment[field] = -1;
    }
}

static std::vector<int> exact_coloring(const IrrigationGraph& graph, const std::vector<int>& heuristic_assignment) {
    std::vector<int> best_assignment = heuristic_assignment;
    int best_colors = 0;
    for (int color : heuristic_assignment) {
        best_colors = std::max(best_colors, color + 1);
    }

    const std::vector<int> order = make_degree_order(graph);
    std::vector<int> assignment(graph.adjacency_list.size(), -1);
    exact_coloring_rec(graph, order, 0, assignment, 0, best_colors, best_assignment);
    return best_assignment;
}

static std::vector<double> read_double_vector_or_default(py::dict inputs, const char* key, const std::vector<double>& fallback) {
    if (!inputs.contains(key)) {
        return fallback;
    }
    return inputs[key].cast<std::vector<double>>();
}

// Expose a callable implementation; the actual PYBIND11_MODULE is defined in solver_py.cpp
py::tuple solve_irrigation_schedule_impl(py::dict inputs, const std::string& mode) {
    const int field_count = inputs.contains("field_count") ? inputs["field_count"].cast<int>() : 16;
    const double conflict_density_input = inputs.contains("conflict_density") ? inputs["conflict_density"].cast<double>() : 40.0;
    const double average_duration = inputs.contains("average_duration") ? inputs["average_duration"].cast<double>() : 4.0;
    const double aqueduct_limit = inputs.contains("aqueduct_supply_limit") ? inputs["aqueduct_supply_limit"].cast<double>() : 2500.0;
    const double priority_bias = inputs.contains("priority_weighting") ? inputs["priority_weighting"].cast<double>() : 70.0;
    const double conflict_density = std::clamp(conflict_density_input / 100.0, 0.0, 0.95);

    const unsigned seed = make_seed(field_count, conflict_density, average_duration, aqueduct_limit);
    const IrrigationGraph graph = build_random_graph(field_count, conflict_density, seed);
    py::dict metrics;

    if (mode == "water_source_allocation" || mode == "resilience") {
        std::vector<double> priorities = read_double_vector_or_default(inputs, "priority_values", {});
        std::vector<double> flows = read_double_vector_or_default(inputs, "flow_rates", {});

        if (priorities.size() < static_cast<std::size_t>(field_count)) {
            priorities.resize(static_cast<std::size_t>(field_count), 0.0);
            for (int index = 0; index < field_count; ++index) {
                priorities[static_cast<std::size_t>(index)] = priority_bias * (1.0 + (static_cast<double>(index % 7) * 0.06));
            }
        }

        if (flows.size() < static_cast<std::size_t>(field_count)) {
            flows.resize(static_cast<std::size_t>(field_count), 0.0);
            for (int index = 0; index < field_count; ++index) {
                flows[static_cast<std::size_t>(index)] = std::max(10.0, 40.0 + average_duration * 35.0 + static_cast<double>((index % 5) * 12));
            }
        }

        std::vector<int> slot_assignment = dsatur_coloring(graph);
        const int slot_count = 1 + *std::max_element(slot_assignment.begin(), slot_assignment.end());
        std::vector<int> schedule_assignment_vector = slot_assignment;
        double allocated_flow = 0.0;
        int unserved_count = 0;
        // track which fields were actually accepted into a slot (to avoid inconsistent overwrites)
        std::vector<char> accepted(static_cast<std::size_t>(field_count), 0);

        for (int slot = 0; slot < slot_count; ++slot) {
            // collect fields assigned to this slot
            std::vector<int> bucket;
            for (int field = 0; field < field_count; ++field) {
                if (slot_assignment[static_cast<std::size_t>(field)] == slot) {
                    bucket.push_back(field);
                }
            }

            const int n = static_cast<int>(bucket.size());
            if (n == 0) continue;

            // build integer weights and values for DP (weights = flows rounded to int units, values = priorities)
            std::vector<int> weights(n);
            std::vector<double> values(n);
            for (int i = 0; i < n; ++i) {
                const double f = flows[static_cast<std::size_t>(bucket[static_cast<std::size_t>(i)])];
                // ensure at least weight 1 to avoid zero-weight items
                weights[i] = std::max(1, static_cast<int>(std::lrint(f)));
                values[i] = priorities[static_cast<std::size_t>(bucket[static_cast<std::size_t>(i)])];
            }

            const int W = std::max(0, static_cast<int>(std::lrint(aqueduct_limit)));
            if (W <= 0) {
                // nothing can be accepted in this slot
                unserved_count += n;
                continue;
            }

            // DP table: dp[i][w] = max value using first i items with capacity w
            std::vector<std::vector<double>> dp(n + 1, std::vector<double>(W + 1, 0.0));
            for (int i = 1; i <= n; ++i) {
                int wt = weights[i - 1];
                double val = values[i - 1];
                for (int w = 0; w <= W; ++w) {
                    dp[i][w] = dp[i - 1][w];
                    if (wt <= w) {
                        double cand = dp[i - 1][w - wt] + val;
                        if (cand > dp[i][w]) dp[i][w] = cand;
                    }
                }
            }

            // backtrack to find chosen items
            int w = W;
            std::vector<char> chosen(n, 0);
            for (int i = n; i >= 1; --i) {
                if (dp[i][w] != dp[i - 1][w]) {
                    chosen[i - 1] = 1;
                    w -= weights[i - 1];
                    if (w < 0) w = 0;
                }
            }

            // mark accepted and accumulate allocated_flow
            int chosen_count = 0;
            for (int i = 0; i < n; ++i) {
                if (chosen[i]) {
                    int field = bucket[static_cast<std::size_t>(i)];
                    accepted[static_cast<std::size_t>(field)] = 1;
                    allocated_flow += flows[static_cast<std::size_t>(field)];
                    ++chosen_count;
                }
            }
            unserved_count += (n - chosen_count);
        }

        // Rebuild schedule vector to reflect accepted fields (accepted -> keep slot, otherwise -1)
        for (int field = 0; field < field_count; ++field) {
            if (accepted[static_cast<std::size_t>(field)]) {
                schedule_assignment_vector[static_cast<std::size_t>(field)] = slot_assignment[static_cast<std::size_t>(field)];
            } else {
                schedule_assignment_vector[static_cast<std::size_t>(field)] = -1;
            }
        }

        metrics["allocated_gpm_throughput"] = allocated_flow;
        metrics["unserved_crop_count"] = unserved_count;
        metrics["system_stress_index"] = aqueduct_limit > 0.0 ? (allocated_flow / aqueduct_limit) * 100.0 : 0.0;
        metrics["solver_strategy"] = "dsatur-allocation";
        return py::make_tuple(schedule_assignment_vector, metrics);
    }

    std::vector<int> heuristic_assignment = dsatur_coloring(graph);
    int chromatic_slots = 0;
    for (int color : heuristic_assignment) {
        chromatic_slots = std::max(chromatic_slots, color + 1);
    }

    const int clique_lower_bound = maximum_clique_size(graph);
    std::vector<int> final_assignment = heuristic_assignment;
    std::string strategy = "dsatur-heuristic";

    if (field_count <= 22 && clique_lower_bound < chromatic_slots) {
        std::vector<int> refined = exact_coloring(graph, heuristic_assignment);
        int refined_slots = 0;
        for (int color : refined) {
            refined_slots = std::max(refined_slots, color + 1);
        }
        if (refined_slots <= chromatic_slots) {
            final_assignment = refined;
            chromatic_slots = refined_slots;
            strategy = "branch-and-bound-refinement";
        }
    } else if (clique_lower_bound == chromatic_slots) {
        strategy = "dsatur-tight-bound";
    }

    metrics["chromatic_slots"] = chromatic_slots;
    metrics["max_pressure_drop"] = average_duration * 0.05 * static_cast<double>(chromatic_slots);
    metrics["clique_lower_bound"] = clique_lower_bound;
    metrics["solver_strategy"] = strategy;

    return py::make_tuple(final_assignment, metrics);
}

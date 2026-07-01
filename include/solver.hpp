#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>

// Continuous Physics Bisection
double solve_bisection(double distance, double elevation_drop, double target_psi, double diameter);

// Discrete Knapsack Integer Solver
struct KnapsackResult {
    int max_value;
    int total_weight;
    std::vector<int> selected;
};
KnapsackResult solve_knapsack(const std::vector<double>& values, const std::vector<double>& weights, double max_weight);

#endif // SOLVER_HPP
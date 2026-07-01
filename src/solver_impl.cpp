#include "solver.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

double solve_bisection(double distance, double elevation_drop, double target_psi, double diameter) {
    if (diameter <= 0) throw std::invalid_argument("Diameter must be positive.");
    
    double low_power = 0.0;
    double high_power = 1000.0;
    double tolerance = 0.01;
    double roughness_c = 130.0;
    
    for (int i = 0; i < 100; ++i) {
        double mid_power = low_power + (high_power - low_power) / 2.0;
        double head_loss = 10.67 * std::pow(mid_power * 15.85, 1.852) / (std::pow(roughness_c, 1.852) * std::pow(diameter, 4.87)) * distance;
        double calculated_psi = (elevation_drop - head_loss) * 0.433;
        
        if (std::abs(calculated_psi - target_psi) < tolerance) return mid_power;
        if (calculated_psi < target_psi) low_power = mid_power;
        else high_power = mid_power;
    }
    return low_power;
}

KnapsackResult solve_knapsack(const std::vector<double>& values, const std::vector<double>& weights, double max_weight) {
    int n = values.size();
    int W = static_cast<int>(max_weight);
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(W + 1, 0));
    
    for (int i = 1; i <= n; ++i) {
        for (int w = 0; w <= W; ++w) {
            if (static_cast<int>(weights[i - 1]) <= w) {
                dp[i][w] = std::max(static_cast<int>(values[i - 1]) + dp[i - 1][w - static_cast<int>(weights[i - 1])], dp[i - 1][w]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }
    
    int remaining_w = W;
    std::vector<int> selected(n, 0);
    int total_weight = 0;
    int v = dp[n][W];
    
    for (int i = n; i > 0 && v > 0; --i) {
        if (v != dp[i - 1][remaining_w]) {
            selected[i - 1] = 1;
            v -= static_cast<int>(values[i - 1]);
            remaining_w -= static_cast<int>(weights[i - 1]);
            total_weight += static_cast<int>(weights[i - 1]);
        }
    }
    return {dp[n][W], total_weight, selected};
}
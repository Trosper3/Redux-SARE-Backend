#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {
struct Agent {
    double pumping_rate;
    double profit;
};

double clamp_value(double value, double low, double high) {
    return std::max(low, std::min(high, value));
}

double read_double(const py::dict& inputs, const char* key, double fallback) {
    if (!inputs.contains(key)) {
        return fallback;
    }
    try {
        return py::cast<double>(inputs[key]);
    } catch (...) {
        return fallback;
    }
}

int read_int(const py::dict& inputs, const char* key, int fallback) {
    if (!inputs.contains(key)) {
        return fallback;
    }
    try {
        return py::cast<int>(inputs[key]);
    } catch (...) {
        return fallback;
    }
}

py::dict make_frame(
    const std::string& action,
    double aquifer_volume,
    double tariff_rate,
    double total_pumping,
    double collective_profit,
    const std::vector<Agent>& agents,
    const std::vector<double>& volume_series,
    bool is_terminal,
    bool is_accepted,
    const std::string& mode_name,
    int epoch,
    int year
) {
    py::list agent_rows;
    for (std::size_t index = 0; index < agents.size(); ++index) {
        const auto& agent = agents[index];
        py::dict row;
        row["name"] = "Agent " + std::to_string(index + 1);
        row["pumping_rate"] = agent.pumping_rate;
        row["profit"] = agent.profit;
        agent_rows.append(row);
    }

    py::list volumes;
    for (double volume : volume_series) {
        volumes.append(volume);
    }

    py::dict metrics;
    metrics["optimal_tariff_rate"] = tariff_rate;
    metrics["sustainable_yield_state"] = volume_series.empty() ? 0.0 : (100.0 * aquifer_volume / std::max(1.0, volume_series.front()));
    metrics["collective_profit"] = collective_profit;
    metrics["years_to_collapse"] = volume_series.empty() ? 0 : static_cast<int>(volume_series.size() - 1);
    metrics["total_pumped_volume"] = total_pumping;
    metrics["aquifer_volume_timeseries"] = volumes;

    py::dict state;
    state["mode"] = mode_name;
    state["epoch"] = epoch;
    state["year"] = year;
    state["aquifer_volume"] = aquifer_volume;
    state["tariff_rate"] = tariff_rate;
    state["total_pumping"] = total_pumping;
    state["collective_profit"] = collective_profit;
    state["agents"] = agent_rows;
    state["aquifer_volume_timeseries"] = volumes;
    state["isTerminal"] = is_terminal;
    state["isAccepted"] = is_accepted;

    py::dict frame;
    frame["action"] = action;
    frame["metrics"] = metrics;
    frame["state"] = state;
    return frame;
}
}  // namespace

py::dict solve_aquifer_nash(const py::dict& inputs) {
    const int agent_count = std::max(2, read_int(inputs, "agent_count", 10));
    const double aquifer_capacity = std::max(1.0, read_double(inputs, "aquifer_capacity", 5000.0));
    const double recharge_rate = std::max(0.0, read_double(inputs, "recharge_rate", 250.0));
    const double crop_value = std::max(0.0, read_double(inputs, "crop_value", 0.15));
    const double starting_pump_rate = std::max(0.5, read_double(inputs, "starting_pump_rate", 1.2));

    const double base_share = recharge_rate / static_cast<double>(agent_count);
    const double max_rate = std::max(base_share * 4.0, recharge_rate * 0.65);
    const double target_rate = recharge_rate;
    const int max_epochs = 120;
    const double tolerance = std::max(0.5, recharge_rate * 0.005);

    double aquifer_volume = aquifer_capacity * 0.72;
    double tariff_rate = std::max(0.0, crop_value * 0.08);
    std::vector<Agent> agents(static_cast<std::size_t>(agent_count), Agent{base_share * starting_pump_rate, 0.0});
    std::vector<double> volume_series;
    volume_series.push_back(aquifer_volume);

    std::vector<py::dict> frames;

    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        const double scarcity = 1.0 - (aquifer_volume / aquifer_capacity);
        const double pumping_cost = 0.02 + 0.10 * clamp_value(scarcity, 0.0, 1.0);
        double total_pumping = 0.0;
        double collective_profit = 0.0;

        for (int index = 0; index < agent_count; ++index) {
            const double pressure = std::max(0.0, crop_value * 20.0 - tariff_rate * 30.0 - pumping_cost * 16.0);
            const double desired_rate = clamp_value(base_share * (1.0 + starting_pump_rate * pressure), 0.0, max_rate);
            agents[static_cast<std::size_t>(index)].pumping_rate = 0.6 * agents[static_cast<std::size_t>(index)].pumping_rate + 0.4 * desired_rate;
            agents[static_cast<std::size_t>(index)].pumping_rate = clamp_value(agents[static_cast<std::size_t>(index)].pumping_rate, 0.0, max_rate);
            agents[static_cast<std::size_t>(index)].profit = agents[static_cast<std::size_t>(index)].pumping_rate * std::max(0.0, crop_value * 1000.0 - pumping_cost * 1000.0 - tariff_rate * 1000.0);
            total_pumping += agents[static_cast<std::size_t>(index)].pumping_rate;
            collective_profit += agents[static_cast<std::size_t>(index)].profit;
        }

        aquifer_volume = clamp_value(aquifer_volume + recharge_rate - total_pumping, 0.0, aquifer_capacity);
        volume_series.push_back(aquifer_volume);

        const double target_gap = total_pumping - target_rate;
        tariff_rate = clamp_value(tariff_rate + 0.0025 * (target_gap / std::max(1.0, recharge_rate)), 0.0, crop_value * 24.0);

        const bool converged = std::abs(target_gap) <= tolerance;
        const bool terminal = converged || epoch == max_epochs - 1;
        const std::string action = converged
            ? "Tariff converged to a sustainable equilibrium"
            : "Fictitious play update across competing aquifer agents";

        frames.push_back(make_frame(action, aquifer_volume, tariff_rate, total_pumping, collective_profit, agents, volume_series, terminal, converged, "tariff_optimizer", epoch, epoch));

        if (converged) {
            break;
        }
    }

    if (frames.empty()) {
        frames.push_back(make_frame(
            "No optimization steps executed",
            aquifer_volume,
            tariff_rate,
            0.0,
            0.0,
            agents,
            volume_series,
            true,
            false,
            "tariff_optimizer",
            0,
            0
        ));
    }

    py::dict result;
    result["steps"] = py::cast(frames);
    result["metrics"] = frames.back()["metrics"];
    return result;
}

py::dict simulate_tragedy(const py::dict& inputs) {
    const int agent_count = std::max(2, read_int(inputs, "agent_count", 10));
    const double aquifer_capacity = std::max(1.0, read_double(inputs, "aquifer_capacity", 5000.0));
    const double recharge_rate = std::max(0.0, read_double(inputs, "recharge_rate", 250.0));
    const double crop_value = std::max(0.0, read_double(inputs, "crop_value", 0.15));
    const double starting_pump_rate = std::max(0.5, read_double(inputs, "starting_pump_rate", 1.2));

    const double base_share = recharge_rate / static_cast<double>(agent_count);
    const double max_rate = std::max(base_share * 5.0, recharge_rate * 0.95);
    double aquifer_volume = aquifer_capacity * 0.75;
    std::vector<Agent> agents(static_cast<std::size_t>(agent_count), Agent{base_share * starting_pump_rate, 0.0});
    std::vector<double> volume_series;
    volume_series.push_back(aquifer_volume);
    std::vector<py::dict> frames;

    int peak_profit_year = 0;
    double peak_profit = -1.0;
    int years_to_collapse = 0;

    for (int year = 0; year < 60; ++year) {
        const double scarcity = 1.0 - (aquifer_volume / aquifer_capacity);
        const double pumping_cost = 0.02 + 0.12 * clamp_value(scarcity, 0.0, 1.0);
        double total_pumping = 0.0;
        double collective_profit = 0.0;

        for (int index = 0; index < agent_count; ++index) {
            const double greed = std::max(0.0, crop_value * 22.0 - pumping_cost * 16.0);
            const double desired_rate = clamp_value(base_share * (1.0 + starting_pump_rate * greed + scarcity * 1.1), 0.0, max_rate);
            agents[static_cast<std::size_t>(index)].pumping_rate = 0.7 * agents[static_cast<std::size_t>(index)].pumping_rate + 0.3 * desired_rate;
            agents[static_cast<std::size_t>(index)].pumping_rate = clamp_value(agents[static_cast<std::size_t>(index)].pumping_rate, 0.0, max_rate);
            agents[static_cast<std::size_t>(index)].profit = agents[static_cast<std::size_t>(index)].pumping_rate * std::max(0.0, crop_value * 1000.0 - pumping_cost * 1000.0);
            total_pumping += agents[static_cast<std::size_t>(index)].pumping_rate;
            collective_profit += agents[static_cast<std::size_t>(index)].profit;
        }

        if (collective_profit > peak_profit) {
            peak_profit = collective_profit;
            peak_profit_year = year;
        }

        aquifer_volume = aquifer_volume + recharge_rate - total_pumping;
        if (aquifer_volume > aquifer_capacity) {
            aquifer_volume = aquifer_capacity;
        }
        if (aquifer_volume <= 0.0) {
            aquifer_volume = 0.0;
            volume_series.push_back(aquifer_volume);
            years_to_collapse = year + 1;
            frames.push_back(make_frame(
                "Aquifer collapsed under unregulated pumping",
                aquifer_volume,
                0.0,
                total_pumping,
                collective_profit,
                agents,
                volume_series,
                true,
                false,
                "tragedy_simulator",
                year,
                year
            ));
            break;
        }

        volume_series.push_back(aquifer_volume);
        years_to_collapse = year + 1;
        frames.push_back(make_frame(
            "Selfish best-response pumping under a tragedy of the commons",
            aquifer_volume,
            0.0,
            total_pumping,
            collective_profit,
            agents,
            volume_series,
            false,
            false,
            "tragedy_simulator",
            year,
            year
        ));
    }

    if (frames.empty()) {
        frames.push_back(make_frame(
            "No simulation steps executed",
            aquifer_volume,
            0.0,
            0.0,
            0.0,
            agents,
            volume_series,
            true,
            false,
            "tragedy_simulator",
            0,
            0
        ));
    }

    py::dict metrics = frames.back()["metrics"];
    metrics["years_to_collapse"] = years_to_collapse;
    metrics["peak_profit_year"] = peak_profit_year;
    metrics["peak_profit"] = peak_profit;
    metrics["aquifer_volume_timeseries"] = py::cast(volume_series);
    metrics["collective_profit"] = peak_profit;

    frames.back()["metrics"] = metrics;

    py::dict result;
    result["steps"] = py::cast(frames);
    result["metrics"] = metrics;
    return result;
}
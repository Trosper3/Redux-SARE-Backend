#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "solver.hpp"

namespace py = pybind11;

// Forward declare irrigation solver implementation (defined in irrigation_solver.cpp)
py::tuple solve_irrigation_schedule_impl(py::dict inputs, const std::string &mode);
py::dict solve_aquifer_nash(const py::dict &inputs);
py::dict simulate_tragedy(const py::dict &inputs);

PYBIND11_MODULE(solver_py, m) {
    m.def("solve_bisection", &solve_bisection);
    m.def("solve_knapsack", [](const std::vector<double>& values, const std::vector<double>& weights, double max_weight) {
        KnapsackResult res = solve_knapsack(values, weights, max_weight);
        py::dict d;
        d["max_value"] = res.max_value;
        d["total_weight"] = res.total_weight;
        d["selected"] = res.selected;
        return d;
    });
    m.def("solve_irrigation_schedule", &solve_irrigation_schedule_impl);
    m.def("solve_aquifer_nash", &solve_aquifer_nash, "Calculates sustainable tariff for N agents");
    m.def("simulate_tragedy", &simulate_tragedy, "Projects aquifer depletion timeline");
}
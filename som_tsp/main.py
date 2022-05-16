import json

from .reader import read_instance
from .solver import Solver
from itertools import product


def solve(filename: str):
    instances = read_instance(filename)

    solver = Solver(instances)
    solver.solve()


def tune(filename: str):
    grid = {
        "node_coefficient": [2, 4, 8, 16, 32],
        "iterations": [1000000],
        "neighborhood_radius_decay": [0.8, 0.9, 0.99, 0.999, 0.9997],
        "learning_rate": [0.7, 0.8, 0.9],
        "learning_rate_decay": [0.8, 0.9, 0.99, 0.999, 0.9999, 0.99997],
    }

    def grid_parameters(parameters):
        for params in product(*parameters.values()):
            yield dict(zip(parameters.keys(), params))

    parameter_count = len(list(grid_parameters(grid)))
    print(f"{parameter_count} parameters to tuples to explore")

    instances = read_instance(filename)

    best_model = None
    best_params = None
    min_route_length = float("inf")
    next_percentage = 1
    for i, params in enumerate(grid_parameters(grid)):
        solver = Solver(instances, False, params)
        solver.solve()
        route_length = solver.calc_route_length()
        if route_length < min_route_length:
            min_route_length = route_length
            best_model = solver
            best_params = params
        if 100 * i / parameter_count > next_percentage:
            print(f"{next_percentage}% done")
            next_percentage = int(100 * i / parameter_count) + 1

    best_model.save_map()
    print(f"Minimum route distance: {min_route_length}")
    print("Found Parameters:")
    print(json.dumps(best_params))

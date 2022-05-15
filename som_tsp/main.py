from .reader import read_instance
from .solver import Solver


def solve(filename: str):
    instances = read_instance(filename)

    solver = Solver(instances)
    solver.solve()

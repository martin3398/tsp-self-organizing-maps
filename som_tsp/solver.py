import numpy as np

from som_tsp.config import default_config
from som_tsp.plot import plot_map

Instances = np.ndarray
Nodes = np.ndarray


class Solver:
    def __init__(self, instances: Instances, verbose=True, config=default_config):
        self.config = config
        self.instances = self.normalize_problem(instances)
        self.verbose = verbose

        self.num_nodes = len(self.instances) * self.config["node_coefficient"]
        self.nodes = np.random.rand(self.num_nodes, 2)

    def solve(self):
        learning_rate = self.config["learning_rate"]
        neighborhood_radius = self.num_nodes

        for i in range(self.config["iterations"]):
            self.calc_step(neighborhood_radius, learning_rate)

            # Decay learning rate
            learning_rate = self.config["learning_rate_decay"] * learning_rate
            neighborhood_radius *= self.config["neighborhood_radius_decay"]

            if neighborhood_radius < 1 or learning_rate < 0.001:
                break

        if self.verbose:
            print(f"Finished after {i} iterations")
            self.save_map()

    def save_map(self):
        plot_map(f"map.png", self.instances, self.nodes, self.calc_route())

    def calc_step(self, neighborhood_radius: float, learning_rate: float):
        instance = self.instances[np.random.randint(len(self.instances)), :]

        # Get best matching unit
        bmu_index = np.linalg.norm(self.nodes - instance, axis=1).argmin()

        # Get gaussian filter for the neighborhood
        gaussian_filter = Solver.get_gaussian_filter(bmu_index, neighborhood_radius // 10, len(self.nodes))

        # Apply filter to update neighborhood
        self.nodes = self.nodes + learning_rate * gaussian_filter * (instance - self.nodes)

    def calc_route(self) -> np.ndarray:
        nearest = [
            (i, np.linalg.norm(self.nodes - instance, axis=1).argmin()) for i, instance in enumerate(self.instances)
        ]
        nearest.sort(key=lambda x: x[1])
        nearest = list(map(lambda x: x[0], nearest))
        nearest = self.instances[nearest]
        return nearest

    def calc_route_length(self) -> float:
        route = self.calc_route()
        return sum(np.linalg.norm(route[:-1] - route[1:], axis=1)) + np.linalg.norm(route[-1] - route[0])

    @staticmethod
    def get_gaussian_filter(bmu: np.ndarray, radius: float, num_instances: int) -> np.ndarray:
        abs_dists = np.abs(bmu - np.arange(num_instances))
        dists = np.minimum(abs_dists, num_instances - abs_dists)
        return np.exp(-(dists**2) / (2 * max(radius, 1) ** 2)).reshape((-1, 1))

    @staticmethod
    def normalize_problem(instance: Instances) -> Instances:
        for dim in [0, 1]:
            min_dim = np.ndarray.min(instance[:, dim])
            max_dim = np.ndarray.max(instance[:, dim])

            instance[:, dim] = (instance[:, dim] - min_dim) / (max_dim - min_dim)

        return instance

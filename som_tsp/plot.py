import os
from os import path

import matplotlib.pyplot as plt
import numpy as np

from som_tsp.config import output_dir


def plot_map(name: str, instances: np.ndarray, nodes: np.ndarray, route: np.ndarray):
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    ax.scatter(nodes[:, 0], nodes[:, 1], c="blue")

    ax.plot(route[:, 0], route[:, 1], c="green", linewidth=4)
    ax.plot(route[[-1, 0], 0], route[[-1, 0], 1], c="green", linewidth=3)

    ax.scatter(instances[:, 0], instances[:, 1], c="red")

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.savefig(f"{output_dir}/{name}")
    plt.close()

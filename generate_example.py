#!/bin/env python3
import json
import sys

import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./generate_example.py <instance_count> <filename>")
        exit(-1)

    instances = np.random.rand(int(sys.argv[1]), 2).tolist()
    with open(sys.argv[2], "w") as file:
        json.dump(instances, file, indent=2)

#!/bin/env python3
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./tune_hyperparameters.py <filename>")
        exit(-1)

    from som_tsp.main import tune

    tune(str(sys.argv[1]))

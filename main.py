#!/bin/env python3
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./main.py <filename>")
        exit(-1)

    from som_tsp.main import solve

    solve(str(sys.argv[1]))

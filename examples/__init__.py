import numpy as np
import scipy.sparse as sp

import fastlap

__all__ = ["run_example"]


def run_example():
    """Run a quick example demonstrating fastlap usage."""
    cost_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    print("Cost matrix:")
    for row in cost_matrix:
        print(f"  {row}")

    print(f"\nSupported algorithms: {fastlap.get_supported_algorithms()}\n")

    for algo in fastlap.get_supported_algorithms():
        total_cost, row_assign, col_assign = fastlap.solve_lap(cost_matrix, algorithm=algo)
        print(f"{algo:12s} -> cost={total_cost:.1f}, rows={row_assign}, cols={col_assign}")


if __name__ == "__main__":
    run_example()

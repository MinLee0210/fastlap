import time
import fastlap
import numpy as np
from scipy.optimize import linear_sum_assignment
import lap


if __name__ == "__main__":
    # Quick example for a 5x5 matrix
    cols = 5
    rows = 5
    algos = ["lapjv", "hungarian", "auction", "dantzig", "subgradient"]
    matrix = np.random.rand(rows, cols)
    
    print(f"Testing on a {rows}x{cols} matrix:\n")
    for algo in algos:
        print(f"Algorithm: {algo}")
        start = time.time()
        fastlap_cost, fastlap_rows, fastlap_cols = fastlap.solve_lap(matrix, algo)
        fastlap_end = time.time()
        print(f"fastlap.{algo}: Time={fastlap_end - start:.8f}s, Cost={fastlap_cost:.6f}")
        
    print("\nAlgorithm: scipy hungarian")
    start = time.time()
    scipy_rows, scipy_cols = linear_sum_assignment(matrix)
    scipy_cost = matrix[scipy_rows, scipy_cols].sum()
    scipy_end = time.time()
    print(f"scipy.optimize.linear_sum_assignment: Time={scipy_end - start:.8f}s, Cost={scipy_cost:.6f}")

    print("\nAlgorithm: lap.lapjv")
    start = time.time()
    lap_cost, lap_rows, lap_cols = lap.lapjv(matrix, extend_cost=True)
    lap_end = time.time()
    print(f"lap.lapjv: Time={lap_end - start:.8f}s, Cost={lap_cost:.6f}")


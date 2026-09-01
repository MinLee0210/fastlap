import time
import pytest
import numpy as np
from scipy.optimize import linear_sum_assignment
import fastlap
from conftest import generate_test_matrix, fastlap_execute, scipy_execute, lap_execute


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_performance(size):
    """Measure execution time and report for fastlap and SciPy."""
    matrix = generate_test_matrix(size)
    iterations = 1000  # Number of runs for stable timing

    print(f"\nMatrix size: {size}x{size}")

    for algo in ["lapjv", "hungarian"]:
        total_time = 0.0
        for _ in range(iterations):
            start_time = time.time()
            fastlap_execute(matrix, algo)
            total_time += time.time() - start_time
        avg_time = total_time / iterations
        print(f"fastlap.{algo}: Avg Time={avg_time:.8f}s")

    # SciPy
    total_time = 0.0
    for _ in range(iterations):
        start_time = time.time()
        scipy_execute(matrix)
        total_time += time.time() - start_time
    avg_time = total_time / iterations
    print(f"scipy: Avg Time={avg_time:.8f}s")

    # lap
    total_time = 0.0
    for _ in range(iterations):
        start_time = time.time()
        lap_execute(matrix)
        total_time += time.time() - start_time
    avg_time = total_time / iterations
    print(f"lap: Avg Time={avg_time:.8f}s")

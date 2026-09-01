import numpy as np
from scipy.optimize import linear_sum_assignment
import lap
import fastlap

# Set random seed for reproducibility
np.random.seed(42)


def generate_test_matrix(size, max_val=100.0):
    """Generate a random square matrix of given size with float64 values."""
    return np.random.uniform(0, max_val, (size, size)).astype(np.float64)


def fastlap_execute(
    matrix: np.ndarray, algo: str
) -> tuple[float, list[int], list[int]]:
    """Execute fastlap with the given algorithm."""
    return fastlap.solve_lap(matrix, algo)


def scipy_execute(matrix: np.ndarray) -> tuple[float, list[int], list[int]]:
    """Execute scipy with the given algorithm."""
    rows, cols = linear_sum_assignment(matrix)
    cost = matrix[rows, cols].sum()
    return cost, rows.tolist(), cols.tolist()


def lap_execute(matrix: np.ndarray) -> tuple[float, list[int], list[int]]:
    """Execute lap with the given algorithm."""
    cost, rows, cols = lap.lapjv(matrix, extend_cost=True)
    return cost, rows.tolist(), cols.tolist()

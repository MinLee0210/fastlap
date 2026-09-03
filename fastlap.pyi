from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, overload
import numpy as np
import numpy.typing as npt

Algorithm = Literal[
    "lapjv",
    "hungarian",
    "lapmod",
    "lapjvsp",
    "subgradient",
    "auction",
    "dantzig",
    "sinkhorn",
    "ssp",
    "cost_scaling",
    "greedy",
]

DualAlgorithm = Literal[
    "lapjv",
    "subgradient",
    "sinkhorn",
    "dantzig",
]

MatrixLike = Union[
    npt.NDArray[Any],
    Sequence[Sequence[float]],
    Any,  # scipy.sparse.csr_matrix
]

BatchLike = Union[
    npt.NDArray[Any],  # (B, N, M) stacked dense matrices
    Sequence[MatrixLike],
]

LapSolution = Tuple[float, List[Optional[int]], List[Optional[int]]]

LapSolutionWithDuals = Tuple[
    float,
    List[Optional[int]],
    List[Optional[int]],
    List[float],  # row duals u
    List[float],  # column duals v
]

def solve_lap(
    cost_matrix: MatrixLike,
    algorithm: Algorithm = "lapjv",
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> LapSolution:
    """Solve a Linear Assignment Problem (minimum-cost or maximum-weight bipartite matching)."""
    ...

def solve_lap_batch(
    cost_matrices: BatchLike,
    algorithm: Algorithm = "lapjv",
    maximize: bool = False,
    cost_limit: Optional[float] = None,
    n_threads: Optional[int] = None,
) -> List[LapSolution]:
    """Solve many independent LAPs in parallel using Rayon. A 3D (B, N, M)
    ndarray is treated as B stacked matrices; `n_threads` limits workers."""
    ...

def solve_lap_weighted(
    cost_matrix: MatrixLike,
    weights: MatrixLike,
    algorithm: Algorithm = "lapjv",
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> LapSolution:
    """Solve a Linear Assignment Problem with per-entry cost weights."""
    ...

def solve_lbap(
    cost_matrix: MatrixLike,
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> LapSolution:
    """Solve the Linear Bottleneck Assignment Problem (min_pi max_i C[i, pi(i)])."""
    ...

def solve_lbap_batch(
    cost_matrices: BatchLike,
    maximize: bool = False,
    cost_limit: Optional[float] = None,
    n_threads: Optional[int] = None,
) -> List[LapSolution]:
    """Solve many Linear Bottleneck Assignment Problems in parallel."""
    ...

def solve_lap_kbest(
    cost_matrix: MatrixLike,
    k: int = 3,
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> List[LapSolution]:
    """Find the K-best (ranked) assignments using Murty's algorithm."""
    ...

def solve_lap_duals(
    cost_matrix: MatrixLike,
    algorithm: DualAlgorithm = "lapjv",
) -> LapSolutionWithDuals:
    """Solve a min-cost LAP and return (cost, row_assign, col_assign, u, v)
    with the optimal dual potentials u (rows) and v (columns)."""
    ...

def linear_sum_assignment(
    cost_matrix: MatrixLike,
    maximize: bool = False,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Drop-in replacement for scipy.optimize.linear_sum_assignment."""
    ...

def lapjvx(
    cost_matrix: MatrixLike,
    maximize: bool = False,
    cost_limit: Optional[float] = None,
    return_cost: bool = True,
) -> Union[
    Tuple[float, npt.NDArray[np.int64], npt.NDArray[np.int64]],
    Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
]:
    """lapx.lapjvx style: scipy-compatible aligned row/col index arrays."""
    ...

def assignment_pairs(
    cost_matrix: MatrixLike,
    maximize: bool = False,
    cost_limit: Optional[float] = None,
    return_cost: bool = True,
) -> Union[Tuple[float, npt.NDArray[np.int64]], npt.NDArray[np.int64]]:
    """lapx.lapjvxa style: return the assignment as a (K, 2) array of [row, col]."""
    ...

@overload
def lapjv(
    cost: MatrixLike,
    extend_cost: bool = True,
    cost_limit: Optional[float] = None,
    return_cost: Literal[True] = True,
) -> Tuple[float, npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...

@overload
def lapjv(
    cost: MatrixLike,
    extend_cost: bool = True,
    cost_limit: Optional[float] = None,
    return_cost: Literal[False] = False,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...

@overload
def lapjv(
    cost: MatrixLike,
    extend_cost: bool = True,
    cost_limit: Optional[float] = None,
    return_cost: bool = True,
) -> Union[
    Tuple[float, npt.NDArray[np.int32], npt.NDArray[np.int32]],
    Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]],
]:
    """Drop-in replacement for lap.lapjv / lapx.lapjv."""
    ...

def get_supported_algorithms() -> List[str]:
    """Return the list of supported algorithm names."""
    ...

class lap:
    @staticmethod
    def lapjv(
        cost: MatrixLike,
        extend_cost: bool = True,
        cost_limit: Optional[float] = None,
        return_cost: bool = True,
    ) -> Union[
        Tuple[float, npt.NDArray[np.int32], npt.NDArray[np.int32]],
        Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]],
    ]:
        """Drop-in replacement for lap.lapjv."""
        ...

class compat:
    @staticmethod
    def linear_sum_assignment(
        cost_matrix: MatrixLike,
        maximize: bool = False,
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Drop-in replacement for scipy.optimize.linear_sum_assignment."""
        ...

    @staticmethod
    def lapjvx(
        cost_matrix: MatrixLike,
        maximize: bool = False,
        cost_limit: Optional[float] = None,
        return_cost: bool = True,
    ) -> Union[
        Tuple[float, npt.NDArray[np.int64], npt.NDArray[np.int64]],
        Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
    ]:
        """lapx.lapjvx style: scipy-compatible aligned row/col index arrays."""
        ...

    @staticmethod
    def assignment_pairs(
        cost_matrix: MatrixLike,
        maximize: bool = False,
        cost_limit: Optional[float] = None,
        return_cost: bool = True,
    ) -> Union[Tuple[float, npt.NDArray[np.int64]], npt.NDArray[np.int64]]:
        """lapx.lapjvxa style: return the assignment as a (K, 2) array of [row, col]."""
        ...

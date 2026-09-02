from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, overload
import numpy as np
import numpy.typing as npt

Algorithm = Literal[
    "lapjv",
    "hungarian",
    "lapmod",
    "subgradient",
    "auction",
    "dantzig",
    "sinkhorn",
    "ssp",
    "cost_scaling",
    "greedy",
]

MatrixLike = Union[
    npt.NDArray[Any],
    Sequence[Sequence[float]],
    Any,  # scipy.sparse.csr_matrix
]

LapSolution = Tuple[float, List[Optional[int]], List[Optional[int]]]

def solve_lap(
    cost_matrix: MatrixLike,
    algorithm: Algorithm = "lapjv",
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> LapSolution:
    """Solve a Linear Assignment Problem (minimum-cost or maximum-weight bipartite matching)."""
    ...

def solve_lap_batch(
    cost_matrices: Sequence[MatrixLike],
    algorithm: Algorithm = "lapjv",
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> List[LapSolution]:
    """Solve multiple independent Linear Assignment Problems in parallel using Rayon."""
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
    cost_matrices: Sequence[MatrixLike],
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> List[LapSolution]:
    """Solve multiple Linear Bottleneck Assignment Problems in parallel."""
    ...

def solve_lap_kbest(
    cost_matrix: MatrixLike,
    k: int = 3,
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> List[LapSolution]:
    """Find the K-best (ranked) assignments using Murty's algorithm."""
    ...

def linear_sum_assignment(
    cost_matrix: MatrixLike,
    maximize: bool = False,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Drop-in replacement for scipy.optimize.linear_sum_assignment."""
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

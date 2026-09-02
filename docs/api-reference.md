# API Reference

Full signatures for every public function, mirrored from [`fastlap.pyi`](https://github.com/MinLee0210/fastlap/blob/main/fastlap.pyi). All solving functions return a **`LapSolution`**:

```python
LapSolution = Tuple[float, List[Optional[int]], List[Optional[int]]]
# (total_cost, row_assign, col_assign)
```

- `row_assign[i]` — column assigned to row `i`, or `None` if unassigned.
- `col_assign[j]` — row assigned to column `j`, or `None` if unassigned.

`cost_matrix` / `cost_matrices` accept a `MatrixLike`: a NumPy array (any numeric dtype), a nested Python sequence, or a `scipy.sparse.csr_matrix`.

`Algorithm` is one of:

```python
"lapjv" | "hungarian" | "lapmod" | "subgradient" | "auction"
| "dantzig" | "sinkhorn" | "ssp" | "cost_scaling" | "greedy"
```

---

## `solve_lap` { #solve_lap }

```python
def solve_lap(
    cost_matrix: MatrixLike,
    algorithm: Algorithm = "lapjv",
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> LapSolution: ...
```

Solve a single Linear Assignment Problem — minimum-cost bipartite matching, or maximum-weight matching with `maximize=True`. Square matrices are solved directly; rectangular matrices are padded internally and unassigned padded rows/columns are reported as `None`.

If `algorithm="lapmod"` and `cost_matrix` is a `scipy.sparse.csr_matrix`, the solve runs on the true sparse adjacency without densifying — see [Sparse Matrices](features/sparse.md).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cost_matrix` | `MatrixLike` | — | An `(n × m)` cost matrix. |
| `algorithm` | `Algorithm` | `"lapjv"` | Which solver to use — see [Algorithms](algorithms/index.md). |
| `maximize` | `bool` | `False` | Find the maximum-weight assignment instead of minimum-cost. |
| `cost_limit` | `float \| None` | `None` | Gating threshold — see [Cost Limit](features/cost-limit.md). |

**Returns:** `LapSolution`.

---

## `solve_lap_batch` { #solve_lap_batch }

```python
def solve_lap_batch(
    cost_matrices: Sequence[MatrixLike],
    algorithm: Algorithm = "lapjv",
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> List[LapSolution]: ...
```

Solve multiple independent Linear Assignment Problems in parallel using Rayon (see [Batch Solving](features/batch.md)). The GIL is released for the duration of the batch.

**Returns:** One `LapSolution` per input matrix, in the same order.

---

## `solve_lap_weighted` { #solve_lap_weighted }

```python
def solve_lap_weighted(
    cost_matrix: MatrixLike,
    weights: MatrixLike,
    algorithm: Algorithm = "lapjv",
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> LapSolution: ...
```

Solve a LAP with per-entry cost weights — see [Weighted Costs](features/weighted.md). The effective optimization cost is `weights[i][j] * cost_matrix[i][j]`; the returned `total_cost` is computed from the **original** (unweighted) matrix. Raises `ValueError` if `cost_matrix` and `weights` don't share a shape.

---

## `solve_lbap` { #solve_lbap }

```python
def solve_lbap(
    cost_matrix: MatrixLike,
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> LapSolution: ...
```

Solve the Linear Bottleneck Assignment Problem — minimises the maximum assigned edge cost, `min_π max_i C[i, π(i)]`. See [Bottleneck (LBAP)](features/bottleneck.md). With `maximize=True`, maximises the minimum edge instead.

---

## `solve_lbap_batch` { #solve_lbap_batch }

```python
def solve_lbap_batch(
    cost_matrices: Sequence[MatrixLike],
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> List[LapSolution]: ...
```

Parallel batch version of `solve_lbap`, same Rayon execution model as `solve_lap_batch`.

---

## `solve_lap_kbest` { #solve_lap_kbest }

```python
def solve_lap_kbest(
    cost_matrix: MatrixLike,
    k: int = 3,
    maximize: bool = False,
    cost_limit: Optional[float] = None,
) -> List[LapSolution]: ...
```

Find the K-best (ranked) assignments using Murty's algorithm — see [K-Best (Murty)](features/kbest.md).

**Returns:** Up to `k` solutions, in increasing order of cost (decreasing order of profit under `maximize=True`). Fewer than `k` come back if the matrix has fewer than `k` distinct feasible assignments.

---

## `linear_sum_assignment` { #linear_sum_assignment }

```python
def linear_sum_assignment(
    cost_matrix: MatrixLike,
    maximize: bool = False,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
```

Drop-in replacement for `scipy.optimize.linear_sum_assignment` — see [Compatibility Layers](features/compat.md#scipy-drop-in). Also available as `fastlap.compat.linear_sum_assignment`. Always solves with `"lapjv"` internally.

**Returns:** `(row_ind, col_ind)` as `int64` NumPy arrays, matching SciPy's format exactly.

---

## `lapjv` { #lapjv }

```python
def lapjv(
    cost: MatrixLike,
    extend_cost: bool = True,
    cost_limit: Optional[float] = None,
    return_cost: bool = True,
) -> Tuple[float, npt.NDArray[np.int32], npt.NDArray[np.int32]] | Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...
```

Drop-in replacement for `lap.lapjv` / `lapx.lapjv` — see [Compatibility Layers](features/compat.md#lapjv-drop-in). Also available as `fastlap.lap.lapjv`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cost` | `MatrixLike` | — | Cost matrix. |
| `extend_cost` | `bool` | `True` | Accepted for compatibility; rectangular padding is always handled automatically. |
| `cost_limit` | `float \| None` | `None` | Maximum allowed cost for a valid assignment. Unassigned pairs return `-1`. |
| `return_cost` | `bool` | `True` | Whether to return the optimal total cost as the first tuple element. |

**Returns:** `(opt_cost, x, y)` if `return_cost=True`, else `(x, y)`. `x` and `y` are `int32` arrays using `-1` for unassigned entries (not `None`).

---

## `get_supported_algorithms` { #get_supported_algorithms }

```python
def get_supported_algorithms() -> List[str]: ...
```

Returns the list of supported algorithm names, in the same order documented in [Algorithms](algorithms/index.md):

```python
>>> fastlap.get_supported_algorithms()
['lapjv', 'hungarian', 'lapmod', 'subgradient', 'auction', 'dantzig', 'sinkhorn', 'ssp', 'cost_scaling', 'greedy']
```

---

## Submodules

### `fastlap.lap`

```python
class lap:
    @staticmethod
    def lapjv(cost, extend_cost=True, cost_limit=None, return_cost=True): ...
```

Same as [`fastlap.lapjv`](#lapjv), namespaced to match `import lap` usage patterns from ByteTrack/BoT-SORT-style codebases.

### `fastlap.compat`

```python
class compat:
    @staticmethod
    def linear_sum_assignment(cost_matrix, maximize=False): ...
```

Same as [`fastlap.linear_sum_assignment`](#linear_sum_assignment), namespaced to match `from scipy.optimize import ...`-style usage.

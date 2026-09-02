# Cost Limit (Gating)

Multi-object trackers (ByteTrack, BoT-SORT, DeepSORT, SORT) don't want *every* row matched to *some* column — a detection shouldn't be linked to a track if the association cost (e.g. IoU distance, appearance distance) is implausibly high. `cost_limit` builds this gating directly into the solver.

```python
# Threshold cost at 10.0 — any pair exceeding 10.0 is unassigned (None)
cost, rows, cols = fastlap.solve_lap(cost_matrix, cost_limit=10.0)
```

## How it works

fastlap always solves the *unconstrained* optimal assignment first, then walks the result and unassigns (sets to `None`) any pair whose cost violates the limit:

- **Minimize mode** (default): an assignment `(i, j)` is rejected if `matrix[i][j] > cost_limit`.
- **Maximize mode** (`maximize=True`): an assignment `(i, j)` is rejected if `matrix[i][j] < cost_limit`.

`total_cost` is always recomputed from the surviving (non-`None`) assignments, so it reflects exactly what's returned.

```python
import numpy as np
import fastlap

matrix = np.array([
    [1.0, 50.0, 50.0],
    [50.0, 5.0, 50.0],
    [50.0, 50.0, 20.0],
])

cost, rows, cols = fastlap.solve_lap(matrix, algorithm="lapjv", cost_limit=10.0)
print(rows)  # [0, 1, None] — row 2's optimal match (cost 20) is gated out
print(cost)  # 6.0 — only the surviving pairs (1 + 5) are counted
```

## Works everywhere

`cost_limit` is supported by every entry point that returns a `LapSolution`:

- [`solve_lap`](../api-reference.md#solve_lap) (dense and sparse/LAPMOD)
- [`solve_lap_batch`](../api-reference.md#solve_lap_batch)
- [`solve_lap_weighted`](../api-reference.md#solve_lap_weighted) — gating uses the *original* (unweighted) costs
- [`solve_lbap`](../api-reference.md#solve_lbap) / [`solve_lbap_batch`](../api-reference.md#solve_lbap_batch)
- [`solve_lap_kbest`](../api-reference.md#solve_lap_kbest) — applied independently to each of the K solutions
- [`lapjv`](../features/compat.md#lapjv-drop-in) drop-in shim, matching `lap.lapjv`'s own `cost_limit` semantics

## Sparse (LAPMOD) example

```python
import scipy.sparse as sp

csr = sp.csr_matrix(matrix)
cost, rows, cols = fastlap.solve_lap(csr, algorithm="lapmod", cost_limit=10.0)
```

## Maximize mode example

```python
profit = np.array([
    [100.0, 10.0],
    [10.0, 5.0],
])

# Maximize optimal is (0→0: 100, 1→1: 5). With limit=50, pair 1→1 (profit 5 < 50) is gated out.
cost, rows, cols = fastlap.solve_lap(profit, algorithm="lapjv", maximize=True, cost_limit=50.0)
print(rows)  # [0, None]
print(cost)  # 100.0
```

# Weighted Costs

`solve_lap_weighted` multiplies each cost entry by a per-element weight *before* solving — useful when you want to bias the assignment (e.g. down-weighting a noisy feature-matching cost, or emphasizing a particular sub-region of the matrix) without discarding the original cost values.

```python
import numpy as np
import fastlap

cost    = np.array([[1, 2], [3, 4]], dtype=np.float64)
weights = np.array([[1, 0.5], [0.5, 1]], dtype=np.float64)

total, rows, cols = fastlap.solve_lap_weighted(cost, weights, algorithm="lapjv")
```

## The returned cost is unweighted

The solver optimizes over `weight[i][j] * cost[i][j]`, but `total_cost` is always computed from the **original, unweighted** `cost_matrix`. This matters: the assignment reflects your weighting preference, but the reported cost stays meaningful in the original units (e.g. real distance, real dollars).

`cost_matrix` and `weights` must have the same shape — a mismatch raises `ValueError`.

## Parameters

Same shape as [`solve_lap`](../api-reference.md#solve_lap), plus the `weights` argument:

```python
fastlap.solve_lap_weighted(
    cost_matrix,
    weights,
    algorithm="lapjv",
    maximize=False,
    cost_limit=None,
)
```

`cost_limit` gating (see [Cost Limit](cost-limit.md)) is applied against the **original** cost values, consistent with the returned `total_cost`:

```python
costs = np.array([[2.0, 50.0], [50.0, 20.0]])
weights = np.ones_like(costs)
cost, rows, cols = fastlap.solve_lap_weighted(costs, weights, algorithm="lapjv", cost_limit=10.0)
print(rows)  # [0, None]
print(cost)  # 2.0
```

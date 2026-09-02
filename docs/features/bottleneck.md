# Bottleneck Assignment (LBAP)

The **Linear Bottleneck Assignment Problem** asks a different question than the standard LAP: instead of minimising the *sum* of assigned costs, minimise the **maximum** cost edge used anywhere in the assignment:

$$
\min_{\pi} \max_i C_{i, \pi(i)}
$$

```python
bottleneck_cost, rows, cols = fastlap.solve_lbap(cost_matrix)
```

## Why this is a different problem

Total-cost-optimal and bottleneck-optimal assignments can disagree. Consider:

```python
import numpy as np
import fastlap

matrix = np.array([
    [1.0, 2.0, 10.0],
    [2.0, 1.0, 10.0],
    [10.0, 10.0, 5.0],
])

# Total-sum optimum: (0→1: 2, 1→0: 2, 2→2: 5) → sum = 9, max edge = 5
cost, rows, cols = fastlap.solve_lbap(matrix)
print(cost)  # 5.0 — the largest edge used, not the sum
```

Bottleneck assignment is the right formulation whenever a single bad match dominates the outcome more than the aggregate cost does — e.g. job scheduling where you care about the slowest machine's finish time, not the total machine-time spent.

## How it works

fastlap binary-searches over the sorted set of distinct finite edge costs in the matrix, using [Hopcroft–Karp](https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm) maximum-cardinality bipartite matching at each candidate threshold `T` to check whether a perfect matching exists using only edges with `cost <= T`. The smallest feasible `T` is the bottleneck cost.

## Parameters

```python
fastlap.solve_lbap(
    cost_matrix,
    maximize=False,
    cost_limit=None,
)
```

- `maximize=True` finds the assignment that **maximises the minimum** edge used instead — the "worst edge is as good as possible" variant.
- Rectangular matrices are supported the same way as [`solve_lap`](../api-reference.md#solve_lap): unmatched rows/columns come back as `None`.

```python
matrix = np.array([[1.0, 9.0, 3.0, 8.0], [7.0, 2.0, 6.0, 4.0]])
cost, rows, cols = fastlap.solve_lbap(matrix)
print(cost, rows, cols)  # 2.0 [0, 1] [0, 1, None, None]
```

## Batch LBAP

```python
matrices = [np.random.rand(20, 20) for _ in range(200)]
results = fastlap.solve_lbap_batch(matrices)
```

See [Batch Solving](batch.md) for the general parallel-batch model.

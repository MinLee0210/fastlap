# Batch Solving

`solve_lap_batch` solves many **independent** assignment problems in parallel, spread across all available CPU cores via [Rayon](https://docs.rs/rayon). The Python GIL is released for the duration of the batch (`py.allow_threads`), so the parallelism is real, not GIL-limited.

```python
import numpy as np
import fastlap

matrices = [np.random.rand(50, 50) for _ in range(500)]
results = fastlap.solve_lap_batch(matrices, algorithm="lapjv")

# Each result is (cost, row_assign, col_assign)
costs = [r[0] for r in results]
```

## When to use it

Reach for `solve_lap_batch` whenever you have a list of cost matrices that don't depend on each other — for example, running the same tracker's association step across many camera streams, or solving assignment problems for many independent scheduling windows in one call.

## Parameters

`solve_lap_batch` accepts the same `algorithm`, `maximize`, and `cost_limit` keywords as [`solve_lap`](../api-reference.md#solve_lap), applied uniformly to every matrix in the list:

```python
results = fastlap.solve_lap_batch(
    matrices,
    algorithm="lapjv",
    maximize=False,
    cost_limit=10.0,
)
```

## Sparse input in a batch

If `algorithm="lapmod"` and an individual matrix is a `scipy.sparse.csr_matrix`, that entry is solved on its sparse adjacency directly — the same true-sparse fast path `solve_lap` uses (see [Sparse Matrices](sparse.md)). Dense entries in the same batch are handled normally, so you can freely mix sparse and dense matrices in one `solve_lap_batch` call when `algorithm="lapmod"`.

## LBAP batches

The bottleneck variant has its own batch entry point, [`solve_lbap_batch`](../api-reference.md#solve_lbap_batch), with the same parallel-Rayon execution model:

```python
matrices = [np.random.rand(20, 20) for _ in range(200)]
results = fastlap.solve_lbap_batch(matrices)
```

# Sparse Matrices

Most cost matrices you'd hand to a LAP solver from a tracking or candidate-gated pipeline are **mostly empty** — a detection only has a handful of plausible track candidates, not every track in the scene. fastlap's `lapmod` algorithm is built specifically for this shape of problem.

```python
import scipy.sparse as sp
import fastlap

csr = sp.csr_matrix(dense_matrix)
cost, rows, cols = fastlap.solve_lap(csr, algorithm="lapmod")
```

## True sparse solving — no densification

Every other algorithm in fastlap accepts a `scipy.sparse.csr_matrix` too, but internally converts it to a dense `nrows × ncols` array first (`matrix::extract_sparse_matrix`) — fine when the matrix is small, wasteful once it's large and mostly empty.

`lapmod` is different: when you pass a CSR matrix with `algorithm="lapmod"`, fastlap routes to `extract_sparse_adjacency` and solves directly against the row-adjacency list of explicit `(col, cost)` entries. The matrix is **never densified**. Missing `(row, col)` pairs are simply treated as infinitely costly (forbidden) — the same convention used when densifying elsewhere, just without paying for the full `nrows × ncols` allocation.

This is what makes `lapmod` scale with the number of **explicit edges**, not the full matrix area — the difference between an `O(nnz)`-ish solve and an `O(n²)` one on a graph with, say, 100,000 nodes but only a few edges per node.

## Rectangular sparse input

Non-square sparse matrices still need *some* place to send a displaced match during augmentation. Rather than densifying the whole matrix, fastlap adds a small number of explicit high-cost slack edges — `dim × |nrows − ncols|` of them — so the padding cost scales with the **rectangular imbalance**, not with `nrows × ncols`.

## Combining with other features

Sparse LAPMOD input works with the features you'd expect:

```python
# Gating threshold on a sparse matrix
cost, rows, cols = fastlap.solve_lap(csr, algorithm="lapmod", cost_limit=10.0)

# Maximize mode
cost, rows, cols = fastlap.solve_lap(csr, algorithm="lapmod", maximize=True)

# Mixed sparse/dense batch (see Batch Solving)
results = fastlap.solve_lap_batch([csr, dense_matrix], algorithm="lapmod")
```

!!! note "The sparse fast path is `lapmod`-only"
    Every other algorithm name still works with a `scipy.sparse.csr_matrix` input — it's just densified first. If you need the true sparse-adjacency solve, use `algorithm="lapmod"`.

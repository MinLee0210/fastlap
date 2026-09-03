# LAPJVsp

A **true-sparse Jonker–Volgenant solver** — the sparse sibling of [LAPJV](lapjv.md), built to run on a `scipy.sparse.csr_matrix` without ever densifying it. Together with [LAPMOD](lapmod.md) it is one of only two algorithms in fastlap with a true sparse fast path.

!!! info "Prerequisites"
    [Augmenting Paths](concepts.md#augmenting-paths) and [LAPJV](lapjv.md) — LAPJVsp is LAPJV's column-reduction + reduction-transfer pipeline, adapted to a row-adjacency list and finished by LAPMOD's sparse shortest-augmenting-path search.

## Why this approach

[LAPJV](lapjv.md) is fast because its O(n²) column-reduction phase resolves most rows for free, leaving only a handful to pay for a full augmenting-path search. But LAPJV's reduction scans *every row of every column*, which assumes a dense matrix.

LAPJVsp keeps the same two-phase structure but makes each phase **sparse**:

- Column reduction only inspects a column's **explicit** entries (a missing `(row, col)` pair is implicitly infinite, so it can never be a column's cheapest candidate).
- Reduction transfer only walks each row's own adjacency list.
- The few rows the reduction leaves unclaimed are finished by the same heap-based sparse shortest-augmenting-path search [LAPMOD](lapmod.md#how-it-works) uses, **warm-started** from the duals the reduction already built.

The result is a solver that scales with the number of **explicit edges** — like LAPMOD — but that usually resolves most rows for free during reduction, like dense LAPJV.

## How it works

All three phases operate on a transposed copy of the sparse adjacency (built once, O(nnz)):

1. **Sparse column reduction.** For each column, scan only its explicit entries to find the cheapest reachable row, and tentatively claim that row — exactly LAPJV's phase 1, just without iterating rows that don't have the column in their adjacency. Each column's minimum becomes its dual `v[j]`; a row claimed by more than one column keeps only its cheapest claimant. This yields a partial matching plus a feasible dual `v` for free.
2. **Reduction transfer.** For every row claimed by exactly one column, tighten that column's dual using the row's second-best reduced cost — the same slack-shrinking step dense LAPJV performs, restricted to the row's explicit edges.
3. **Warm-started sparse SAP.** Compute feasible row duals `u[i] = min_j (cost[i][j] − v[j])`, then hand the partial matching and `(u, v)` to the sparse augmenting-path search. Only rows the reduction left unclaimed pay for a Dijkstra search; the partial matching is already optimal for the rows it covers.

Rectangular input gets the same high-cost slack-edge padding LAPMOD uses (see [Sparse Matrices](../features/sparse.md)) — cost scales with the rectangular imbalance, not `nrows × ncols`.

## Pseudocode

```text
function LAPJVsp(adjacency, nrows, ncols):
    dim = max(nrows, ncols)
    add high-cost slack edges so every row reaches a virtual column, and
    (if rectangular) every virtual row reaches every real column

    col_adjacency = transpose(adjacency)          # O(nnz)

    # Phase 1: sparse column reduction
    for each column j:
        (i*, best) = cheapest explicit (row, cost) in col_adjacency[j]
        v[j] = best; claim row i* for column j
        a row keeps only its cheapest claimant

    # Phase 2: reduction transfer
    for each row i claimed by exactly one column:
        v[assigned_column] -= second-cheapest reduced cost of row i

    # Phase 3: warm-started completion
    u[i] = min over row i's edges of (cost − v[col])
    run the sparse shortest-augmenting-path search warm-started from (u, v)
        only for rows the reduction left unclaimed

    return assignment trimmed back to (nrows, ncols)
```

## Complexity

| | Cost |
|---|---|
| **Time** | O(nnz) for phases 1–2 (one pass over each edge to build the transpose, then each column/row scans only its own edges); phase 3 pays `O(E_touched · log E_touched)` per still-unclaimed row. **O(n² log n)** worst case on dense input |
| **Space** | O(n + nnz) for sparse input — the adjacency list, its transpose, and O(n) dual/heap bookkeeping; **O(n²)** if the input is effectively dense |

## Worked example

Reuse LAPMOD's worked structure — a sparse graph where most pairs are forbidden:

| | col 0 | col 1 | col 2 |
|---|---|---|---|
| **row 0** | 2 | 5 | ✗ |
| **row 1** | ✗ | 3 | ✗ |
| **row 2** | 4 | ✗ | 1 |

**Phase 1** inspects each column's explicit edges only:

- Col 0: candidates row 0 (2), row 2 (4) → cheapest row 0, `v[0] = 2`, row 0 claims col 0.
- Col 1: candidates row 0 (5), row 1 (3) → cheapest row 1, `v[1] = 3`, row 1 claims col 1.
- Col 2: candidate row 2 (1) → `v[2] = 1`, row 2 claims col 2.

Every row is resolved by reduction alone — **phase 3 has nothing to do**, and the optimum is `row 0→0, row 1→1, row 2→2`, cost `2 + 3 + 1 = 6`. Notice the search never once considered row 0–col 2 or row 1–col 0: those pairs don't exist in the adjacency list, so they're skipped, not scanned and rejected.

```python
import scipy.sparse as sp
import fastlap

csr = sp.csr_matrix(
    ([2, 5, 3, 4, 1], ([0, 0, 1, 2, 2], [0, 1, 1, 0, 2])),
    shape=(3, 3),
)
total, rows, cols = fastlap.solve_lap(csr, algorithm="lapjvsp")
print(total, rows)  # 6.0 [0, 1, 2]
```

## LAPJVsp vs LAPMOD

Both solve directly on sparse adjacency with the same forbidden-edge convention, and both return identical optimal costs. The difference is *how* they get there:

- **LAPMOD** runs the sparse augmenting-path search from scratch for every row (cold start).
- **LAPJVsp** resolves most rows in the sparse column-reduction phase and only searches for the leftovers, warm-started from reduction-built duals.

On many real sparse matrices — where each row/column has a handful of clearly-best candidates — LAPJVsp's reduction resolves most rows for free, mirroring why dense LAPJV usually beats a cold Hungarian solve. Which is faster in practice depends on the graph; benchmark on your own data if it matters.

## Common pitfalls

!!! danger "A missing entry means forbidden, not free"
    The exact same convention and pitfall as [LAPMOD](lapmod.md#common-pitfalls): an absent `(row, col)` pair is an edge that doesn't exist. A CSR matrix where "no data yet" reads back as `0.0` instead of being genuinely absent from `indices`/`data` will silently offer edges you never intended, often yielding a suspiciously cheap wrong answer.

Passing a `scipy.sparse.csr_matrix` to any algorithm **other than** `"lapmod"` or `"lapjvsp"` still works, but densifies first — same correct answer, no sparse fast path.

## When to use it

Use `"lapjvsp"` for large, genuinely sparse CSR matrices where you want the JV reduction warm start instead of LAPMOD's cold search — candidate-gated tracking, big mostly-empty bipartite graphs. If your input is dense, plain [LAPJV](lapjv.md) is faster (no heap log factor). See [Sparse Matrices](../features/sparse.md) for the full writeup.

## References

- R. Jonker & A. Volgenant, *"A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems"*, Computing, 1987.
- D. F. Crouse, *"On Implementing 2D Rectangular Assignment Algorithms"*, IEEE Transactions on Aerospace and Electronic Systems, 2016 (the LAPJVsp variant used by SciPy's `min_weight_full_bipartite_matching`).

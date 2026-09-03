<div align="center">

<img src="https://raw.githubusercontent.com/MinLee0210/fastlap/main/docs/static/fastlap.png" alt="fastlap — high-performance linear assignment problem solver in Python and Rust" width="400"/>

# fastlap

**Fast Linear Assignment Problem (LAP) Solver for Python — Powered by Rust**

[![PyPI version](https://img.shields.io/pypi/v/fastlap?color=blue&label=PyPI)](https://pypi.org/project/fastlap/)
[![Python](https://img.shields.io/pypi/pyversions/fastlap?label=Python)](https://pypi.org/project/fastlap/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/MinLee0210/fastlap/actions/workflows/ci.yml/badge.svg)](https://github.com/MinLee0210/fastlap/actions)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blueviolet)](https://minlee0210.github.io/fastlap/)

</div>

**📖 Full documentation: [minlee0210.github.io/fastlap](https://minlee0210.github.io/fastlap/)**

---

**fastlap** solves the [linear assignment problem](https://en.wikipedia.org/wiki/Assignment_problem) — minimum-cost bipartite matching, maximum weight matching (`maximize=True`), bottleneck assignment (`solve_lbap`), and ranked $K$-best assignments (`solve_lap_kbest`) — at high speed from Python. It ships **eleven algorithmically distinct solvers** behind a single `solve_lap()` call, with **parallel batch solving** (3D ndarray batches + `n_threads`), **gating threshold support (`cost_limit`)**, **optimal dual extraction (`solve_lap_duals`)**, **weighted costs**, and **drop-in compatibility layers** for SciPy and `lap`/`lapx`.

If you work with **object tracking (ByteTrack, BoT-SORT, DeepSORT)**, **task scheduling**, **resource allocation**, **feature matching**, or **combinatorial optimisation**, fastlap gives you a drop-in Rust accelerator for the core assignment step.

## Why fastlap?

| | fastlap (Rust) | scipy.optimize | lap / lapx (C++) |
|---|---|---|---|
| **Speed** | Sub-ms on 100×100 | ~ms | ~ms |
| **Algorithms** | 11 (algorithmically distinct) + LBAP + K-Best | 1 | 1 |
| **Gating threshold** | `cost_limit=...` built-in | manual filtering | `cost_limit` |
| **Bottleneck (LBAP)** | `solve_lbap` built-in | no | no |
| **K-Best (Murty)** | `solve_lap_kbest` built-in | no | no |
| **Batch parallel** | `solve_lap_batch` (Rayon) | manual | manual |
| **Weighted costs** | built-in | no | no |
| **Maximize mode** | `maximize=True` | manual negation | manual negation |
| **Sparse-aware solve** | LAPMOD & LAPJVsp skip densification | densifies | densifies |
| **Rectangular matrices** | yes | yes | yes |
| **Drop-in compat** | `scipy` & `lap.lapjv` shims | baseline | baseline |
| **Type stubs** | Full `fastlap.pyi` | yes | no |
| **Dependencies** | numpy | numpy+scipy | numpy |

## Installation

```bash
# From source (requires Rust toolchain)
git clone https://github.com/MinLee0210/fastlap.git
cd fastlap
pip install maturin && maturin develop --release

# Or via pip
pip install fastlap
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.26.

## Quick Start

```python
import fastlap

cost_matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

total_cost, row_assign, col_assign = fastlap.solve_lap(cost_matrix, algorithm="lapjv")

print(total_cost)      # 15.0
print(row_assign)      # [0, 1, 2]
print(col_assign)      # [0, 1, 2]
```

`solve_lap` accepts plain Python lists, NumPy arrays, or SciPy CSR sparse matrices. Unassigned entries return `None`:

```python
import numpy as np

# Rectangular 2×3 matrix — one column is unassigned
cost, rows, cols = fastlap.solve_lap(
    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64), algorithm="lapjv"
)
print(cols)  # [0, 1, None] — column 2 unassigned
```

Pass `maximize=True` for maximum-weight matching instead of negating the matrix yourself:

```python
profit = np.array([[1, 9], [9, 1]], dtype=np.float64)
total, rows, cols = fastlap.solve_lap(profit, algorithm="lapjv", maximize=True)
print(total)  # 18.0 — pairs the high-value cells instead of the low-cost ones
```

### Cost Limit / Gating Threshold (Tracking & Data Association)

Reject assignments exceeding a maximum allowable cost (essential for Multi-Object Tracking like ByteTrack):

```python
# Threshold cost at 10.0 — any pair exceeding 10.0 is unassigned (None)
cost, rows, cols = fastlap.solve_lap(cost_matrix, cost_limit=10.0)
```

## Drop-in Compatibility Layers

### 1. Drop-in for `scipy.optimize.linear_sum_assignment`
```python
from fastlap.compat import linear_sum_assignment

# Returns (row_ind, col_ind) int64 ndarrays exactly like SciPy
row_ind, col_ind = linear_sum_assignment(cost_matrix)
```

### 2. Drop-in for `lap.lapjv` / `lapx.lapjv` (ByteTrack / YOLO MOT)
```python
import fastlap.lap as lap

# Matches lap.lapjv signature and return format (opt_cost, x, y)
opt_cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=0.5)
```

## Eleven Algorithms

| Algorithm | Approach | Time Complexity | Optimal? | Best for |
|-----------|----------|----------------|----------|----------|
| **LAPJV** | Column reduction + reduction transfer, then warm-started shortest-augmenting-path | O(n³) | Yes | General-purpose default |
| **Hungarian** | Classical Kuhn-Munkres: row/column reduction + zero-covering | O(n³) | Yes | Classical / academic use |
| **LAPMOD** | Shortest-augmenting-path directly on sparse adjacency — skips densification entirely for `scipy.sparse` CSR input | O(rows·nnz) sparse, O(n³) dense | Yes | Sparse cost matrices (candidate-gated tracking, large mostly-empty graphs) |
| **LAPJVsp** | Sparse JV: sparse column reduction + reduction transfer, warm-started sparse SAP — like LAPJV but never densifies CSR input | O(rows·nnz) sparse | Yes | True-sparse JV on CSR input (scipy `min_weight_full_bipartite_matching` territory) |
| **Dantzig** | Primal network simplex on the assignment LP, Dantzig's most-negative-reduced-cost pivoting rule | O(n³) typical | Yes | Simplex-based / LP-adjacent workflows |
| **Auction** | Bertsekas' auction algorithm — bidding/price-raising, ε-optimal | O(n²·k) | ε-optimal | Large square cost matrices |
| **Subgradient** | Coordinate-wise dual ascent warm start, then shortest-augmenting-path completion | O(n³) | Yes | Dual-based warm-up |
| **Sinkhorn** | Entropic regularized optimal transport (Sinkhorn-Knopp) dual scaling | O(n²) per iter | Yes (exact discrete) | Differentiable / OT-adjacent matching |
| **SSP** | Successive Shortest Path / Min-Cost Max-Flow with exact Johnson potentials | O(n³) | Yes | Graph theory / min-cost flow workflows |
| **Cost Scaling** | Goldberg-Kennedy push-relabel with cost scaling (ε-relaxation) | O(n³ log(nC)) | Yes | Network flow & cost-scaling research |
| **Greedy** | 1/2-approximation greedy edge selection | O(n² log n) | 1/2-approx | Ultra-fast approximate matching |

```python
>>> fastlap.get_supported_algorithms()
['lapjv', 'hungarian', 'lapmod', 'lapjvsp', 'subgradient', 'auction', 'dantzig', 'sinkhorn', 'ssp', 'cost_scaling', 'greedy']
```

## Ranked $K$-Best Assignments (Murty's Algorithm)

Find the top $K$ ranked alternative assignments in increasing order of cost:

```python
# Returns up to k solutions: [(cost_1, rows_1, cols_1), (cost_2, rows_2, cols_2), ...]
top_k_solutions = fastlap.solve_lap_kbest(cost_matrix, k=5)
```

Useful in Multi-Hypothesis Tracking (MHT), target tracking under ambiguous detections, and structural bioinformatics.

## Linear Bottleneck Assignment Problem (LBAP)

Find an assignment that **minimises the maximum cost edge** ($\min_\pi \max_i C_{i, \pi(i)}$):

```python
bottleneck_cost, rows, cols = fastlap.solve_lbap(cost_matrix)
```

Also available in parallel via `fastlap.solve_lbap_batch(matrices)`.

## Batch Solving (Parallel)

Solve hundreds of independent assignment problems across all CPU cores via Rayon.
A batch can be a plain list of matrices, or a single 3D `(B, N, M)` ndarray, and
the worker count is controllable:

```python
import numpy as np
import fastlap

matrices = np.random.rand(500, 50, 50)          # 500 × (50×50), stacked
results = fastlap.solve_lap_batch(matrices, algorithm="lapjv", n_threads=8)

# Each result is (cost, row_assign, col_assign)
costs = [r[0] for r in results]
```

## Optimal Duals (`solve_lap_duals`)

Beyond the primal assignment, `solve_lap_duals` returns the **optimal dual
potentials** `u` (rows) and `v` (columns): feasible (`u[i] + v[j] <= cost[i][j]`),
tight on every matched pair, with `total_cost == sum(u) + sum(v)`.

```python
cost, rows, cols, u, v = fastlap.solve_lap_duals(cost_matrix, algorithm="lapjv")
# u[i] / v[j] are the shadow prices of row/column resources
```

Supported for the exact dual-convergent algorithms (`lapjv`, `subgradient`,
`sinkhorn`, `dantzig`); maximization is not supported.

## Visualisation & Terminal Demos

```
uv run python examples/terminal_ui.py heatmap          # ANSI heatmap + assignment
uv run python examples/terminal_ui.py compare          # all algorithms head-to-head
uv run python examples/bipartite_assignment.py         # bipartite graph PNG (needs matplotlib+networkx)
uv run python examples/visualize_assignment.py         # matplotlib heatmap overlay
```

## Weighted Costs

Multiply each entry by a per-element weight during optimization:

```python
cost    = np.array([[1, 2], [3, 4]], dtype=np.float64)
weights = np.array([[1, 0.5], [0.5, 1]], dtype=np.float64)

total, rows, cols = fastlap.solve_lap_weighted(cost, weights, algorithm="lapjv")
```

The returned `total_cost` is computed from the **original** (unweighted) matrix.

## Use Cases

- **Object tracking** — frame-to-frame data association (ByteTrack, BoT-SORT, DeepSORT, SORT)
- **Multi-Hypothesis Tracking (MHT)** — ranked $K$-best associations via Murty's algorithm
- **Task scheduling & LBAP** — assign jobs to machines minimising total or bottleneck cost
- **Resource allocation** — match supply to demand in logistics
- **Feature matching** — point set registration and bipartite graph matching
- **Robotics** — multi-robot task allocation

## License

MIT — see [LICENSE](LICENSE).

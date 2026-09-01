<div align="center">

<img src="https://raw.githubusercontent.com/LakoreAI/fastlap/main/docs/static/fastlap.png" alt="fastlap — high-performance linear assignment problem solver in Python and Rust" width="400"/>

# fastlap

**Fast Linear Assignment Problem (LAP) Solver for Python — Powered by Rust**

[![PyPI version](https://img.shields.io/pypi/v/fastlap?color=blue&label=PyPI)](https://pypi.org/project/fastlap/)
[![Python](https://img.shields.io/pypi/pyversions/fastlap?label=Python)](https://pypi.org/project/fastlap/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/LakoreAI/fastlap/actions/workflows/ci.yml/badge.svg)](https://github.com/LakoreAI/fastlap/actions)

</div>

---

**fastlap** solves the [linear assignment problem](https://en.wikipedia.org/wiki/Assignment_problem) — minimum-cost bipartite matching, or its dual, **maximum weight matching** via `maximize=True` — at high speed from Python. It ships six algorithmically distinct solvers (LAPJV, Hungarian, LAPMOD, Dantzig, Auction, Subgradient) behind a single `solve_lap()` call, with optional **parallel batch solving** and **weighted cost** support.

If you work with **object tracking**, **task scheduling**, **resource allocation**, **matching algorithms**, or **combinatorial optimisation**, fastlap gives you an drop-in Rust accelerator for the core assignment step.

## Why fastlap?

| | fastlap (Rust) | scipy.optimize | lapjv (Python) |
|---|---|---|---|
| **Speed** | Sub-ms on 100×100 | ~ms | ~ms |
| **Algorithms** | 6 (algorithmically distinct) | 1 | 1 |
| **Batch parallel** | `solve_lap_batch` | manual | manual |
| **Weighted costs** | built-in | no | no |
| **Maximize mode** | `maximize=True` | manual negation | manual negation |
| **Sparse-aware solve** | LAPMOD skips densification | densifies | densifies |
| **Rectangular matrices** | yes | yes | yes |
| **Input validation** | NaN/Inf/empty guard | basic | none |
| **Dependencies** | numpy | numpy+scipy | numpy+cython |

## Installation

```bash
# From source (requires Rust toolchain)
git clone https://github.com/LakoreAI/fastlap.git
cd fastlap
pip install maturin && maturin develop

# Or via pip (once published)
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

`maximize` is also accepted by `solve_lap_batch` and `solve_lap_weighted`.

## Six Algorithms

Each algorithm is a genuinely different implementation, not a relabeled wrapper around a shared core — they use different data structures and pivoting/search strategies, and (correctness aside) have different practical performance characteristics as a result.

| Algorithm | Approach | Time Complexity | Optimal? | Best for |
|-----------|----------|----------------|----------|----------|
| **LAPJV** | Column reduction + reduction transfer, then warm-started shortest-augmenting-path | O(n³) | Yes | General-purpose default |
| **Hungarian** | Classical Kuhn-Munkres: row/column reduction + zero-covering | O(n³) | Yes | Classical / academic use |
| **LAPMOD** | Shortest-augmenting-path directly on sparse adjacency — skips densification entirely for `scipy.sparse` CSR input | O(rows·nnz) sparse, O(n³) dense | Yes | Sparse cost matrices (candidate-gated tracking, large mostly-empty graphs) |
| **Dantzig** | Primal network simplex on the assignment LP, Dantzig's most-negative-reduced-cost pivoting rule | O(n³) typical | Yes | Simplex-based / LP-adjacent workflows |
| **Auction** | Bertsekas' auction algorithm — bidding/price-raising, ε-optimal | O(n²·k) | ε-optimal | Large square cost matrices |
| **Subgradient** | Coordinate-wise dual ascent warm start, then shortest-augmenting-path completion | O(n³) | Yes | Dual-based warm-up |

```python
>>> fastlap.get_supported_algorithms()
['lapjv', 'hungarian', 'lapmod', 'subgradient', 'auction', 'dantzig']
```

Select with the `algorithm` parameter — all return the same format: `(cost, row_assign, col_assign)`. All six handle rectangular matrices.

`lapmod` is the only algorithm that actually reads a `scipy.sparse.csr_matrix` as sparse — the other five densify it first (fine for moderate sizes, wasteful for a large mostly-empty matrix):

```python
import scipy.sparse as sp
import fastlap

# Only some (row, col) pairs are valid candidates — everything else is
# implicitly forbidden (infinite cost), the same convention scipy uses.
sparse_cost = sp.random(5000, 5000, density=0.003, format="csr")
total, rows, cols = fastlap.solve_lap(sparse_cost, algorithm="lapmod")
```

## Batch Solving (Parallel)

Solve hundreds of independent assignment problems across all CPU cores:

```python
import numpy as np
import fastlap

matrices = [np.random.rand(50, 50) for _ in range(500)]
results = fastlap.solve_lap_batch(matrices, algorithm="lapjv")

# Each result is (cost, row_assign, col_assign)
costs = [r[0] for r in results]
```

Uses [Rayon](https://docs.rs/rayon) internally — linear speedup with core count.

Batch solving is sparse-aware too: with `algorithm="lapmod"`, scipy CSR matrices in the batch are solved directly on their sparse structure (never densified), exactly like a single `solve_lap(csr, "lapmod")` call:

```python
sparse_matrices = [sp.random(2000, 2000, density=0.002, format="csr") for _ in range(64)]
results = fastlap.solve_lap_batch(sparse_matrices, "lapmod")
```

## Weighted Costs

Multiply each entry by a per-element weight before solving (useful in tracking pipelines where confidence scores gate assignment costs):

```python
import numpy as np
import fastlap

cost    = np.array([[1, 2], [3, 4]], dtype=np.float64)
weights = np.array([[1, 0.5], [0.5, 1]], dtype=np.float64)

total, rows, cols = fastlap.solve_lap_weighted(cost, weights, algorithm="lapjv")
```

The returned `total_cost` is computed from the **original** (unweighted) matrix.

## Input Validation

fastlap rejects invalid inputs with clear error messages:

```python
import fastlap, numpy as np

# NaN
fastlap.solve_lap(np.array([[1, float("nan")], [3, 4]]), "lapjv")
# ValueError: Matrix contains NaN at position [0, 1]

# Inf
fastlap.solve_lap(np.array([[1, float("inf")], [3, 4]]), "lapjv")
# ValueError: Matrix contains infinite value at position [0, 1]

# Empty
fastlap.solve_lap(np.array([]), "lapjv")
# ValueError: Matrix must not be empty
```

## Use Cases

- **Object tracking** — frame-to-frame data association (Hungarian tracker, SORT, DeepSORT)
- **Task scheduling** — assign jobs to machines minimising total cost
- **Resource allocation** — match supply to demand in logistics
- **Graph matching** — bipartite matching in network analysis
- **Experimental design** — optimal matching in causal inference
- **Robotics** — multi-robot task allocation

## Performance Benchmarks

Run the built-in benchmark yourself:

```bash
uv run pytest tests/test_correctness.py -k benchmark -v
```

Or use the comparison script:

```bash
uv run python examples/examples.py
```

## Citation

If you use fastlap in research, please cite:

```bibtex
@software{fastlap2025,
  author       = {Le Duc Minh},
  title        = {fastlap: A High-Performance Python LAP Solver Powered by Rust},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/LakoreAI/fastlap},
  note         = {Python-Rust implementation of LAPJV, Hungarian, LAPMOD, Dantzig, Auction, and Subgradient algorithms}
}
```

## FAQ

<details>
<summary><strong>What is the linear assignment problem?</strong></summary>

The linear assignment problem (LAP) is a combinatorial optimisation problem: given an n×n cost matrix, find a one-to-one mapping (permutation) between rows and columns that minimises the total cost. It is polynomially solvable (unlike the travelling salesman problem) and appears in many applied contexts.

</details>

<details>
<summary><strong>How do I choose an algorithm?</strong></summary>

Use **LAPJV** (the default) unless you have a specific reason not to. If your cost matrix comes from `scipy.sparse` and is large and mostly empty (e.g. a gated tracking association matrix), use **LAPMOD** — it's the only algorithm that solves directly on the sparse structure instead of densifying first. For large *dense* square matrices where an ε-optimal solution is acceptable, try **Auction**. All six algorithms handle rectangular matrices.

</details>

<details>
<summary><strong>Does fastlap support GPU acceleration?</strong></summary>

Not yet. All computation runs on the CPU via Rust. GPU support is on the roadmap.

</details>

<details>
<summary><strong>How does fastlap compare to scipy.optimize.linear_sum_assignment?</strong></summary>

fastlap is typically 2–10× faster for matrices up to 1000×1000, offers six algorithms (SciPy only implements one), supports parallel batch solving, and provides input validation. SciPy is a better choice if you already depend on it and performance is not critical.

</details>

<details>
<summary><strong>Can I use fastlap with PyTorch/TensorFlow tensors?</strong></summary>

Convert to NumPy first: `fastlap.solve_lap(tensor.numpy(), algorithm="lapjv")`.

</details>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and how to add a new algorithm.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Open an issue at [github.com/LakoreAI/fastlap/issues](https://github.com/LakoreAI/fastlap/issues).

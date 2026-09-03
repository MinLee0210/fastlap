# Getting Started

## Requirements

- Python **≥ 3.9**
- NumPy **≥ 1.26**
- A Rust toolchain — only if you're building from source

## Installation

=== "From PyPI"

    ```bash
    pip install fastlap
    ```

=== "From source"

    ```bash
    git clone https://github.com/MinLee0210/fastlap.git
    cd fastlap
    pip install maturin
    maturin develop --release
    ```

=== "With uv (development)"

    ```bash
    git clone https://github.com/MinLee0210/fastlap.git
    cd fastlap
    uv sync                # creates a venv + installs dev deps
    uv run maturin develop --release
    ```

## Your first solve

`solve_lap` is the single entry point for the standard assignment problem. Give it a cost matrix and it returns the optimal assignment:

```python
import fastlap

cost_matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

total_cost, row_assign, col_assign = fastlap.solve_lap(cost_matrix, algorithm="lapjv")

print(total_cost)  # 15.0
print(row_assign)  # [0, 1, 2]
print(col_assign)  # [0, 1, 2]
```

`fastlap` accepts plain Python lists, NumPy arrays, or SciPy CSR sparse matrices — no conversion boilerplate needed.

## Core concepts

### The return shape

Every solving function returns a **`LapSolution`**: `(total_cost, row_assign, col_assign)`.

- `row_assign[i]` is the column assigned to row `i`, or `None` if row `i` is unassigned.
- `col_assign[j]` is the row assigned to column `j`, or `None` if column `j` is unassigned.
- `total_cost` is always computed from the *original, untransformed* matrix — even when `maximize=True` or `cost_limit` gates out edges internally.

### Rectangular matrices

Non-square matrices are handled natively. If a matrix has more columns than rows (or vice versa), the extra rows/columns simply come back unassigned (`None`):

```python
import numpy as np

# 2×3 matrix — one column has nothing to match against
cost, rows, cols = fastlap.solve_lap(
    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64), algorithm="lapjv"
)
print(cols)  # [0, 1, None] — column 2 unassigned
```

### Minimize vs. maximize

By default, `solve_lap` finds the **minimum-cost** assignment. Pass `maximize=True` for maximum-weight matching instead of negating the matrix yourself:

```python
profit = np.array([[1, 9], [9, 1]], dtype=np.float64)
total, rows, cols = fastlap.solve_lap(profit, algorithm="lapjv", maximize=True)
print(total)  # 18.0 — pairs the high-value cells instead of the low-cost ones
```

### Choosing an algorithm

All eleven algorithms are exact (or near-exact) solvers reachable through the same `algorithm=` keyword. `"lapjv"` is a solid general-purpose default. See the [Algorithms](algorithms/index.md) page for a full breakdown of complexity, optimality guarantees, and when to reach for each one.

```python
>>> fastlap.get_supported_algorithms()
['lapjv', 'hungarian', 'lapmod', 'lapjvsp', 'subgradient', 'auction', 'dantzig', 'sinkhorn', 'ssp', 'cost_scaling', 'greedy']
```

## Where to next

<div class="grid cards" markdown>

-   **[Algorithms](algorithms/index.md)**

    Complexity, optimality, and best-fit use case for all eleven solvers.

-   **[Features](features/index.md)**

    Cost limits, batch solving (incl. 3D batches), weighted costs, K-best, LBAP, optimal duals, sparse input, and compatibility shims.

-   **[API Reference](api-reference.md)**

    Full function signatures and parameter documentation.

</div>

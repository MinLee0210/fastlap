# K-Best Assignments (Murty's Algorithm)

`solve_lap_kbest` finds the top-K ranked alternative assignments, in increasing order of cost (or decreasing order of profit under `maximize=True`), using [Murty's algorithm](https://en.wikipedia.org/wiki/Murty%27s_algorithm).

```python
# Returns up to k solutions: [(cost_1, rows_1, cols_1), (cost_2, rows_2, cols_2), ...]
top_k_solutions = fastlap.solve_lap_kbest(cost_matrix, k=5)
```

## Why you'd want more than one answer

The single optimal assignment is only the *most likely* interpretation of a cost matrix — in ambiguous scenarios, the second- or third-best assignment might be the correct one once more evidence arrives. This is exactly the situation:

- **Multi-Hypothesis Tracking (MHT)** — keep several candidate associations alive per frame instead of committing to one, and prune later once ambiguity resolves.
- **Target tracking under ambiguous detections** — near-identical objects (e.g. two players close together) produce near-tied costs; the top-K assignments capture that ambiguity explicitly.
- **Structural bioinformatics** — ranked alternative matchings between predicted and reference structures.

## How it works

Murty's algorithm partitions the assignment search space into disjoint subproblems by fixing and forbidding specific `(row, col)` edges, solving each subproblem with the crate's shortest-augmenting-path solver, and popping candidates off a min-heap ordered by cost. Each popped candidate is a genuinely optimal solution to its constrained subproblem, so the sequence popped from the heap is guaranteed to be the true global ranking — not just K arbitrary near-optimal assignments.

## Example

```python
import numpy as np
import fastlap

matrix = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
])

# A 3×3 matrix has 3! = 6 possible assignments — ask for the top 4.
solutions = fastlap.solve_lap_kbest(matrix, k=4)

for cost, rows, cols in solutions:
    print(cost, rows)
```

Solutions come back sorted: `costs == sorted(costs)`.

## Parameters

```python
fastlap.solve_lap_kbest(
    cost_matrix,
    k=3,
    maximize=False,
    cost_limit=None,
)
```

`cost_limit` is applied independently to each of the K returned solutions (see [Cost Limit](cost-limit.md)) — a given solution's gated-out pairs don't affect any other solution's ranking.

# Optimal Duals (`solve_lap_duals`)

Every assignment solver you've seen on this site maintains **dual variables** — row potentials `u` and column potentials `v` — even when it doesn't return them. `solve_lap_duals` hands them to you.

```python
import numpy as np
import fastlap

m = np.array([
    [4.0, 1.0, 3.0],
    [2.0, 0.0, 5.0],
    [3.0, 2.0, 2.0],
])

cost, rows, cols, u, v = fastlap.solve_lap_duals(m, algorithm="lapjv")
print(cost, rows)  # 5.0 [1, 0, 2]
print(u)           # row duals, one per row
print(v)           # column duals, one per column
```

## What the duals mean

For the minimum-cost assignment LP, `u[i]` and `v[j]` are the optimal multipliers on the "row `i` used exactly once" and "column `j` used exactly once" constraints. Concretely:

- **`u[i] + v[j] ≤ cost[i][j]`** for every entry — *dual feasibility* (no pair is over-charged relative to its cost).
- **`u[i] + v[j] == cost[i][j]`** on every matched pair — *complementary slackness* (the matching is tight exactly where it's used).
- **`sum(u) + sum(v) == total_cost`** — *strong duality*: the dual objective equals the primal optimum.

Economically these are the **shadow prices** of the row and column resources: `v[j]` is roughly the marginal value of column `j`, `u[i]` of row `i`. Raise a column's supply or cost and the duals tell you how the optimum would shift.

## Supported algorithms

Duals are returned for the exact, dual-convergent algorithms:

```python
>>> fastlap.solve_lap_duals(m, algorithm="sinkhorn")   # ok
>>> fastlap.solve_lap_duals(m, algorithm="dantzig")    # ok
>>> fastlap.solve_lap_duals(m, algorithm="greedy")     # ValueError
ValueError: Algorithm 'greedy' does not support dual extraction. Supported: lapjv, subgradient, sinkhorn, dantzig
```

The approximate solvers ([Auction](../algorithms/auction.md), [Greedy](../algorithms/greedy.md)) have no exact dual to report, and the integer/flow formulations ([LAPMOD](../algorithms/lapmod.md), [LAPJVsp](../algorithms/lapjvsp.md), [SSP](../algorithms/ssp.md), [Cost Scaling](../algorithms/cost-scaling.md)) don't expose potentials in a form that maps back to the input rows/columns cleanly.

!!! note "Maximization is not supported"
    `solve_lap_duals` solves the minimum-cost form only — pass `maximize=False` semantics. If you need the duals of a maximum-weight problem, negate the matrix yourself and interpret the result with care.

## Rectangular matrices

`u` has one entry per **row** and `v` one per **column** of the original matrix — padded rows/columns never leak into the output:

```python
cost, rows, cols, u, v = fastlap.solve_lap_duals(np.ones((3, 5)))
print(len(u), len(v))  # 3 5
```

## Duals as a warm start

Because duals summarize the "shape" of a solved problem, they're also the natural currency for [warm-starting](../algorithms/concepts.md#dual-variables-and-reduced-cost) related problems — e.g. tracking the same assignment matrix across frames where only a few costs changed. fastlap uses this idea internally ([Subgradient](../algorithms/subgradient.md), [LAPJV](../algorithms/lapjv.md)); you can exploit it too by reusing `u`/`v` from a previous frame as an initialization heuristic for a re-solve.

## See also

- [Key Concepts — dual variables & complementary slackness](../algorithms/concepts.md#dual-variables-and-reduced-cost)
- The [LAPJV](../algorithms/lapjv.md), [Subgradient](../algorithms/subgradient.md), [Sinkhorn](../algorithms/sinkhorn.md) and [Dantzig](../algorithms/dantzig.md) algorithm pages for how each computes potentials.

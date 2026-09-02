---
description: A high-performance Linear Assignment Problem solver for Python, written in Rust.
---

# fastlap

<p class="hero-tagline" markdown>
**Fast Linear Assignment Problem (LAP) Solver for Python — Powered by Rust**
</p>

<div class="hero-badges" markdown>
[![PyPI version](https://img.shields.io/pypi/v/fastlap?color=blue&label=PyPI)](https://pypi.org/project/fastlap/)
[![Python](https://img.shields.io/pypi/pyversions/fastlap?label=Python)](https://pypi.org/project/fastlap/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/MinLee0210/fastlap/blob/main/LICENSE)
[![CI](https://github.com/MinLee0210/fastlap/actions/workflows/ci.yml/badge.svg)](https://github.com/MinLee0210/fastlap/actions)
</div>

**fastlap** solves the [linear assignment problem](https://en.wikipedia.org/wiki/Assignment_problem) — minimum-cost bipartite matching, maximum-weight matching (`maximize=True`), bottleneck assignment (`solve_lbap`), and ranked K-best assignments (`solve_lap_kbest`) — at high speed from Python. It ships **ten algorithmically distinct solvers** behind a single `solve_lap()` call, with **parallel batch solving**, **gating threshold support** (`cost_limit`), **weighted costs**, and **drop-in compatibility layers** for SciPy and `lap`/`lapx`.

If you work with **object tracking** (ByteTrack, BoT-SORT, DeepSORT), **task scheduling**, **resource allocation**, **feature matching**, or **combinatorial optimisation**, fastlap gives you a drop-in Rust accelerator for the core assignment step.

[Get started :material-arrow-right:](getting-started.md){ .md-button .md-button--primary }
[View on GitHub :fontawesome-brands-github:](https://github.com/MinLee0210/fastlap){ .md-button }

## Why fastlap?

| | fastlap (Rust) | scipy.optimize | lap / lapx (C++) |
|---|---|---|---|
| **Speed** | Sub-ms on 100×100 | ~ms | ~ms |
| **Algorithms** | 10 (algorithmically distinct) + LBAP + K-Best | 1 | 1 |
| **Gating threshold** | `cost_limit=...` built-in | manual filtering | `cost_limit` |
| **Bottleneck (LBAP)** | `solve_lbap` built-in | no | no |
| **K-Best (Murty)** | `solve_lap_kbest` built-in | no | no |
| **Batch parallel** | `solve_lap_batch` (Rayon) | manual | manual |
| **Weighted costs** | built-in | no | no |
| **Maximize mode** | `maximize=True` | manual negation | manual negation |
| **Sparse-aware solve** | LAPMOD skips densification | densifies | densifies |
| **Rectangular matrices** | yes | yes | yes |
| **Drop-in compat** | `scipy` & `lap.lapjv` shims | baseline | baseline |
| **Type stubs** | full `fastlap.pyi` | yes | no |
| **Dependencies** | numpy | numpy + scipy | numpy |

## At a glance

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **Ten algorithms, one API**

    ---

    LAPJV, Hungarian, LAPMOD, Dantzig, Auction, Subgradient, Sinkhorn, SSP, Cost Scaling, and Greedy — all behind `solve_lap(algorithm=...)`.

    [:octicons-arrow-right-24: Browse algorithms](algorithms/index.md)

-   :material-target:{ .lg .middle } **Tracking-ready gating**

    ---

    `cost_limit` rejects assignments above (or below, in `maximize` mode) a threshold — essential for multi-object tracking data association.

    [:octicons-arrow-right-24: Cost limit](features/cost-limit.md)

-   :material-source-branch:{ .lg .middle } **Ranked K-best**

    ---

    Murty's algorithm returns the top-K alternative assignments in increasing-cost order — built for multi-hypothesis tracking.

    [:octicons-arrow-right-24: K-best assignments](features/kbest.md)

-   :material-swap-horizontal:{ .lg .middle } **Drop-in replacements**

    ---

    Swap in fastlap for `scipy.optimize.linear_sum_assignment` or `lap.lapjv` with a one-line import change.

    [:octicons-arrow-right-24: Compatibility layers](features/compat.md)

</div>

## Quick taste

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

Continue to [Getting Started](getting-started.md) for installation and a full walkthrough, or jump straight to the [API Reference](api-reference.md).

## Use cases

- **Object tracking** — frame-to-frame data association (ByteTrack, BoT-SORT, DeepSORT, SORT)
- **Multi-Hypothesis Tracking (MHT)** — ranked K-best associations via Murty's algorithm
- **Task scheduling & LBAP** — assign jobs to machines minimising total or bottleneck cost
- **Resource allocation** — match supply to demand in logistics
- **Feature matching** — point-set registration and bipartite graph matching
- **Robotics** — multi-robot task allocation

# Features

Beyond the basic min-cost assignment, fastlap ships a set of features aimed squarely at real production workloads — tracking pipelines, scheduling systems, and large batch jobs.

<div class="grid cards" markdown>

-   :material-target:{ .lg .middle } **[Cost Limit (Gating)](cost-limit.md)**

    ---

    Reject assignments that exceed (or fall below) a threshold cost — the gating step every multi-object tracker needs.

-   :material-view-grid-plus:{ .lg .middle } **[Batch Solving](batch.md)**

    ---

    Solve hundreds of independent matrices in parallel across all CPU cores via Rayon.

-   :material-scale-balance:{ .lg .middle } **[Weighted Costs](weighted.md)**

    ---

    Multiply each cost entry by a per-element weight before solving, while total cost is still reported unweighted.

-   :material-chart-timeline-variant:{ .lg .middle } **[Optimal Duals](duals.md)**

    ---

    Solve and return the row/column dual potentials `(u, v)` — feasible, tight on the matching, with `sum(u) + sum(v)` equal to the optimum.

-   :material-source-branch:{ .lg .middle } **[K-Best (Murty)](kbest.md)**

    ---

    Rank the top-K alternative assignments in increasing cost order — for multi-hypothesis tracking.

-   :material-arrow-collapse-down:{ .lg .middle } **[Bottleneck (LBAP)](bottleneck.md)**

    ---

    Minimise the *maximum* edge cost in the assignment, not the sum — the bottleneck assignment problem.

-   :material-grid-off:{ .lg .middle } **[Sparse Matrices](sparse.md)**

    ---

    Feed a `scipy.sparse.csr_matrix` straight into LAPMOD or LAPJVsp without ever densifying it.

-   :material-swap-horizontal:{ .lg .middle } **[Compatibility Layers](compat.md)**

    ---

    Drop-in replacements for `scipy.optimize.linear_sum_assignment` and `lap.lapjv` / `lapx.lapjv` — plus lapx-style `lapjvx` and `assignment_pairs` helpers.

-   :material-monitor-eye:{ .lg .middle } **[Visualisation & Demos](viz.md)**

    ---

    Terminal heatmaps, an algorithm head-to-head, a bipartite-graph render, and a matplotlib overlay — runnable examples under `examples/`.

</div>

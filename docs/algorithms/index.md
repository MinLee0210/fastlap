# Algorithms

fastlap ships **ten algorithmically distinct** solvers behind one dispatch table (`src/utils.rs::solve_with`), selected via the `algorithm=` keyword on [`solve_lap`](../api-reference.md#solve_lap) and friends.

```python
>>> fastlap.get_supported_algorithms()
['lapjv', 'hungarian', 'lapmod', 'subgradient', 'auction', 'dantzig', 'sinkhorn', 'ssp', 'cost_scaling', 'greedy']
```

Each algorithm has its own deep-dive page: motivation, how it works, pseudocode, time/space complexity, a worked example (with a diagram), common pitfalls, and Python usage. The pages below are ordered **easy → advanced** — roughly, how much background theory each one assumes — not by runtime importance; `"lapjv"` stays the recommended default regardless of where it falls in this reading order.

!!! info "New to duals, reduced costs, or augmenting paths?"
    Read [Key Concepts](concepts.md) first. It's a short shared glossary — bipartite matching, dual variables, complementary slackness, augmenting paths, approximation ratios, ε-optimality, and the min-cost-flow framing — that every page below assumes.

## Comparison table

| Algorithm | Approach | Time complexity | Space complexity | Optimal? | Best for |
|-----------|----------|------------------|-------------------|----------|----------|
| [Greedy](greedy.md) | 1/2-approximation greedy edge selection | O(n² log n) | O(n²) | 1/2-approx | Ultra-fast approximate matching |
| [Hungarian](hungarian.md) | Classical Kuhn–Munkres: row/column reduction + zero-covering | O(n³) | O(n²) | Yes | Classical / academic use |
| [LAPJV](lapjv.md) | Column reduction + reduction transfer, then warm-started shortest-augmenting-path | O(n³) worst case | O(n²) | Yes | General-purpose default |
| [Subgradient](subgradient.md) | Coordinate-wise dual ascent warm start, then shortest-augmenting-path completion | O(n³) | O(n²) | Yes | Dual-based warm-up |
| [Sinkhorn](sinkhorn.md) | Entropic regularized optimal transport (Sinkhorn–Knopp) dual scaling | O(n²) per iter, O(n³) worst-case total | O(n²) | Yes (exact discrete recovery) | Differentiable / OT-adjacent matching |
| [Auction](auction.md) | Bertsekas' auction algorithm — bidding/price-raising with ε-scaling | O(n²·k) | O(n²) | ε-optimal | Large square cost matrices |
| [SSP](ssp.md) | Successive Shortest Path / Min-Cost Max-Flow with exact Johnson potentials | O(n³ log n) | O(n²) | Yes | Graph theory / min-cost flow workflows |
| [Cost Scaling](cost-scaling.md) | Goldberg–Kennedy push-relabel with cost scaling (ε-relaxation) | O(n³ log(nC)) | O(n²) | Yes | Network flow & cost-scaling research |
| [Dantzig](dantzig.md) | Primal network simplex on the assignment LP, Dantzig's most-negative-reduced-cost pivoting rule | O(n³) typical, O(n⁴) worst-case bound | O(n²) | Yes | Simplex-based / LP-adjacent workflows |
| [LAPMOD](lapmod.md) | Shortest-augmenting-path directly on sparse adjacency — skips densification for `scipy.sparse` CSR input | O(rows·nnz) sparse, O(n³) dense | O(n + nnz) sparse, O(n²) dense | Yes | Sparse cost matrices (candidate-gated tracking, large mostly-empty graphs) |

!!! tip "Not sure which to pick?"
    Start with `"lapjv"`. It's the default, it's exact, and its warm-start preprocessing makes it the fastest exact solver in the suite on most real cost matrices. Reach for a different algorithm only when you have a specific reason — sparse input ([LAPMOD](lapmod.md)), an approximate answer under a tight time budget ([Greedy](greedy.md)), or you're studying a particular algorithm family ([Dantzig](dantzig.md), [SSP](ssp.md), [Cost Scaling](cost-scaling.md), [Sinkhorn](sinkhorn.md)).

## The ten algorithms

<div class="grid cards" markdown>

-   :material-numeric-1-box:{ .lg .middle } **[Greedy](greedy.md)**

    ---

    A 1/2-approximation baseline: sort every edge, claim the cheapest still-available pair. No duality, no graph search — the simplest thing that could work.

-   :material-numeric-2-box:{ .lg .middle } **[Hungarian](hungarian.md)**

    ---

    Kuhn, 1955 / Munkres, 1957. The textbook zero-covering algorithm — star/prime marks and augmenting paths.

-   :material-numeric-3-box:{ .lg .middle } **[LAPJV](lapjv.md)**

    ---

    Jonker & Volgenant, 1987. Column reduction resolves most rows for free before a warm-started shortest-augmenting-path finishes the rest.

-   :material-numeric-4-box:{ .lg .middle } **[Subgradient](subgradient.md)**

    ---

    Held & Karp, 1971. Coordinate-wise dual ascent builds a warm start for the same shortest-augmenting-path solver LAPJV uses.

-   :material-numeric-5-box:{ .lg .middle } **[Sinkhorn](sinkhorn.md)**

    ---

    Cuturi, 2013 (entropic OT); Sinkhorn & Knopp, 1967 (scaling). Matrix-scaling duals, then exact discrete recovery.

-   :material-numeric-6-box:{ .lg .middle } **[Auction](auction.md)**

    ---

    Bertsekas, 1988. A different paradigm entirely — bidders raise item prices in an economic auction, with ε-scaling for exactness.

-   :material-numeric-7-box:{ .lg .middle } **[SSP](ssp.md)**

    ---

    Successive Shortest Path — min-cost max-flow with Dijkstra and Johnson potentials.

-   :material-numeric-8-box:{ .lg .middle } **[Cost Scaling](cost-scaling.md)**

    ---

    Goldberg & Kennedy, 1995. Push-relabel with ε-relaxation, scaled down across phases.

-   :material-numeric-9-box:{ .lg .middle } **[Dantzig](dantzig.md)**

    ---

    Dantzig, 1963. Primal network simplex — an explicit spanning-tree basis with stepping-stone pivots.

-   :material-numeric-10-box:{ .lg .middle } **[LAPMOD](lapmod.md)**

    ---

    A sparse-adjacency extension of the shortest-augmenting-path idea already covered above — the only algorithm here with a true sparse fast path.

</div>

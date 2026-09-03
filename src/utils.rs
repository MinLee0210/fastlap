use crate::lap::{
    auction, cost_scaling, dantzig, greedy, hungarian, lapjv, lapjvsp, lapmod, sinkhorn, ssp,
    subgradient,
};
use crate::types::{LapSolution, SparseCost};

pub fn supported_algorithms() -> &'static [&'static str] {
    &[
        "lapjv",
        "hungarian",
        "lapmod",
        "lapjvsp",
        "subgradient",
        "auction",
        "dantzig",
        "sinkhorn",
        "ssp",
        "cost_scaling",
        "greedy",
    ]
}

/// Dispatch to the named algorithm. Single source of truth for all entry points.
pub fn solve_with(matrix: Vec<Vec<f64>>, algorithm: &str) -> Result<LapSolution, String> {
    match algorithm {
        "lapjv" => Ok(lapjv::solve(matrix)),
        "hungarian" => Ok(hungarian::solve(matrix)),
        "lapmod" => Ok(lapmod::solve(matrix)),
        "lapjvsp" => Ok(lapjvsp::solve(matrix)),
        "subgradient" => Ok(subgradient::solve(matrix)),
        "auction" => auction::solve(matrix),
        "dantzig" => Ok(dantzig::solve(matrix)),
        "sinkhorn" => Ok(sinkhorn::solve(matrix)),
        "ssp" => Ok(ssp::solve(matrix)),
        "cost_scaling" => Ok(cost_scaling::solve(matrix)),
        "greedy" => Ok(greedy::solve(matrix)),
        _ => Err(format!(
            "Unknown algorithm '{}'. Supported: {}",
            algorithm,
            supported_algorithms().join(", ")
        )),
    }
}

/// Algorithms for which [`solve_lap_duals`](crate::solve_lap_duals) can return
/// exact optimal dual variables. These all converge to exact optimal duals
/// (`u[i] + v[j] <= cost[i][j]`, tight on every matched pair); the other
/// algorithms are either approximate (auction, greedy), integer-specialized,
/// or do not retain dual potentials in a way that maps cleanly back to the
/// input rows/columns (hungarian, lapmod, ssp, cost_scaling).
pub fn dual_supported_algorithms() -> &'static [&'static str] {
    &["lapjv", "subgradient", "sinkhorn", "dantzig"]
}

/// Apply cost threshold gating (`cost_limit`).
/// In minimization mode, any assignment where matrix[i][j] > limit is unassigned (None).
/// In maximize mode, any assignment where matrix[i][j] < limit is unassigned (None).
pub fn apply_cost_limit_dense(
    matrix: &[Vec<f64>],
    mut row_assign: Vec<Option<usize>>,
    mut col_assign: Vec<Option<usize>>,
    cost_limit: Option<f64>,
    maximize: bool,
) -> LapSolution {
    if let Some(limit) = cost_limit {
        let nrows = matrix.len();
        let ncols = if nrows > 0 { matrix[0].len() } else { 0 };
        for i in 0..nrows {
            if let Some(j) = row_assign[i] {
                if j < ncols {
                    let cost = matrix[i][j];
                    let reject = if maximize { cost < limit } else { cost > limit };
                    if reject {
                        row_assign[i] = None;
                        col_assign[j] = None;
                    }
                }
            }
        }
    }
    let total_cost = recompute_cost(matrix, &row_assign);
    (total_cost, row_assign, col_assign)
}

/// Apply cost threshold gating to a sparse solution.
pub fn apply_cost_limit_sparse(
    sc: &SparseCost,
    mut row_assign: Vec<Option<usize>>,
    mut col_assign: Vec<Option<usize>>,
    cost_limit: Option<f64>,
    maximize: bool,
) -> LapSolution {
    if let Some(limit) = cost_limit {
        for i in 0..sc.nrows {
            if let Some(j) = row_assign[i] {
                if let Some(&(_, cost)) = sc.rows[i].iter().find(|&&(jj, _)| jj == j) {
                    let reject = if maximize { cost < limit } else { cost > limit };
                    if reject {
                        row_assign[i] = None;
                        if j < col_assign.len() {
                            col_assign[j] = None;
                        }
                    }
                }
            }
        }
    }
    let total_cost = sc.cost_of(&row_assign);
    (total_cost, row_assign, col_assign)
}

/// O(n³) shortest-augmenting-path (SAP) solver for a square n×n cost matrix.
///
/// This is the standard competitive-programming formulation of the
/// Hungarian / Jonker-Volgenant algorithm. It maintains dual variables u[i]
/// and v[j] to track reduced costs and finds augmenting paths in O(n²) per row,
/// giving O(n³) overall.
///
/// Requires: `cost` is an n×n slice with finite entries.
/// Returns: `(total_cost, row_assign, col_assign)` with all vectors of length n.
pub fn sap_solve(cost: &[Vec<f64>]) -> LapSolution {
    let n = cost.len();
    sap_solve_warm(cost, &vec![0.0; n], &vec![0.0; n])
}

/// [`sap_solve`] plus the optimal row duals `u` and column duals `v`
/// (0-indexed, length n) found along the way. The returned triple is the
/// standard `(cost, row_assign, col_assign)`; the duals satisfy
/// `u[i] + v[j] <= cost[i][j]` for every `i, j` (dual feasibility) with
/// equality on every matched pair (complementary slackness).
pub fn sap_solve_duals(cost: &[Vec<f64>]) -> (LapSolution, Vec<f64>, Vec<f64>) {
    let n = cost.len();
    sap_solve_partial_duals(cost, &vec![0.0; n], &vec![0.0; n], &vec![None; n])
}

/// Shortest-augmenting-path solve, warm-started from a feasible dual pair
/// `(u0, v0)` (0-indexed, one entry per row/column) satisfying
/// `u0[i] + v0[j] <= cost[i][j]` for all `i, j`.
///
/// Starting from feasible duals close to optimal shrinks the per-row
/// augmenting-path search, since `delta` (the shortest-path distance) is
/// small when `u0, v0` are already near-optimal. Passing all-zero duals
/// (via [`sap_solve`]) is always safe since the zero vector is trivially
/// feasible whenever `cost` is non-negative; callers with signed costs must
/// ensure their warm-start duals are actually feasible or fall back to zero.
pub fn sap_solve_warm(cost: &[Vec<f64>], u0: &[f64], v0: &[f64]) -> LapSolution {
    let n = cost.len();
    sap_solve_partial(cost, u0, v0, &vec![None; n])
}

/// Shortest-augmenting-path solve, warm-started from a feasible dual pair
/// `(u0, v0)` *and* a pre-existing partial matching `row_assign0` (0-indexed;
/// `row_assign0[i] = Some(j)` means row `i` is already matched to column `j`).
///
/// The caller must guarantee two invariants or the result may not be optimal:
/// 1. `row_assign0` is a valid partial permutation (no column repeated).
/// 2. Every pre-matched pair satisfies complementary slackness exactly —
///    `u0[i] + v0[row_assign0[i]] == cost[i][row_assign0[i]]` — so the partial
///    matching is itself optimal for the rows it covers under `(u0, v0)`.
///
/// Only the rows left as `None` are resolved via augmenting-path search;
/// pre-matched rows are taken as-is. This lets a cheap O(n²) preprocessing
/// pass (e.g. LAPJV's column reduction) resolve most rows for free, leaving
/// only the remainder to pay the full O(n²)-per-row search cost.
pub fn sap_solve_partial(
    cost: &[Vec<f64>],
    u0: &[f64],
    v0: &[f64],
    row_assign0: &[Option<usize>],
) -> LapSolution {
    sap_solve_partial_duals(cost, u0, v0, row_assign0).0
}

/// [`sap_solve_partial`] plus the optimal duals reached by the search. The
/// duals are feasible (`u[i] + v[j] <= cost[i][j]`) with equality on every
/// pair the search actually resolved via an augmenting path, so for a cold
/// solve (empty `row_assign0`) they are an exact optimal dual pair.
pub fn sap_solve_partial_duals(
    cost: &[Vec<f64>],
    u0: &[f64],
    v0: &[f64],
    row_assign0: &[Option<usize>],
) -> (LapSolution, Vec<f64>, Vec<f64>) {
    let n = cost.len();
    if n == 0 {
        return ((0.0, vec![], vec![]), vec![], vec![]);
    }

    // 1-indexed storage; p[j] = row matched to column j (0 = free column).
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    u[1..=n].copy_from_slice(u0);
    v[1..=n].copy_from_slice(v0);
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for (i, opt_j) in row_assign0.iter().enumerate() {
        if let Some(j) = opt_j {
            p[j + 1] = i + 1;
        }
    }
    let free_rows: Vec<usize> = (0..n).filter(|&i| row_assign0[i].is_none()).collect();

    // Per-row buffers hoisted out of the loop: `minv` and `used` are reset
    // (O(n)) for each free row anyway, but reusing the same allocations avoids
    // paying a fresh `vec!`/malloc for every row — the difference is visible
    // for large n and for batch solves that run many rows back to back.
    let mut minv = vec![f64::INFINITY; n + 1];
    let mut used = vec![false; n + 1];

    for i0 in free_rows {
        let i = i0 + 1;
        p[0] = i;
        let mut j0 = 0usize;
        minv.iter_mut().for_each(|m| *m = f64::INFINITY);
        used.iter_mut().for_each(|u| *u = false);

        // Find the shortest augmenting path for row i.
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;

            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            // Shift duals by the shortest-path distance delta.
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break; // Reached a free column; augmenting path is complete.
            }
        }

        // Flip the augmenting path.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // Convert 1-indexed p[] into 0-indexed row_assign / col_assign.
    let mut row_assign = vec![None; n];
    let mut col_assign = vec![None; n];
    for j in 1..=n {
        if p[j] != 0 {
            row_assign[p[j] - 1] = Some(j - 1);
            col_assign[j - 1] = Some(p[j] - 1);
        }
    }

    let total_cost: f64 = (0..n)
        .filter_map(|i| row_assign[i].map(|j| cost[i][j]))
        .sum();

    let u_duals = u[1..=n].to_vec();
    let v_duals = v[1..=n].to_vec();
    ((total_cost, row_assign, col_assign), u_duals, v_duals)
}

/// Coordinate-wise dual ascent used to build a *feasible* warm-start pair of
/// duals for any real-valued cost matrix (see `subgradient::solve` for the
/// proof that the update never leaves the feasible region):
/// `u[i] = min_j (cost[i][j] - v[j])`, then `v[j] = min_i (cost[i][j] - u[i])`.
pub fn dual_ascent(cost: &[Vec<f64>], rounds: usize) -> (Vec<f64>, Vec<f64>) {
    let n = cost.len();
    let mut u = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];
    for _ in 0..rounds {
        for i in 0..n {
            u[i] = (0..n)
                .map(|j| cost[i][j] - v[j])
                .fold(f64::INFINITY, f64::min);
        }
        for j in 0..n {
            v[j] = (0..n)
                .map(|i| cost[i][j] - u[i])
                .fold(f64::INFINITY, f64::min);
        }
    }
    (u, v)
}

/// Solve a square matrix via [`sap_solve`] (padding first if needed) and also
/// return the optimal dual vectors. Assignments into padded rows/columns are
/// trimmed back to the original `(nrows, ncols)` dimensions and the duals are
/// truncated to match, so the caller only ever sees real rows/columns.
///
/// Works for any finite real costs (signed included): a short feasible
/// dual-ascent warm start is used instead of the all-zero duals, which are
/// only feasible for non-negative matrices.
pub fn sap_solve_duals_matrix(matrix: &[Vec<f64>]) -> (LapSolution, Vec<f64>, Vec<f64>) {
    let nrows = matrix.len();
    if nrows == 0 {
        return ((0.0, vec![], vec![]), vec![], vec![]);
    }
    let ncols = matrix[0].len();
    let fill = matrix
        .iter()
        .flatten()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        + 1.0;
    let padded = pad_to_square(matrix, fill);
    let n = padded.len();

    let (u0, v0) = dual_ascent(&padded, 8);
    let ((_, row_assign, col_assign), mut u, mut v) =
        sap_solve_partial_duals(&padded, &u0, &v0, &vec![None; n]);

    if nrows != ncols {
        let trimmed = trim_solution(matrix, row_assign, col_assign);
        u.truncate(nrows);
        v.truncate(ncols);
        return (trimmed, u, v);
    }

    let total_cost: f64 = (0..n)
        .filter_map(|i| row_assign[i].map(|j| padded[i][j]))
        .sum();
    ((total_cost, row_assign, col_assign), u, v)
}

/// Negate every entry of a matrix, turning a maximum-weight problem into an
/// equivalent minimum-cost one (`argmax` of `x` is `argmin` of `-x`).
pub fn negate_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|&v| -v).collect())
        .collect()
}

/// Recompute total cost from the original matrix given a row assignment.
/// Used after solving on a transformed matrix (negated for `maximize`, or
/// weighted) so the reported cost always reflects real, untransformed entries.
pub fn recompute_cost(matrix: &[Vec<f64>], row_assign: &[Option<usize>]) -> f64 {
    row_assign
        .iter()
        .enumerate()
        .filter_map(|(i, opt_j)| opt_j.map(|j| matrix[i][j]))
        .sum()
}

/// Pad a (possibly non-square) cost matrix to dim×dim, filling added entries with `fill`.
pub fn pad_to_square(matrix: &[Vec<f64>], fill: f64) -> Vec<Vec<f64>> {
    let nrows = matrix.len();
    let ncols = if nrows > 0 { matrix[0].len() } else { 0 };
    let dim = nrows.max(ncols);
    if nrows == ncols {
        return matrix.to_vec();
    }
    let mut padded = vec![vec![fill; dim]; dim];
    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            padded[i][j] = val;
        }
    }
    padded
}

/// Trim a SAP solution back to the original (nrows × ncols) dimensions.
///
/// Assignments that went to padded rows/columns are mapped to None.
/// The returned cost is recomputed from the original matrix.
pub fn trim_solution(
    matrix: &[Vec<f64>],
    row_assign: Vec<Option<usize>>,
    col_assign: Vec<Option<usize>>,
) -> LapSolution {
    let nrows = matrix.len();
    let ncols = if nrows > 0 { matrix[0].len() } else { 0 };

    let trimmed_row: Vec<Option<usize>> = (0..nrows)
        .map(|i| row_assign[i].filter(|&j| j < ncols))
        .collect();

    let trimmed_col: Vec<Option<usize>> = (0..ncols)
        .map(|j| col_assign[j].filter(|&i| i < nrows))
        .collect();

    let total_cost: f64 = (0..nrows)
        .filter_map(|i| trimmed_row[i].map(|j| matrix[i][j]))
        .sum();

    (total_cost, trimmed_row, trimmed_col)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / ((1u64 << 31) as f64)
    }

    fn random_matrix(
        nrows: usize,
        ncols: usize,
        lo: f64,
        hi: f64,
        seed: &mut u64,
    ) -> Vec<Vec<f64>> {
        (0..nrows)
            .map(|_| (0..ncols).map(|_| lo + (hi - lo) * lcg(seed)).collect())
            .collect()
    }

    #[test]
    fn duals_feasible_and_tight_on_square() {
        let mut seed = 42u64;
        for n in [2usize, 5, 12] {
            for _ in 0..50 {
                let m = random_matrix(n, n, 0.0, 100.0, &mut seed);
                let ((cost, row_assign, col_assign), u, v) = sap_solve_duals_matrix(&m);
                for i in 0..n {
                    for j in 0..n {
                        assert!(
                            u[i] + v[j] <= m[i][j] + 1e-9,
                            "dual infeasible at ({i},{j})"
                        );
                    }
                }
                for i in 0..n {
                    let j = row_assign[i].unwrap();
                    assert!(
                        (u[i] + v[j] - m[i][j]).abs() < 1e-7,
                        "not tight on matched pair ({i},{j})"
                    );
                    assert_eq!(col_assign[j], Some(i));
                }
                // Strong duality: primal == dual objective.
                let dual_obj: f64 = u.iter().sum::<f64>() + v.iter().sum::<f64>();
                assert!(
                    (cost - dual_obj).abs() < 1e-7,
                    "duality gap {cost} vs {dual_obj}"
                );
            }
        }
    }

    #[test]
    fn duals_feasible_with_negative_costs() {
        let mut seed = 7u64;
        let m = random_matrix(6, 6, -50.0, 50.0, &mut seed);
        let ((cost, row_assign, _), u, v) = sap_solve_duals_matrix(&m);
        for i in 0..6 {
            for j in 0..6 {
                assert!(u[i] + v[j] <= m[i][j] + 1e-9);
            }
            let j = row_assign[i].unwrap();
            assert!((u[i] + v[j] - m[i][j]).abs() < 1e-7);
        }
        let dual_obj: f64 = u.iter().sum::<f64>() + v.iter().sum::<f64>();
        assert!((cost - dual_obj).abs() < 1e-7);
    }

    #[test]
    fn duals_trimmed_on_rectangular() {
        let mut seed = 99u64;
        for (nrows, ncols) in [(5usize, 3usize), (3, 8)] {
            let m = random_matrix(nrows, ncols, 1.0, 100.0, &mut seed);
            let ((_, row_assign, col_assign), u, v) = sap_solve_duals_matrix(&m);
            assert_eq!(u.len(), nrows);
            assert_eq!(v.len(), ncols);
            for i in 0..nrows {
                if let Some(j) = row_assign[i] {
                    assert!(u[i] + v[j] <= m[i][j] + 1e-9);
                    assert!((u[i] + v[j] - m[i][j]).abs() < 1e-7);
                }
            }
            for j in 0..ncols {
                if let Some(i) = col_assign[j] {
                    assert_eq!(row_assign[i], Some(j));
                }
            }
        }
    }
}

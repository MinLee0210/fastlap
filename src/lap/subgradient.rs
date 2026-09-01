use crate::types::LapSolution;
use crate::utils::{pad_to_square, sap_solve_warm, trim_solution};

/// Number of coordinate dual-ascent sweeps used to build the warm-start.
const DUAL_ASCENT_ROUNDS: usize = 8;

/// Solves the LAP using dual (subgradient-style) ascent to warm-start the
/// O(n³) shortest-augmenting-path primal recovery.
///
/// Phase 1 performs coordinate-wise dual ascent: alternately setting
/// `u[i] = min_j (cost[i][j] - v[j])` and `v[j] = min_i (cost[i][j] - u[i])`.
/// Each step is an exact maximization along one coordinate of the assignment
/// LP's dual objective, so — unlike a naive subgradient step — it never
/// needs a projection: `u, v` are dual-feasible (`u[i] + v[j] <= cost[i][j]`
/// for all `i, j`) after every round, by construction, for any real-valued
/// cost matrix (no non-negativity requirement).
///
/// Phase 2 runs the shortest-augmenting-path solver *warm-started* from
/// these near-optimal duals (`sap_solve_warm`), rather than discarding them —
/// starting close to the optimal dual shrinks the per-row augmenting-path
/// search, making this measurably faster than a cold Hungarian solve while
/// remaining exactly optimal (SAP always converges to the true optimum
/// regardless of the feasible starting point).
pub fn solve(matrix: Vec<Vec<f64>>) -> LapSolution {
    let nrows = matrix.len();
    if nrows == 0 {
        return (0.0, vec![], vec![]);
    }
    let ncols = matrix[0].len();
    let fill = matrix
        .iter()
        .flatten()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        + 1.0;
    let padded = pad_to_square(&matrix, fill);
    let n = padded.len();

    // Phase 1: coordinate-wise dual ascent — builds a feasible, near-optimal
    // warm start for the duals instead of running the SAP solver cold.
    let mut u = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];

    for _ in 0..DUAL_ASCENT_ROUNDS {
        for i in 0..n {
            u[i] = (0..n)
                .map(|j| padded[i][j] - v[j])
                .fold(f64::INFINITY, f64::min);
        }
        for j in 0..n {
            v[j] = (0..n)
                .map(|i| padded[i][j] - u[i])
                .fold(f64::INFINITY, f64::min);
        }
    }

    // Phase 2: SAP primal recovery, warm-started from the ascended duals —
    // guarantees the globally optimal feasible solution.
    let (_, row_assign, col_assign) = sap_solve_warm(&padded, &u, &v);
    if nrows == ncols {
        let total_cost: f64 = (0..n)
            .filter_map(|i| row_assign[i].map(|j| padded[i][j]))
            .sum();
        (total_cost, row_assign, col_assign)
    } else {
        trim_solution(&matrix, row_assign, col_assign)
    }
}

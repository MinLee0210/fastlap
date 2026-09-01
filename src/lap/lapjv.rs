use crate::types::LapSolution;
use crate::utils::{pad_to_square, sap_solve_partial, trim_solution};

/// Solves the LAP using the Jonker-Volgenant algorithm: cheap O(n²) column
/// reduction and reduction-transfer preprocessing resolve as many rows as
/// possible for free, before the remainder fall back to a warm-started
/// shortest-augmenting-path search.
///
/// This is what makes LAPJV faster in practice than a cold Hungarian solve
/// at the same O(n³) worst case: on many real cost matrices, column
/// reduction alone assigns most rows, leaving only a handful to pay the
/// full per-row search cost.
///
/// Non-square matrices are padded with a cost slightly above the maximum real cost so that
/// padded assignments are never preferred over real ones.
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

    // Phase 1: column reduction. For each column, find its cheapest row and
    // tentatively claim that row for it; a row keeps only its cheapest
    // claimant (by column dual value) if claimed by more than one column.
    let mut v = vec![0.0f64; n];
    let mut row_assign: Vec<Option<usize>> = vec![None; n];
    let mut col_assign: Vec<Option<usize>> = vec![None; n];
    let mut claims = vec![0usize; n];

    for j in 0..n {
        let mut best_i = 0usize;
        let mut best_val = padded[0][j];
        for i in 1..n {
            if padded[i][j] < best_val {
                best_val = padded[i][j];
                best_i = i;
            }
        }
        v[j] = best_val;
        claims[best_i] += 1;

        match row_assign[best_i] {
            None => {
                row_assign[best_i] = Some(j);
                col_assign[j] = Some(best_i);
            }
            Some(j1) if best_val < v[j1] => {
                row_assign[best_i] = Some(j);
                col_assign[j] = Some(best_i);
                col_assign[j1] = None;
            }
            _ => col_assign[j] = None,
        }
    }

    // Phase 2: reduction transfer. Rows claimed by exactly one column can
    // have their assigned column's dual tightened using the row's
    // second-best reduced cost, shrinking the search space for phase 3
    // without touching the (still-optimal) partial matching.
    for i in 0..n {
        if claims[i] != 1 {
            continue;
        }
        let j1 = row_assign[i].unwrap();
        let mut min_other = f64::INFINITY;
        for j in 0..n {
            if j != j1 {
                let reduced = padded[i][j] - v[j];
                if reduced < min_other {
                    min_other = reduced;
                }
            }
        }
        if min_other.is_finite() {
            v[j1] -= min_other;
        }
    }

    // Feasible row duals consistent with v: u[i] = min_j(cost[i][j] - v[j]).
    // For rows resolved in phase 1 this exactly equals cost[i][j]-v[j] at
    // their assigned column (complementary slackness holds by construction
    // of column reduction), so the partial matching is optimal under (u, v)
    // and phase 3 only needs to extend it to the remaining free rows.
    let u: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| padded[i][j] - v[j])
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    // Phase 3: complete the assignment for any row column reduction left
    // unresolved via warm-started shortest-augmenting-path search.
    let (_, row_assign, col_assign) = sap_solve_partial(&padded, &u, &v, &row_assign);

    if nrows == ncols {
        let total_cost: f64 = (0..n)
            .filter_map(|i| row_assign[i].map(|j| padded[i][j]))
            .sum();
        (total_cost, row_assign, col_assign)
    } else {
        trim_solution(&matrix, row_assign, col_assign)
    }
}

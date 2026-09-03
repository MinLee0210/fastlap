use crate::types::LapSolution;
use crate::utils::{dual_ascent, pad_to_square, sap_solve_partial, trim_solution};
use std::collections::VecDeque;

/// Solves the LAP using the Cost-Scaling Push-Relabel algorithm (Goldberg & Kennedy).
///
/// Implements epsilon-relaxation on the bipartite assignment graph with cost scaling.
/// In each phase, the algorithm maintains an epsilon-optimal pseudoflow and performs
/// pushes along admissible edges (reduced cost <= 0) and dual potential relabels until
/// all rows are matched. Epsilon is scaled down until epsilon < 1 / (n + 1).
///
/// The push-relabel sweep is bounded by a per-phase step budget and a phase budget.
/// If either is exhausted, the row potentials are no longer guaranteed ε-optimal and
/// warm-starting the final SAP polish from them could silently produce a suboptimal
/// cost. In that case the polish is instead run **cold** (from a fresh feasible dual
/// ascent), so the returned assignment is always exactly optimal — never silently wrong.
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

    let max_cost = padded
        .iter()
        .flatten()
        .cloned()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));

    let mut u = vec![0.0f64; n];
    let v = vec![0.0f64; n];

    // Initial dual values
    for i in 0..n {
        u[i] = padded[i].iter().cloned().fold(f64::INFINITY, f64::min);
    }

    let mut row_match: Vec<Option<usize>> = vec![None; n];
    let mut col_match: Vec<Option<usize>> = vec![None; n];

    let alpha = 4.0f64;
    let mut epsilon = (max_cost * 0.5).max(1.0);
    let target_eps = (1.0 / ((n as f64) + 1.0)).min(1e-4);

    let max_phases = 30;
    let mut phase = 0;
    let mut step_capped = false;

    while epsilon >= target_eps && phase < max_phases {
        phase += 1;
        // Unmatch any edges violating complementary slackness with current epsilon
        for r in 0..n {
            if let Some(c) = row_match[r] {
                let red_cost = padded[r][c] - u[r] - v[c];
                if red_cost > epsilon {
                    row_match[r] = None;
                    col_match[c] = None;
                }
            }
        }

        let mut active_rows: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if row_match[i].is_none() {
                active_rows.push_back(i);
            }
        }

        let max_steps = 50 * n * n + 2000;
        let mut step = 0;

        while let Some(r) = active_rows.pop_front() {
            step += 1;
            if step > max_steps {
                step_capped = true;
                break;
            }

            // Find column minimizing reduced cost: (padded[r][c] - u[r] - v[c])
            let mut best_c = None;
            let mut min_red_cost = f64::INFINITY;
            let mut second_min = f64::INFINITY;

            for c in 0..n {
                let red_cost = padded[r][c] - u[r] - v[c];
                if red_cost < min_red_cost {
                    second_min = min_red_cost;
                    min_red_cost = red_cost;
                    best_c = Some(c);
                } else if red_cost < second_min {
                    second_min = red_cost;
                }
            }

            if let Some(c) = best_c {
                if min_red_cost <= 0.0 {
                    // Push flow into column c
                    if let Some(prev_r) = col_match[c] {
                        row_match[prev_r] = None;
                        active_rows.push_back(prev_r);
                    }
                    row_match[r] = Some(c);
                    col_match[c] = Some(r);
                } else {
                    // Relabel row potential u[r] upwards so min reduced cost becomes <= 0
                    let raise = if second_min.is_finite() && second_min > min_red_cost {
                        second_min + epsilon
                    } else {
                        min_red_cost + epsilon
                    };
                    u[r] += raise;
                    active_rows.push_back(r);
                }
            }
        }

        epsilon /= alpha;
    }

    // Whether the bounded ε-sweep finished within budget. Leftover unmatched
    // rows are normal (the polish below resolves them); what matters is whether
    // the row potentials can still be trusted as near-optimal.
    let budget_exhausted = (phase >= max_phases && epsilon >= target_eps) || step_capped;

    // Complete / polish solution with SAP. If the sweep hit its budget the
    // potentials are untrustworthy, so fall back to a cold exact completion
    // rather than risk a subtly wrong warm-started answer.
    let (_, final_row, final_col) = if budget_exhausted {
        let (u0, v0) = dual_ascent(&padded, 8);
        sap_solve_partial(&padded, &u0, &v0, &vec![None; n])
    } else {
        sap_solve_partial(&padded, &u, &v, &row_match)
    };

    if nrows == ncols {
        let total_cost: f64 = (0..n)
            .filter_map(|i| final_row[i].map(|j| padded[i][j]))
            .sum();
        (total_cost, final_row, final_col)
    } else {
        trim_solution(&matrix, final_row, final_col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_scaling_square() {
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 1.0],
            vec![3.0, 1.0, 2.0],
        ];
        let (cost, rows, cols) = solve(matrix);
        assert!((cost - 3.0).abs() < 1e-6);
        assert_eq!(rows, vec![Some(0), Some(2), Some(1)]);
        assert_eq!(cols, vec![Some(0), Some(2), Some(1)]);
    }
}

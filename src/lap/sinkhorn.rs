use crate::types::LapSolution;
use crate::utils::{pad_to_square, sap_solve_warm, trim_solution};

/// Number of Sinkhorn iterations
const SINKHORN_ITERS: usize = 100;

/// Solves the LAP using Sinkhorn's algorithm (Entropic Regularized Optimal Transport)
/// to compute smooth dual potentials, followed by warm-started Shortest Augmenting Path
/// primal recovery.
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

    let (max_c, min_c) = padded
        .iter()
        .flatten()
        .fold((f64::NEG_INFINITY, f64::INFINITY), |(hi, lo), &v| {
            (hi.max(v), lo.min(v))
        });
    let range = (max_c - min_c).abs().max(1.0);
    let eps = (range * 0.05).max(1e-4);

    // Gibbs kernel: K[i][j] = exp(-C[i][j] / eps)
    // Stabilize by subtracting row minima
    let mut k_mat = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let r_min = padded[i].iter().cloned().fold(f64::INFINITY, f64::min);
        for j in 0..n {
            let val = -((padded[i][j] - r_min) / eps);
            k_mat[i][j] = val.max(-700.0).exp();
        }
    }

    let mut u_vec = vec![1.0f64 / (n as f64); n];
    let mut v_vec = vec![1.0f64 / (n as f64); n];

    for _ in 0..SINKHORN_ITERS {
        // u = 1 / (K * v)
        for i in 0..n {
            let mut kv = 0.0;
            for j in 0..n {
                kv += k_mat[i][j] * v_vec[j];
            }
            u_vec[i] = if kv > 1e-300 { 1.0 / kv } else { 1.0 };
        }

        // v = 1 / (K^T * u)
        for j in 0..n {
            let mut kt_u = 0.0;
            for i in 0..n {
                kt_u += k_mat[i][j] * u_vec[i];
            }
            v_vec[j] = if kt_u > 1e-300 { 1.0 / kt_u } else { 1.0 };
        }
    }

    // Convert scaling vectors to dual potentials: u_dual[i] = eps * log(u_vec[i])
    let mut u_dual = vec![0.0f64; n];
    let mut v_dual = vec![0.0f64; n];
    for i in 0..n {
        u_dual[i] = if u_vec[i] > 0.0 {
            eps * u_vec[i].ln()
        } else {
            0.0
        };
    }
    for j in 0..n {
        v_dual[j] = if v_vec[j] > 0.0 {
            eps * v_vec[j].ln()
        } else {
            0.0
        };
    }

    // Ensure feasibility: u_dual[i] + v_dual[j] <= padded[i][j]
    for i in 0..n {
        let max_slack = (0..n)
            .map(|j| padded[i][j] - v_dual[j])
            .fold(f64::INFINITY, f64::min);
        if u_dual[i] > max_slack {
            u_dual[i] = max_slack;
        }
    }

    let (_, row_assign, col_assign) = sap_solve_warm(&padded, &u_dual, &v_dual);

    if nrows == ncols {
        let total_cost: f64 = (0..n)
            .filter_map(|i| row_assign[i].map(|j| padded[i][j]))
            .sum();
        (total_cost, row_assign, col_assign)
    } else {
        trim_solution(&matrix, row_assign, col_assign)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinkhorn_square() {
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

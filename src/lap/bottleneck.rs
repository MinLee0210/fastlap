use crate::types::LapSolution;
use crate::utils::{pad_to_square, trim_solution};
use std::collections::VecDeque;

type AssignmentPair = (Vec<Option<usize>>, Vec<Option<usize>>);

/// Solves the Linear Bottleneck Assignment Problem (LBAP):
/// Find a matching that minimizes the maximum cost assigned:
/// min_pi max_i C_{i, pi(i)}
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

    // Collect and sort all unique finite edge values
    let mut costs: Vec<f64> = padded
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    costs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    costs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    if costs.is_empty() {
        return (0.0, vec![None; nrows], vec![None; ncols]);
    }

    // Binary search for minimum threshold T where a perfect matching of size n exists
    let mut low = 0;
    let mut high = costs.len() - 1;
    let mut best_idx = high;
    let mut best_matching: Option<AssignmentPair> = None;

    while low <= high {
        let mid = (low + high) / 2;
        let threshold = costs[mid];

        let (size, r_match, c_match) = hopcroft_karp(&padded, n, threshold);
        if size == n {
            best_idx = mid;
            best_matching = Some((r_match, c_match));
            if mid == 0 {
                break;
            }
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    let (r_assign, c_assign) = match best_matching {
        Some(m) => m,
        None => {
            let (_, r, c) = hopcroft_karp(&padded, n, costs[best_idx]);
            (r, c)
        }
    };

    if nrows == ncols {
        let bottleneck_cost = (0..n)
            .filter_map(|i| r_assign[i].map(|j| padded[i][j]))
            .fold(f64::NEG_INFINITY, f64::max);
        let total_cost = if bottleneck_cost.is_finite() {
            bottleneck_cost
        } else {
            0.0
        };
        (total_cost, r_assign, c_assign)
    } else {
        let (_, trimmed_r, trimmed_c) = trim_solution(&matrix, r_assign, c_assign);
        let bottleneck_cost = (0..nrows)
            .filter_map(|i| trimmed_r[i].map(|j| matrix[i][j]))
            .fold(f64::NEG_INFINITY, f64::max);
        let total_cost = if bottleneck_cost.is_finite() {
            bottleneck_cost
        } else {
            0.0
        };
        (total_cost, trimmed_r, trimmed_c)
    }
}

/// Hopcroft-Karp algorithm for maximum cardinality bipartite matching
/// on edges where cost <= threshold.
fn hopcroft_karp(
    cost: &[Vec<f64>],
    n: usize,
    threshold: f64,
) -> (usize, Vec<Option<usize>>, Vec<Option<usize>>) {
    // 1-indexed for BFS/DFS; 0 is the dummy/unmatched node
    let mut pair_u = vec![0usize; n + 1];
    let mut pair_v = vec![0usize; n + 1];
    let mut dist = vec![usize::MAX; n + 1];

    let mut matching_size = 0;

    while bfs(cost, n, threshold, &pair_u, &pair_v, &mut dist) {
        for u in 1..=n {
            if pair_u[u] == 0 && dfs(cost, threshold, u, &mut pair_u, &mut pair_v, &dist) {
                matching_size += 1;
            }
        }
    }

    let mut row_assign = vec![None; n];
    let mut col_assign = vec![None; n];

    for u in 1..=n {
        if pair_u[u] != 0 {
            row_assign[u - 1] = Some(pair_u[u] - 1);
        }
    }
    for v in 1..=n {
        if pair_v[v] != 0 {
            col_assign[v - 1] = Some(pair_v[v] - 1);
        }
    }

    (matching_size, row_assign, col_assign)
}

fn bfs(
    cost: &[Vec<f64>],
    n: usize,
    threshold: f64,
    pair_u: &[usize],
    pair_v: &[usize],
    dist: &mut [usize],
) -> bool {
    let mut queue = VecDeque::new();

    for u in 1..=n {
        if pair_u[u] == 0 {
            dist[u] = 0;
            queue.push_back(u);
        } else {
            dist[u] = usize::MAX;
        }
    }
    dist[0] = usize::MAX;

    while let Some(u) = queue.pop_front() {
        if dist[u] < dist[0] {
            for v in 1..=n {
                if cost[u - 1][v - 1] <= threshold + 1e-12 {
                    let next_u = pair_v[v];
                    if dist[next_u] == usize::MAX {
                        dist[next_u] = dist[u] + 1;
                        queue.push_back(next_u);
                    }
                }
            }
        }
    }

    dist[0] != usize::MAX
}

fn dfs(
    cost: &[Vec<f64>],
    threshold: f64,
    u: usize,
    pair_u: &mut [usize],
    pair_v: &mut [usize],
    dist: &[usize],
) -> bool {
    if u == 0 {
        return true;
    }

    let n = cost.len();
    for v in 1..=n {
        if cost[u - 1][v - 1] <= threshold + 1e-12 {
            let next_u = pair_v[v];
            if dist[next_u] == dist[u] + 1 && dfs(cost, threshold, next_u, pair_u, pair_v, dist) {
                pair_v[v] = u;
                pair_u[u] = v;
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bottleneck_simple() {
        let matrix = vec![
            vec![1.0, 2.0, 10.0],
            vec![2.0, 1.0, 10.0],
            vec![10.0, 10.0, 5.0],
        ];
        let (cost, rows, cols) = solve(matrix);
        assert!(cost <= 5.0);
        assert_eq!(rows.len(), 3);
        assert_eq!(cols.len(), 3);
    }

    #[test]
    fn test_bottleneck_rectangular() {
        let matrix = vec![vec![1.0, 9.0, 3.0, 8.0], vec![7.0, 2.0, 6.0, 4.0]];
        let (cost, rows, cols) = solve(matrix);
        assert_eq!(cost, 2.0);
        assert_eq!(rows, vec![Some(0), Some(1)]);
        assert_eq!(cols, vec![Some(0), Some(1), None, None]);
    }
}

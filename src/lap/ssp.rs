use crate::types::LapSolution;
use crate::utils::{pad_to_square, trim_solution};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(PartialEq)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Solves the LAP using the Successive Shortest Path (SSP) algorithm for
/// Minimum Cost Maximum Flow on the bipartite assignment network with exact
/// Johnson node potentials.
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

    // Shift costs to non-negative so initial potential pi = 0 is dual feasible
    let min_cost = padded
        .iter()
        .flatten()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let shift = if min_cost < 0.0 { -min_cost } else { 0.0 };

    let mut cost_mat = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            cost_mat[i][j] = padded[i][j] + shift;
        }
    }

    // Node indexing:
    // 0: Source S
    // 1..=n: Row nodes (u = 1..=n)
    // n+1..=2n: Col nodes (v = n+1..=2n)
    // 2n+1: Sink T
    let total_nodes = 2 * n + 2;
    let s = 0;
    let t = 2 * n + 1;

    let mut matched_col: Vec<Option<usize>> = vec![None; n];
    let mut matched_row: Vec<Option<usize>> = vec![None; n];

    let mut pi = vec![0.0f64; total_nodes];
    let mut dist = vec![f64::INFINITY; total_nodes];
    let mut parent = vec![usize::MAX; total_nodes];
    let mut heap = BinaryHeap::new();

    for _ in 0..n {
        dist.fill(f64::INFINITY);
        parent.fill(usize::MAX);
        heap.clear();

        dist[s] = 0.0;
        heap.push(Reverse((OrdF64(0.0), s)));

        while let Some(Reverse((OrdF64(d), u))) = heap.pop() {
            if d > dist[u] + 1e-12 {
                continue;
            }
            if u == t {
                break;
            }

            if u == s {
                // Outgoing edges to unmatched rows with cost 0
                for i in 0..n {
                    if matched_col[i].is_none() {
                        let row_node = 1 + i;
                        let red_cost = 0.0 + pi[s] - pi[row_node];
                        let new_d = d + red_cost;
                        if new_d + 1e-12 < dist[row_node] {
                            dist[row_node] = new_d;
                            parent[row_node] = s;
                            heap.push(Reverse((OrdF64(new_d), row_node)));
                        }
                    }
                }
            } else if (1..=n).contains(&u) {
                let r = u - 1;
                // Forward edges to columns
                for c in 0..n {
                    if matched_col[r] != Some(c) {
                        let col_node = n + 1 + c;
                        let red_cost = cost_mat[r][c] + pi[u] - pi[col_node];
                        let new_d = d + red_cost;
                        if new_d + 1e-12 < dist[col_node] {
                            dist[col_node] = new_d;
                            parent[col_node] = u;
                            heap.push(Reverse((OrdF64(new_d), col_node)));
                        }
                    }
                }
            } else if (n + 1..=2 * n).contains(&u) {
                let c = u - (n + 1);
                // Backward edge to matched row
                if let Some(r) = matched_row[c] {
                    let row_node = 1 + r;
                    let red_cost = -cost_mat[r][c] + pi[u] - pi[row_node];
                    let new_d = d + red_cost;
                    if new_d + 1e-12 < dist[row_node] {
                        dist[row_node] = new_d;
                        parent[row_node] = u;
                        heap.push(Reverse((OrdF64(new_d), row_node)));
                    }
                } else {
                    // Edge from unmatched column to sink T with cost 0
                    let red_cost = 0.0 + pi[u] - pi[t];
                    let new_d = d + red_cost;
                    if new_d + 1e-12 < dist[t] {
                        dist[t] = new_d;
                        parent[t] = u;
                        heap.push(Reverse((OrdF64(new_d), t)));
                    }
                }
            }
        }

        let d_t = dist[t];
        if !d_t.is_finite() {
            break;
        }

        // Potential update: pi'(u) = pi(u) + min(dist(u), dist(t))
        for node in 0..total_nodes {
            let delta = if dist[node] < d_t { dist[node] } else { d_t };
            pi[node] += delta;
        }

        // Augment path from t to s
        let mut curr = t;
        while curr != s {
            let p = parent[curr];
            if (1..=n).contains(&p) && (n + 1..=2 * n).contains(&curr) {
                let r = p - 1;
                let c = curr - (n + 1);
                matched_col[r] = Some(c);
                matched_row[c] = Some(r);
            }
            curr = p;
        }
    }

    if nrows == ncols {
        let total_cost: f64 = (0..n)
            .filter_map(|i| matched_col[i].map(|j| padded[i][j]))
            .sum();
        (total_cost, matched_col, matched_row)
    } else {
        trim_solution(&matrix, matched_col, matched_row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssp_square() {
        let matrix = vec![
            vec![10.0, 5.0, 13.0],
            vec![3.0, 7.0, 18.0],
            vec![9.0, 11.0, 4.0],
        ];
        let (cost, rows, cols) = solve(matrix);
        assert!((cost - 12.0).abs() < 1e-6);
        assert_eq!(rows, vec![Some(1), Some(0), Some(2)]);
        assert_eq!(cols, vec![Some(1), Some(0), Some(2)]);
    }
}

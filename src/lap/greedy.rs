use crate::types::LapSolution;

/// Solves the assignment problem using a fast Greedy 1/2-approximation:
/// Sorts all available matrix entries by cost and greedily assigns the cheapest
/// available (row, column) pairs.
pub fn solve(matrix: Vec<Vec<f64>>) -> LapSolution {
    let nrows = matrix.len();
    if nrows == 0 {
        return (0.0, vec![], vec![]);
    }
    let ncols = matrix[0].len();

    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(nrows * ncols);
    for (i, row) in matrix.iter().enumerate() {
        for (j, &c) in row.iter().enumerate() {
            if c.is_finite() {
                edges.push((i, j, c));
            }
        }
    }

    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut row_assign = vec![None; nrows];
    let mut col_assign = vec![None; ncols];
    let mut row_used = vec![false; nrows];
    let mut col_used = vec![false; ncols];

    for (r, c, _) in edges {
        if !row_used[r] && !col_used[c] {
            row_used[r] = true;
            col_used[c] = true;
            row_assign[r] = Some(c);
            col_assign[c] = Some(r);
        }
    }

    let total_cost: f64 = (0..nrows)
        .filter_map(|i| row_assign[i].map(|j| matrix[i][j]))
        .sum();

    (total_cost, row_assign, col_assign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_simple() {
        let matrix = vec![
            vec![1.0, 5.0, 9.0],
            vec![8.0, 2.0, 6.0],
            vec![4.0, 7.0, 3.0],
        ];
        let (cost, rows, cols) = solve(matrix);
        assert!((cost - 6.0).abs() < 1e-6);
        assert_eq!(rows, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(cols, vec![Some(0), Some(1), Some(2)]);
    }
}

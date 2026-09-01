use crate::types::LapSolution;
use crate::utils::{pad_to_square, trim_solution};

#[derive(Clone, Copy, PartialEq)]
enum Mark {
    None,
    Star,
    Prime,
}

/// Solves the LAP using the classical Kuhn-Munkres (Hungarian) algorithm:
/// row/column reduction followed by iterative zero-covering with starred
/// and primed zeros, tracing augmenting paths through the star/prime
/// marking rather than through dual-variable shortest paths.
///
/// This is a distinct implementation from the dual-based shortest-augmenting-path
/// solver used elsewhere in this crate (`utils::sap_solve`) — different data
/// structures (star/prime marks, row/column cover flags) and a different
/// control-flow shape — even though both are O(n³) and always agree on the result.
/// Non-square matrices are padded with a cost above the maximum real entry.
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

    let scale = padded
        .iter()
        .flatten()
        .cloned()
        .fold(1.0f64, |acc, v| acc.max(v.abs()));
    let eps = scale * 1e-9;

    let mut cost = padded.clone();

    // Step 1: subtract each row's minimum, then each column's minimum.
    for row in cost.iter_mut() {
        let min = row.iter().cloned().fold(f64::INFINITY, f64::min);
        for v in row.iter_mut() {
            *v -= min;
        }
    }
    for j in 0..n {
        let min = (0..n).map(|i| cost[i][j]).fold(f64::INFINITY, f64::min);
        for i in 0..n {
            cost[i][j] -= min;
        }
    }

    let mut mask = vec![vec![Mark::None; n]; n];
    let mut row_cover = vec![false; n];
    let mut col_cover = vec![false; n];

    // Step 2: star an independent zero in each row/column where possible.
    for i in 0..n {
        for j in 0..n {
            if cost[i][j].abs() < eps && !row_cover[i] && !col_cover[j] {
                mask[i][j] = Mark::Star;
                row_cover[i] = true;
                col_cover[j] = true;
            }
        }
    }
    row_cover.iter_mut().for_each(|c| *c = false);
    col_cover.iter_mut().for_each(|c| *c = false);

    let mut path_row0 = 0usize;
    let mut path_col0 = 0usize;
    let mut step = 3u8;

    loop {
        match step {
            // Cover every column containing a starred zero. All columns
            // covered means the starred zeros are a complete assignment.
            3 => {
                for j in 0..n {
                    if (0..n).any(|i| mask[i][j] == Mark::Star) {
                        col_cover[j] = true;
                    }
                }
                step = if col_cover.iter().filter(|&&c| c).count() >= n {
                    7
                } else {
                    4
                };
            }
            // Prime an uncovered zero. If its row has a starred zero,
            // recover that column and keep scanning; otherwise start an
            // augmenting path from here.
            4 => {
                let mut zero = None;
                'scan: for i in 0..n {
                    if row_cover[i] {
                        continue;
                    }
                    for j in 0..n {
                        if !col_cover[j] && cost[i][j].abs() < eps {
                            zero = Some((i, j));
                            break 'scan;
                        }
                    }
                }
                match zero {
                    None => step = 6,
                    Some((i, j)) => {
                        mask[i][j] = Mark::Prime;
                        match (0..n).find(|&jj| mask[i][jj] == Mark::Star) {
                            Some(j_star) => {
                                row_cover[i] = true;
                                col_cover[j_star] = false;
                            }
                            None => {
                                path_row0 = i;
                                path_col0 = j;
                                step = 5;
                            }
                        }
                    }
                }
            }
            // Augment along the alternating star/prime path starting at
            // the uncovered primed zero found in step 4.
            5 => {
                let mut path = vec![(path_row0, path_col0)];
                loop {
                    let (_, c) = *path.last().unwrap();
                    match (0..n).find(|&r| mask[r][c] == Mark::Star) {
                        Some(r) => {
                            path.push((r, c));
                            let pc = (0..n).find(|&cc| mask[r][cc] == Mark::Prime).unwrap();
                            path.push((r, pc));
                        }
                        None => break,
                    }
                }
                for &(r, c) in &path {
                    mask[r][c] = if mask[r][c] == Mark::Star {
                        Mark::None
                    } else {
                        Mark::Star
                    };
                }
                row_cover.iter_mut().for_each(|c| *c = false);
                col_cover.iter_mut().for_each(|c| *c = false);
                for row in mask.iter_mut() {
                    for m in row.iter_mut() {
                        if *m == Mark::Prime {
                            *m = Mark::None;
                        }
                    }
                }
                step = 3;
            }
            // No uncovered zero remains: shift the reduced-cost matrix by
            // the smallest uncovered value to expose a new one.
            6 => {
                let mut min_val = f64::INFINITY;
                for i in 0..n {
                    if row_cover[i] {
                        continue;
                    }
                    for j in 0..n {
                        if !col_cover[j] && cost[i][j] < min_val {
                            min_val = cost[i][j];
                        }
                    }
                }
                for i in 0..n {
                    for j in 0..n {
                        if row_cover[i] {
                            cost[i][j] += min_val;
                        }
                        if !col_cover[j] {
                            cost[i][j] -= min_val;
                        }
                    }
                }
                step = 4;
            }
            _ => break,
        }
    }

    let mut row_assign = vec![None; n];
    let mut col_assign = vec![None; n];
    for i in 0..n {
        for j in 0..n {
            if mask[i][j] == Mark::Star {
                row_assign[i] = Some(j);
                col_assign[j] = Some(i);
            }
        }
    }

    if nrows == ncols {
        let total_cost: f64 = (0..n)
            .filter_map(|i| row_assign[i].map(|j| padded[i][j]))
            .sum();
        (total_cost, row_assign, col_assign)
    } else {
        trim_solution(&matrix, row_assign, col_assign)
    }
}

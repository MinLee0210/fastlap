use crate::types::LapSolution;
use crate::utils::{pad_to_square, sap_solve, trim_solution};
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

/// A partitioned subproblem in Murty's algorithm.
#[derive(Clone)]
struct Subproblem {
    fixed: Vec<(usize, usize)>,
    forbidden: Vec<(usize, usize)>,
    fixed_count: usize,
}

struct SolvedCandidate {
    cost: f64,
    row_assign: Vec<Option<usize>>,
    col_assign: Vec<Option<usize>>,
    subproblem: Subproblem,
}

impl PartialEq for SolvedCandidate {
    fn eq(&self, other: &Self) -> bool {
        (self.cost - other.cost).abs() < 1e-12
    }
}
impl Eq for SolvedCandidate {}
impl PartialOrd for SolvedCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SolvedCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrdF64(self.cost).cmp(&OrdF64(other.cost))
    }
}

/// Solves a single constrained assignment subproblem with fixed and forbidden edges.
fn solve_subproblem(orig_padded: &[Vec<f64>], dim: usize, sub: &Subproblem) -> Option<LapSolution> {
    let mut modified = orig_padded.to_vec();

    // Mask forbidden edges
    for &(r, c) in &sub.forbidden {
        if r < dim && c < dim {
            modified[r][c] = 1e12;
        }
    }

    // Enforce fixed edges: make all other choices in row r and column c infinitely costly
    for &(r, c) in &sub.fixed {
        if r < dim && c < dim {
            for j in 0..dim {
                if j != c {
                    modified[r][j] = 1e12;
                }
            }
            for i in 0..dim {
                if i != r {
                    modified[i][c] = 1e12;
                }
            }
        }
    }

    let (_, row_assign, col_assign) = sap_solve(&modified);

    // Verify all fixed edges are respected and no forbidden edge was selected
    for &(r, c) in &sub.fixed {
        if row_assign[r] != Some(c) {
            return None;
        }
    }
    for &(r, c) in &sub.forbidden {
        if row_assign[r] == Some(c) {
            return None;
        }
    }

    let total_cost: f64 = (0..dim)
        .filter_map(|i| row_assign[i].map(|j| orig_padded[i][j]))
        .sum();

    Some((total_cost, row_assign, col_assign))
}

/// Solves for the k-best assignments using Murty's algorithm.
pub fn solve_kbest(matrix: Vec<Vec<f64>>, k: usize) -> Vec<LapSolution> {
    let nrows = matrix.len();
    if nrows == 0 || k == 0 {
        return Vec::new();
    }
    let ncols = matrix[0].len();
    let fill = matrix
        .iter()
        .flatten()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        + 1.0;
    let padded = pad_to_square(&matrix, fill);
    let dim = padded.len();

    let initial_sub = Subproblem {
        fixed: Vec::new(),
        forbidden: Vec::new(),
        fixed_count: 0,
    };

    let mut heap: BinaryHeap<Reverse<SolvedCandidate>> = BinaryHeap::new();

    if let Some((cost, r_assign, c_assign)) = solve_subproblem(&padded, dim, &initial_sub) {
        heap.push(Reverse(SolvedCandidate {
            cost,
            row_assign: r_assign,
            col_assign: c_assign,
            subproblem: initial_sub,
        }));
    }

    let mut solutions: Vec<LapSolution> = Vec::with_capacity(k);

    while let Some(Reverse(cand)) = heap.pop() {
        // Record solution (trimmed to original dimensions)
        let trimmed = if nrows == ncols {
            (cand.cost, cand.row_assign.clone(), cand.col_assign.clone())
        } else {
            trim_solution(&matrix, cand.row_assign.clone(), cand.col_assign.clone())
        };
        solutions.push(trimmed);

        if solutions.len() >= k {
            break;
        }

        // Branch subproblems according to Murty's scheme
        let active_pairs: Vec<(usize, usize)> = cand
            .row_assign
            .iter()
            .enumerate()
            .filter_map(|(r, opt_c)| opt_c.map(|c| (r, c)))
            .collect();

        let mut current_fixed = cand.subproblem.fixed.clone();

        for &(r, c) in active_pairs.iter().skip(cand.subproblem.fixed_count) {
            // New subproblem: fixed current_fixed, forbid (r, c)
            let mut new_forbidden = cand.subproblem.forbidden.clone();
            new_forbidden.push((r, c));

            let new_sub = Subproblem {
                fixed: current_fixed.clone(),
                forbidden: new_forbidden,
                fixed_count: current_fixed.len(),
            };

            if let Some((sub_cost, sub_r, sub_c)) = solve_subproblem(&padded, dim, &new_sub) {
                heap.push(Reverse(SolvedCandidate {
                    cost: sub_cost,
                    row_assign: sub_r,
                    col_assign: sub_c,
                    subproblem: new_sub,
                }));
            }

            // Fix (r, c) for subsequent branches
            current_fixed.push((r, c));
        }
    }

    solutions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_murty_kbest() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        // 2x2 matrix has 2 possible permutations:
        // 1. (0->0: 1, 1->1: 4) = cost 5.0
        // 2. (0->1: 2, 1->0: 3) = cost 5.0
        let res = solve_kbest(matrix, 5);
        assert_eq!(res.len(), 2);
        assert!((res[0].0 - 5.0).abs() < 1e-6);
        assert!((res[1].0 - 5.0).abs() < 1e-6);
    }
}

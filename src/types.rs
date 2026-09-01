/// A standard type for the result of a Linear Assignment Problem algorithm.
/// Returns a tuple containing:
/// 1. The total optimal cost (f64).
/// 2. The assignment mapping from rows to columns (`Vec<Option<usize>>`).
///    `row_assign[i]` gives the column assigned to row `i`, or `None` if unassigned.
/// 3. The assignment mapping from columns to rows (`Vec<Option<usize>>`).
///    `col_assign[j]` gives the row assigned to column `j`, or `None` if unassigned.
pub type LapSolution = (f64, Vec<Option<usize>>, Vec<Option<usize>>);

/// A sparse cost matrix in row-adjacency form: `rows[i]` lists the explicit
/// `(col, cost)` entries for row `i`. Any `(i, j)` pair not present is
/// treated as an infinite (forbidden) cost — the same convention the crate
/// already uses when densifying scipy CSR input (see `matrix::extract_sparse_matrix`).
///
/// Unlike a densified `Vec<Vec<f64>>`, this never materializes the implicit
/// `nrows * ncols` entries, so algorithms built on it (see `lap::lapmod`)
/// scale with the number of explicit edges rather than the full matrix area.
#[derive(Clone)]
pub struct SparseCost {
    pub nrows: usize,
    pub ncols: usize,
    pub rows: Vec<Vec<(usize, f64)>>,
}

impl SparseCost {
    /// Negate every explicit entry (missing entries stay forbidden), turning
    /// a maximum-weight problem into an equivalent minimum-cost one.
    pub fn negate(&self) -> SparseCost {
        SparseCost {
            nrows: self.nrows,
            ncols: self.ncols,
            rows: self
                .rows
                .iter()
                .map(|row| row.iter().map(|&(j, c)| (j, -c)).collect())
                .collect(),
        }
    }

    /// Recompute total cost from this (original, untransformed) sparse
    /// matrix given a row assignment produced by solving a transformed copy.
    pub fn cost_of(&self, row_assign: &[Option<usize>]) -> f64 {
        row_assign
            .iter()
            .enumerate()
            .filter_map(|(i, opt_j)| {
                opt_j.and_then(|j| {
                    self.rows[i]
                        .iter()
                        .find(|&&(jj, _)| jj == j)
                        .map(|&(_, c)| c)
                })
            })
            .sum()
    }
}

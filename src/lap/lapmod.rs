use crate::types::{LapSolution, SparseCost};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Total ordering wrapper for `f64` used in the augmenting-path priority
/// queue. Costs are validated finite/non-NaN before reaching this module
/// (see `matrix::extract_sparse_adjacency` / `validate_matrix`), and no
/// arithmetic here mixes infinities, so `partial_cmp` never returns `None`.
#[derive(PartialEq)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("cost values are always finite, non-NaN")
    }
}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Solves the LAP using LAPMOD — a sparse-aware shortest-augmenting-path
/// solver that operates on a row-adjacency list of explicit `(col, cost)`
/// entries, rather than a densified `nrows x ncols` matrix.
///
/// Missing `(row, col)` pairs are treated as infinitely costly (forbidden),
/// matching the convention used elsewhere in this crate when densifying
/// scipy CSR input. This is what lets a `solve_lap(csr_matrix, "lapmod")`
/// call skip densification entirely (see `matrix::extract_sparse_adjacency`).
///
/// For a square `nrows == ncols` input, this touches only the explicit
/// sparse entries — no padding at all. Rectangular input still needs the
/// same trick `utils::pad_to_square` uses for the dense algorithms: a row
/// insertion has to have *somewhere* to send a displaced match, or later
/// rows can never improve on whichever earlier rows happened to grab the
/// scarce side's columns first. Here that "somewhere" is added as a small
/// number of explicit high-cost slack edges (`dim * |nrows - ncols|` of
/// them) rather than densifying the whole matrix, so the padding cost scales
/// with the *rectangular imbalance*, not with `nrows * ncols`.
pub fn solve_sparse(sc: &SparseCost) -> LapSolution {
    let nrows = sc.nrows;
    let ncols = sc.ncols;
    if nrows == 0 || ncols == 0 {
        return (0.0, vec![None; nrows], vec![None; ncols]);
    }

    let dim = nrows.max(ncols);
    let fill = sc
        .rows
        .iter()
        .flat_map(|row| row.iter().map(|&(_, c)| c))
        .fold(f64::NEG_INFINITY, f64::max);
    let fill = if fill.is_finite() { fill + 1.0 } else { 1.0 };

    let rows: Vec<Vec<(usize, f64)>> = (0..dim)
        .map(|i| {
            let mut adj: Vec<(usize, f64)> = if i < nrows {
                sc.rows[i].clone()
            } else {
                Vec::new()
            };
            // Every row — real or virtual — can reach every virtual column
            // at `fill` cost, giving augmenting paths a slack target to
            // displace into once the real columns are all spoken for.
            for j in ncols..dim {
                adj.push((j, fill));
            }
            if i >= nrows {
                // Virtual row: can also reach every real column at `fill`
                // cost, so it never displaces a cheaper real assignment.
                adj.extend((0..ncols).map(|j| (j, fill)));
            }
            adj
        })
        .collect();

    let row_assign_full = augmenting_search(dim, dim, &rows);

    let row_assign: Vec<Option<usize>> = row_assign_full[..nrows]
        .iter()
        .map(|&opt_j| opt_j.filter(|&j| j < ncols))
        .collect();
    let mut col_assign: Vec<Option<usize>> = vec![None; ncols];
    for (i, opt_j) in row_assign.iter().enumerate() {
        if let Some(j) = opt_j {
            col_assign[*j] = Some(i);
        }
    }

    let total_cost = sc.cost_of(&row_assign);
    (total_cost, row_assign, col_assign)
}

/// Core shortest-augmenting-path search on a *square* sparse adjacency list
/// (`rows.len() == ncols == n`). Returns the row -> column assignment; every
/// row is guaranteed a match as long as the graph doesn't leave some row
/// with zero reachable columns (callers needing rectangular or
/// possibly-infeasible input should pad first, as `solve_sparse` does).
///
/// Each row insertion runs a Dijkstra-style search with a binary heap, so a
/// single row costs `O(E_touched * log E_touched)` rather than the
/// `O(E_touched^2)` a plain linear "smallest untouched" scan would cost —
/// the difference matters once an augmenting/displacement chain has to walk
/// through a large fraction of an already-matched sparse graph. Potentials
/// (`u`, `v`) are only "shifted" for columns actually touched, using a
/// lazily-applied running offset (`total_shift` / `shift_at_use`) instead of
/// rewriting every entry on every step — this is what keeps a step from
/// costing more than the handful of edges it actually relaxes.
fn augmenting_search(n: usize, ncols: usize, rows: &[Vec<(usize, f64)>]) -> Vec<Option<usize>> {
    // 1-indexed storage; p[j] = row matched to column j (0 = free column).
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; ncols + 1];
    let mut p = vec![0usize; ncols + 1];
    let mut way = vec![0usize; ncols + 1];
    // raw[j] - total_shift is the true current reduced-cost distance to j;
    // see the module-level comment on why this avoids an O(ncols) rewrite
    // of every entry each time the running shift changes.
    let mut raw = vec![0.0f64; ncols + 1];
    let mut used = vec![false; ncols + 1];
    let mut is_touched = vec![false; ncols + 1];
    let mut shift_at_use = vec![0.0f64; ncols + 1];
    let mut heap: BinaryHeap<Reverse<(OrdF64, usize)>> = BinaryHeap::new();

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut total_shift = 0.0f64;
        // Columns whose state got modified this round — reset just these
        // at the end instead of the full O(ncols) arrays.
        let mut touched: Vec<usize> = Vec::new();
        let mut feasible = true;
        heap.clear();

        loop {
            used[j0] = true;
            shift_at_use[j0] = total_shift;
            if !is_touched[j0] {
                is_touched[j0] = true;
                touched.push(j0);
            }
            let i0 = p[j0];

            // Relax only the real sparse edges out of row i0. A column
            // absent from this adjacency list is implicitly cost=infinity,
            // so it can never beat an existing finite distance — skipping
            // it is exactly equivalent to relaxing it and losing.
            for &(j_idx, cost_val) in &rows[i0 - 1] {
                let j = j_idx + 1;
                if !used[j] {
                    let cur = cost_val - u[i0] - v[j];
                    let candidate_raw = cur + total_shift;
                    if !is_touched[j] || candidate_raw < raw[j] {
                        raw[j] = candidate_raw;
                        way[j] = j0;
                        if !is_touched[j] {
                            is_touched[j] = true;
                            touched.push(j);
                        }
                        heap.push(Reverse((OrdF64(candidate_raw), j)));
                    }
                }
            }

            // Pop the smallest not-yet-used, not-stale entry. A heap entry
            // is stale if `raw[j]` was improved by a later relaxation after
            // this entry was pushed — cheaper than removing it in place.
            let mut found: Option<(f64, usize)> = None;
            while let Some(Reverse((OrdF64(raw_val), j))) = heap.pop() {
                if !used[j] && raw_val == raw[j] {
                    found = Some((raw_val, j));
                    break;
                }
            }

            let Some((raw_val, j1)) = found else {
                // No column is reachable from row i at all.
                feasible = false;
                break;
            };
            let delta = raw_val - total_shift;
            total_shift += delta;

            j0 = j1;
            if p[j0] == 0 {
                break; // Reached a free column; augmenting path is complete.
            }
        }

        // Finalize potentials for every column touched this round, using
        // the running shift each became used under, before p[] changes
        // during the flip below.
        for &j in &touched {
            if used[j] {
                let extra = total_shift - shift_at_use[j];
                u[p[j]] += extra;
                v[j] -= extra;
            }
        }

        if feasible {
            loop {
                let j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
                if j0 == 0 {
                    break;
                }
            }
        }

        for &j in &touched {
            used[j] = false;
            is_touched[j] = false;
        }
    }

    let mut row_assign: Vec<Option<usize>> = vec![None; n];
    for j in 1..=ncols {
        if p[j] != 0 {
            row_assign[p[j] - 1] = Some(j - 1);
        }
    }
    row_assign
}

/// Dense entry point: every cell of `matrix` is treated as an explicit
/// (finite) edge, so this is exactly [`solve_sparse`] with a fully-populated
/// adjacency list — used whenever the caller supplies a plain array rather
/// than a scipy CSR matrix.
pub fn solve(matrix: Vec<Vec<f64>>) -> LapSolution {
    let nrows = matrix.len();
    if nrows == 0 {
        return (0.0, vec![], vec![]);
    }
    let ncols = matrix[0].len();
    let rows: Vec<Vec<(usize, f64)>> = matrix
        .iter()
        .map(|row| row.iter().enumerate().map(|(j, &c)| (j, c)).collect())
        .collect();

    solve_sparse(&SparseCost { nrows, ncols, rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression test for a duplicate-in-`touched` bug that double-applied a
    // potential shift and broke dual feasibility (see git history): this
    // specific 4x4 matrix reliably triggered it before the `is_touched` fix.
    #[test]
    fn matches_known_optimum() {
        let matrix = vec![
            vec![
                80.61939890460857,
                70.38885835403663,
                10.022688731230112,
                91.94826137446735,
            ],
            vec![
                71.42412995491114,
                99.88470065678665,
                14.944830465799374,
                86.81260573682142,
            ],
            vec![
                16.249293467637482,
                61.55595642838442,
                12.381998284944151,
                84.80082293222344,
            ],
            vec![
                80.73189587250107,
                56.91007386145933,
                40.71832972259997,
                6.916699545513804,
            ],
        ];
        let (cost, _, _) = solve(matrix);
        assert!((cost - 108.49968183298729).abs() < 1e-6, "got {cost}");
    }

    #[test]
    fn rectangular_matches_dense_optimum() {
        // 12x5, regression case for the "no slack to displace into" bug.
        let matrix: Vec<Vec<f64>> = (0..12)
            .map(|i| (0..5).map(|j| ((i * 7 + j * 13) % 97) as f64).collect())
            .collect();
        let (cost, row_assign, _) = solve(matrix.clone());
        assert_eq!(row_assign.iter().filter(|o| o.is_some()).count(), 5);

        // Brute-force optimum over all 5-subsets of 12 rows for cross-check.
        fn min_cost_bruteforce(m: &[Vec<f64>]) -> f64 {
            let nrows = m.len();
            let ncols = m[0].len();
            let mut best = f64::INFINITY;
            let mut rows: Vec<usize> = (0..nrows).collect();
            // permutations of columns assigned to a chosen ordered subset of rows
            fn permute(rows: &[usize], m: &[Vec<f64>], ncols: usize, best: &mut f64) {
                if rows.len() < ncols {
                    return;
                }
                // choose ncols rows out of rows, then all permutations of columns
                let n = rows.len();
                let mut idx: Vec<usize> = (0..ncols).collect();
                loop {
                    let chosen: Vec<usize> = idx.iter().map(|&k| rows[k]).collect();
                    let mut perm: Vec<usize> = (0..ncols).collect();
                    loop {
                        let cost: f64 = (0..ncols).map(|c| m[chosen[c]][perm[c]]).sum();
                        if cost < *best {
                            *best = cost;
                        }
                        if !next_permutation(&mut perm) {
                            break;
                        }
                    }
                    if !next_combination(&mut idx, n) {
                        break;
                    }
                }
            }
            fn next_permutation(a: &mut [usize]) -> bool {
                let n = a.len();
                if n < 2 {
                    return false;
                }
                let mut i = n - 1;
                while i > 0 && a[i - 1] >= a[i] {
                    i -= 1;
                }
                if i == 0 {
                    return false;
                }
                let mut j = n - 1;
                while a[j] <= a[i - 1] {
                    j -= 1;
                }
                a.swap(i - 1, j);
                a[i..].reverse();
                true
            }
            fn next_combination(idx: &mut [usize], n: usize) -> bool {
                let k = idx.len();
                let mut i = k;
                loop {
                    if i == 0 {
                        return false;
                    }
                    i -= 1;
                    if idx[i] != i + n - k {
                        break;
                    }
                }
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                true
            }
            permute(&rows, m, ncols, &mut best);
            rows.clear();
            best
        }
        let expected = min_cost_bruteforce(&matrix);
        assert!(
            (cost - expected).abs() < 1e-6,
            "got {cost}, expected {expected}"
        );
    }
}

use crate::lap::lapmod::{
    augmenting_search_warm, build_square_adjacency, finalize_sparse_solution,
};
use crate::types::{LapSolution, SparseCost};

/// Solves the LAP with **LAPJVsp** — the sparse adaptation of the
/// Jonker-Volgenant algorithm, implemented directly on a row-adjacency list of
/// explicit `(col, cost)` entries rather than a densified `nrows x ncols`
/// matrix. Missing `(row, col)` pairs are treated as infinitely costly
/// (forbidden), exactly as in LAPMOD.
///
/// The JV preprocessing is what separates this from LAPMOD: a cheap *sparse*
/// column-reduction pass (each column claims its cheapest reachable row, a row
/// keeps only its cheapest claimant, then single-claim rows transfer slack
/// into their assigned column's dual) resolves most rows for free and yields a
/// feasible, complementary dual pair `(u, v)`. Only the rows that reduction
/// left unclaimed pay for a warm-started shortest-augmenting-path search —
/// see [`solve_sparse`] for the dispatch, and `lap::lapmod` for the shared
/// sparse augmenting-path machinery.
pub fn solve_sparse(sc: &SparseCost) -> LapSolution {
    let nrows = sc.nrows;
    let ncols = sc.ncols;
    if nrows == 0 || ncols == 0 {
        return (0.0, vec![None; nrows], vec![None; ncols]);
    }

    let (dim, _, rows) = build_square_adjacency(sc);
    let full = column_reduction_warm(dim, &rows);
    finalize_sparse_solution(sc, full)
}

/// Phase 1 (column reduction) + Phase 2 (reduction transfer) over the padded
/// sparse adjacency, warm-starting the augmenting-path completion with the
/// resulting partial matching and feasible duals.
fn column_reduction_warm(dim: usize, rows: &[Vec<(usize, f64)>]) -> Vec<Option<usize>> {
    // Transpose to column adjacency once (O(E)); each real/virtual column then
    // scans only its own explicit entries to find its cheapest row.
    let mut col_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); dim];
    for (i, row) in rows.iter().enumerate() {
        for &(j, c) in row {
            col_adj[j].push((i, c));
        }
    }

    // Phase 1: column reduction. For each column, find its cheapest reachable
    // row and tentatively claim that row for it; a row keeps only its cheapest
    // claimant (by column dual value) if claimed by more than one column.
    let mut v = vec![0.0f64; dim];
    let mut row_assign: Vec<Option<usize>> = vec![None; dim];
    let mut claims = vec![0usize; dim];

    for j in 0..dim {
        let mut best_i = None;
        let mut best_val = f64::INFINITY;
        for &(i, c) in &col_adj[j] {
            if c < best_val {
                best_val = c;
                best_i = Some(i);
            }
        }
        let Some(bi) = best_i else {
            continue; // Unreachable column — cannot take part in a matching.
        };
        v[j] = best_val;
        claims[bi] += 1;

        match row_assign[bi] {
            None => row_assign[bi] = Some(j),
            Some(j1) if best_val < v[j1] => row_assign[bi] = Some(j),
            _ => {}
        }
    }

    // Phase 2: reduction transfer. Rows claimed by exactly one column can
    // have their assigned column's dual tightened using the row's second-best
    // reduced cost, shrinking the search space for phase 3 without touching
    // the (still-optimal) partial matching.
    for i in 0..dim {
        if claims[i] != 1 {
            continue;
        }
        let j1 = row_assign[i].unwrap();
        let mut min_other = f64::INFINITY;
        for &(j, c) in &rows[i] {
            if j != j1 {
                let reduced = c - v[j];
                if reduced < min_other {
                    min_other = reduced;
                }
            }
        }
        if min_other.is_finite() {
            v[j1] -= min_other;
        }
    }

    // Feasible row duals consistent with v: u[i] = min_j (cost[i][j] - v[j]),
    // which may legitimately be negative after a reduction transfer. For rows
    // resolved by column reduction this makes the reduced cost exactly zero at
    // their assigned column (complementary slackness by construction), so the
    // partial matching is optimal under (u, v) and phase 3 only needs to
    // extend it to the remaining free rows.
    let u: Vec<f64> = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|&(j, c)| c - v[j])
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    // Phase 3: complete the assignment for rows column reduction left
    // unresolved via warm-started shortest-augmenting-path search.
    augmenting_search_warm(dim, dim, rows, &u, &v, &row_assign)
}

/// Dense entry point: every cell is an explicit finite edge, so this is
/// exactly [`solve_sparse`] with a fully-populated adjacency list.
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
    use crate::lap::lapmod::solve_sparse as lapmod_sparse;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / ((1u64 << 31) as f64)
    }

    fn random_sc(n: usize, density: f64, seed: &mut u64) -> SparseCost {
        let mut sc = SparseCost {
            nrows: n,
            ncols: n,
            rows: vec![Vec::new(); n],
        };
        for i in 0..n {
            let mut has_diag = false;
            for j in 0..n {
                if i == j || lcg(seed) < density {
                    if i == j {
                        has_diag = true;
                    }
                    sc.rows[i].push((j, 1.0 + 99.0 * lcg(seed)));
                }
            }
            if !has_diag {
                sc.rows[i].push((i, 1.0 + 99.0 * lcg(seed)));
            }
        }
        sc
    }

    fn cost_of(sc: &SparseCost, assign: &[Option<usize>]) -> f64 {
        assign
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.map(|j| sc.rows[i].iter().find(|&&(jj, _)| jj == j).unwrap().1))
            .sum()
    }

    #[test]
    fn sparse_matches_lapmod_on_random_square() {
        let mut seed = 123u64;
        for n in [3usize, 5, 10, 16, 24] {
            for _ in 0..30 {
                let sc = random_sc(n, 0.55, &mut seed);
                let a = solve_sparse(&sc);
                let b = lapmod_sparse(&sc);
                assert!(
                    (a.0 - b.0).abs() < 1e-9,
                    "n={n} lapjvsp {:.9} != lapmod {:.9}",
                    a.0,
                    b.0
                );
                assert!(
                    !a.1.contains(&None),
                    "lapjvsp left a row unassigned (n={n})"
                );
            }
        }
    }

    #[test]
    fn sparse_matches_bruteforce_on_small() {
        // Brute force every perfect matching for tiny random sparse graphs and
        // compare against the (possibly non-perfect) sparse optimum.
        let mut seed = 4242u64;
        for n in [2usize, 3, 4] {
            for _ in 0..40 {
                let sc = random_sc(n, 0.6, &mut seed);
                let (got, assign, _) = solve_sparse(&sc);
                let lapmod_cost = lapmod_sparse(&sc).0;
                assert!(assign.iter().all(|o| o.is_some()), "no perfect matching");
                assert!(
                    (got - lapmod_cost).abs() < 1e-9,
                    "n={n} lapjvsp {got:.9} != lapmod {lapmod_cost:.9}"
                );

                // brute force over all permutations
                let mut perm: Vec<usize> = (0..n).collect();
                let mut best = f64::INFINITY;
                loop {
                    let mut total = 0.0;
                    let mut feasible = true;
                    for (i, &j) in perm.iter().enumerate() {
                        match sc.rows[i].iter().find(|&&(jj, _)| jj == j) {
                            Some(&(_, c)) => total += c,
                            None => {
                                feasible = false;
                                break;
                            }
                        }
                    }
                    if feasible && total < best {
                        best = total;
                    }
                    if !next_perm(&mut perm) {
                        break;
                    }
                }
                assert!(
                    (got - best).abs() < 1e-9,
                    "n={n} lapjvsp {got:.9} != brute {best:.9} (cost_of {})",
                    cost_of(&sc, &assign)
                );
            }
        }
    }

    fn next_perm(a: &mut [usize]) -> bool {
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

    #[test]
    fn dense_matches_lapmod() {
        let matrix = vec![
            vec![80.6, 70.3, 10.0, 91.9],
            vec![71.4, 99.8, 14.9, 86.8],
            vec![16.2, 61.5, 12.3, 84.8],
            vec![80.7, 56.9, 40.7, 6.9],
        ];
        let (cost, _, _) = solve(matrix.clone());
        let ref_cost = crate::lap::lapmod::solve(matrix).0;
        assert!(
            (cost - ref_cost).abs() < 1e-9,
            "lapjvsp {cost} vs lapmod {ref_cost}"
        );
    }

    #[test]
    fn rectangular_sparse_matches_lapmod() {
        let mut seed = 555u64;
        for (nrows, ncols) in [(12usize, 5usize), (5, 12), (7, 9), (3, 8)] {
            let mut sc = SparseCost {
                nrows,
                ncols,
                rows: vec![Vec::new(); nrows],
            };
            for i in 0..nrows {
                for j in 0..ncols {
                    sc.rows[i].push((j, 1.0 + 99.0 * lcg(&mut seed)));
                }
            }
            let a = solve_sparse(&sc);
            let b = lapmod_sparse(&sc);
            let (ca, cb) = (a.0, b.0);
            assert!(
                (ca - cb).abs() < 1e-9,
                "{nrows}x{ncols}: lapjvsp {ca} vs lapmod {cb}"
            );
        }
    }
}

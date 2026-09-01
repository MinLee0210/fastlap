use crate::types::LapSolution;
use crate::utils::{pad_to_square, trim_solution};
use std::collections::VecDeque;

/// Solves the LAP using Dantzig's primal simplex method (the classical
/// "transportation simplex" / MODI method), applied directly to the
/// assignment LP's spanning-tree basis.
///
/// Unlike the dual/augmenting-path algorithms used elsewhere in this crate,
/// this maintains an explicit basic feasible solution as a spanning tree
/// over `2n` nodes (`n` rows + `n` columns, `2n-1` tree edges/basic cells),
/// and repeatedly:
/// 1. Solves for node potentials `u[i], v[j]` via a single tree traversal
///    (`u[i] + v[j] = cost[i][j]` for every basic cell).
/// 2. Picks the non-basic cell with the **most negative reduced cost**
///    (`cost[i][j] - u[i] - v[j]`) as the entering variable — this is
///    Dantzig's original pivoting rule.
/// 3. Finds the unique cycle the entering edge creates in the tree, and
///    re-bases along it (the "stepping-stone" pivot).
///
/// The assignment LP's constraint matrix is totally unimodular with unit
/// supplies/demands, so every basic solution here stays integral (every
/// basic cell's flow is exactly 0 or 1) — at optimality, the flow=1 cells
/// are exactly the assignment. This also means the problem is maximally
/// degenerate (`n` real assignment edges out of `2n-1` basic cells, the
/// rest carrying zero flow), which is precisely the case Dantzig's rule
/// alone can cycle on; a Bland's-rule fallback after a run of degenerate
/// pivots guarantees termination without giving up the Dantzig pivoting
/// this algorithm is named for.
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

    let row_assign = simplex_solve(&padded, n);
    let col_assign = invert(&row_assign, n);

    if nrows == ncols {
        let total_cost: f64 = (0..n)
            .filter_map(|i| row_assign[i].map(|j| padded[i][j]))
            .sum();
        (total_cost, row_assign, col_assign)
    } else {
        trim_solution(&matrix, row_assign, col_assign)
    }
}

fn invert(row_assign: &[Option<usize>], n: usize) -> Vec<Option<usize>> {
    let mut col_assign = vec![None; n];
    for (i, opt_j) in row_assign.iter().enumerate() {
        if let Some(j) = opt_j {
            col_assign[*j] = Some(i);
        }
    }
    col_assign
}

/// Tree-BFS from node 0 to solve `u[i] + v[j] = cost[i][j]` for every basic
/// cell. Node `r` (0..n) is row `r`; node `n+c` is column `c`.
fn compute_potentials(
    cost: &[Vec<f64>],
    n: usize,
    tree_adj: &[Vec<usize>],
) -> (Vec<f64>, Vec<f64>) {
    let mut u = vec![f64::NAN; n];
    let mut v = vec![f64::NAN; n];
    u[0] = 0.0;
    let mut visited = vec![false; 2 * n];
    visited[0] = true;
    let mut queue = VecDeque::new();
    queue.push_back(0usize);
    while let Some(node) = queue.pop_front() {
        for &nb in &tree_adj[node] {
            if !visited[nb] {
                visited[nb] = true;
                if node < n {
                    let (r, c) = (node, nb - n);
                    v[c] = cost[r][c] - u[r];
                } else {
                    let (r, c) = (nb, node - n);
                    u[r] = cost[r][c] - v[c];
                }
                queue.push_back(nb);
            }
        }
    }
    (u, v)
}

/// BFS parent-pointer path from node `start` to node `target` in the tree.
fn tree_path(tree_adj: &[Vec<usize>], start: usize, target: usize, total: usize) -> Vec<usize> {
    let mut parent = vec![usize::MAX; total];
    let mut visited = vec![false; total];
    visited[start] = true;
    let mut queue = VecDeque::new();
    queue.push_back(start);
    while let Some(node) = queue.pop_front() {
        if node == target {
            break;
        }
        for &nb in &tree_adj[node] {
            if !visited[nb] {
                visited[nb] = true;
                parent[nb] = node;
                queue.push_back(nb);
            }
        }
    }
    let mut path = vec![target];
    while *path.last().unwrap() != start {
        path.push(parent[*path.last().unwrap()]);
    }
    path.reverse();
    path
}

fn simplex_solve(cost: &[Vec<f64>], n: usize) -> Vec<Option<usize>> {
    let mut flow = vec![vec![0.0f64; n]; n];
    let mut is_basic = vec![vec![false; n]; n];
    let mut tree_adj: Vec<Vec<usize>> = vec![Vec::new(); 2 * n];

    // Initial basic feasible solution: a greedy min-cost assignment (each
    // row claims its cheapest still-available column) rather than the
    // textbook Northwest-Corner rule. NW-corner ignores cost entirely, so
    // it typically starts many pivots away from optimal; starting from a
    // decent greedy assignment instead cuts the pivot count dramatically
    // (this is the difference between the benchmark suite finishing in
    // seconds vs. minutes at n=50). The `n` greedy edges initially form `n`
    // disconnected 2-node components; union-find then adds exactly `n-1`
    // more degenerate (zero-flow) edges to connect them into one spanning
    // tree, which is all a basic feasible solution requires.
    let mut col_taken = vec![false; n];
    let mut uf: Vec<usize> = (0..2 * n).collect();
    fn find(uf: &mut [usize], x: usize) -> usize {
        let mut x = x;
        while uf[x] != x {
            uf[x] = uf[uf[x]];
            x = uf[x];
        }
        x
    }
    fn union(uf: &mut [usize], a: usize, b: usize) -> bool {
        let (ra, rb) = (find(uf, a), find(uf, b));
        if ra == rb {
            false
        } else {
            uf[ra] = rb;
            true
        }
    }

    for r in 0..n {
        let mut best_c = 0usize;
        let mut best_val = f64::INFINITY;
        for c in 0..n {
            if !col_taken[c] && cost[r][c] < best_val {
                best_val = cost[r][c];
                best_c = c;
            }
        }
        col_taken[best_c] = true;
        flow[r][best_c] = 1.0;
        is_basic[r][best_c] = true;
        tree_adj[r].push(n + best_c);
        tree_adj[n + best_c].push(r);
        union(&mut uf, r, n + best_c);
    }

    let mut edges_needed = n - 1;
    'connect: for r in 0..n {
        if edges_needed == 0 {
            break;
        }
        for c in 0..n {
            if edges_needed == 0 {
                break 'connect;
            }
            if !is_basic[r][c] && union(&mut uf, r, n + c) {
                is_basic[r][c] = true; // flow stays 0.0: a degenerate connector.
                tree_adj[r].push(n + c);
                tree_adj[n + c].push(r);
                edges_needed -= 1;
            }
        }
    }

    let eps = 1e-9;
    let max_iters = 50 * n * n + 1000;
    let mut degenerate_streak = 0usize;
    let bland_threshold = 4 * n + 16;

    for _ in 0..max_iters {
        let (u, v) = compute_potentials(cost, n, &tree_adj);

        let use_bland = degenerate_streak >= bland_threshold;
        let mut enter: Option<(usize, usize)> = None;
        let mut best = -eps;
        'search: for r in 0..n {
            for c in 0..n {
                if !is_basic[r][c] {
                    let reduced = cost[r][c] - u[r] - v[c];
                    if use_bland {
                        if reduced < -eps {
                            enter = Some((r, c));
                            break 'search;
                        }
                    } else if reduced < best {
                        best = reduced;
                        enter = Some((r, c));
                    }
                }
            }
        }

        let Some((ei, ej)) = enter else {
            break; // No improving non-basic cell: optimal.
        };

        let path = tree_path(&tree_adj, ei, n + ej, 2 * n);

        // Cycle: entering cell is '+', then alternate starting at '-' for
        // the tree edge nearest the column end of the path.
        let mut cycle: Vec<((usize, usize), bool)> = vec![((ei, ej), true)];
        let mut sign = false;
        for w in (1..path.len()).rev() {
            let (a, b) = (path[w - 1], path[w]);
            let (r, c) = if a < n { (a, b - n) } else { (b, a - n) };
            cycle.push(((r, c), sign));
            sign = !sign;
        }

        let mut theta = f64::INFINITY;
        let mut leave: Option<(usize, usize)> = None;
        for &((r, c), s) in &cycle {
            if !s {
                let f = flow[r][c];
                let better = f < theta - eps
                    || (f < theta + eps && leave.is_none_or(|(lr, lc)| (r, c) < (lr, lc)));
                if better {
                    theta = f;
                    leave = Some((r, c));
                }
            }
        }
        let Some((lr, lc)) = leave else {
            break; // A well-formed cycle always has a '-' cell; defensive.
        };

        if theta <= eps {
            degenerate_streak += 1;
        } else {
            degenerate_streak = 0;
        }

        for &((r, c), s) in &cycle {
            if s {
                flow[r][c] += theta;
            } else {
                flow[r][c] -= theta;
            }
        }

        is_basic[ei][ej] = true;
        tree_adj[ei].push(n + ej);
        tree_adj[n + ej].push(ei);
        is_basic[lr][lc] = false;
        tree_adj[lr].retain(|&x| x != n + lc);
        tree_adj[n + lc].retain(|&x| x != lr);
    }

    (0..n)
        .map(|r| (0..n).find(|&c| is_basic[r][c] && flow[r][c] > 0.5))
        .collect()
}

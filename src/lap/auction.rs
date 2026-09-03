use crate::types::LapSolution;
use crate::utils::{pad_to_square, sap_solve, trim_solution};
use std::collections::VecDeque;

/// Maximum number of auction bids allowed in a single epsilon-scaling phase.
/// Every bid assigns a bidder (or displaces one), so this bound keeps a
/// pathological phase from running forever; the loop below usually
/// converges in far fewer bids because prices are warm-started.
const MAX_PHASE_BIDS: usize = 200_000;

/// Maximum number of epsilon-scaling phases (each halves epsilon).
const MAX_PHASES: usize = 60;

/// Solves the LAP using the Auction algorithm (Bertsekas, 1988) for cost
/// minimization, with **epsilon-scaling**.
///
/// Each bidder (row) bids on the item (column) with the lowest adjusted cost
/// `matrix[i][j] + price[j]`, raising that item's price to deter future
/// competition. The algorithm terminates with an ε-optimal solution: the
/// total cost is at most `n · ε` above the true optimum.
///
/// A single fixed ε is catastrophic for integer- or tie-heavy cost matrices:
/// breaking a tie needs the price to move by ~1, so with `ε ≈ 1e-9` that is
/// ~1e9 bids and the loop exhausts its budget with rows still unassigned
/// (silently returning a *partial*, sub-optimal assignment). Instead we run
/// the auction at a coarse ε and repeatedly halve it, warm-starting prices
/// between phases — the coarse phases resolve ties cheaply, and the later
/// fine phases only polish prices that are already near-optimal, so the whole
/// sweep converges quickly and reaches the documented `n · ε_final` gap.
///
/// If the bidding budget is ever exhausted with rows still unassigned — the
/// residual pathological case the ε-scaling sweep exists to avoid — this
/// returns an `Err` rather than quietly handing back a partial assignment
/// that `solve_lap` would misreport as an optimal cost.
pub fn solve(matrix: Vec<Vec<f64>>) -> Result<LapSolution, String> {
    let n = matrix.len();
    if n == 0 {
        return Ok((0.0, vec![], vec![]));
    }
    let m = matrix[0].len();
    if n != m {
        // Rectangular matrices are padded and solved with SAP, then trimmed
        // back to the original dimensions. The trim is essential: a padded
        // solution has `dim` rows/columns and would index out of bounds when
        // `solve_lap` recomputes the cost from the unpadded matrix.
        let fill = matrix
            .iter()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            + 1.0;
        let padded = pad_to_square(&matrix, fill);
        let (_, row_assign, col_assign) = sap_solve(&padded);
        return Ok(trim_solution(&matrix, row_assign, col_assign));
    }

    let (max_cost, min_cost) = matrix
        .iter()
        .flatten()
        .fold((f64::NEG_INFINITY, f64::INFINITY), |(hi, lo), &v| {
            (hi.max(v), lo.min(v))
        });
    let cost_scale = max_cost.abs().max(min_cost.abs()).max(1.0);

    // Extreme magnitudes: epsilon-scaling accumulates prices over dozens of
    // phases, and once `price + gamma` exceeds f64::MAX every bidder sees an
    // infinite value on every item and the auction can no longer resolve
    // (previous code hit an `unreachable!` here). Below ~1e150 the accumulated
    // price stays comfortably inside f64 range; above it we fall back to SAP,
    // which is exact and handles such values without overflow.
    if cost_scale > 1e150 {
        return Ok(sap_solve(&matrix));
    }

    let mut prices = vec![0.0f64; n];
    let mut row_assign = vec![None; n];
    let mut col_assign = vec![None; n];

    // Start with a coarse epsilon and halve toward the target tolerance.
    let mut epsilon = (max_cost - min_cost).abs() * 0.5;
    if !epsilon.is_finite() || epsilon == 0.0 {
        epsilon = cost_scale;
    }
    let target = cost_scale * 1e-8;

    for _ in 0..MAX_PHASES {
        if epsilon < target {
            break;
        }
        let complete = auction_pass(
            &matrix,
            &mut prices,
            &mut row_assign,
            &mut col_assign,
            epsilon,
        );
        if !complete {
            break; // bidding budget exhausted within this phase
        }
        epsilon *= 0.5;
    }

    if row_assign.iter().any(|item| item.is_none()) {
        return Err(format!(
            "auction exhausted its bidding budget ({MAX_PHASE_BIDS} bids per phase, \
             {MAX_PHASES} phases) with rows still unassigned; the ε-scaling sweep did not \
             converge. Try \"lapjv\" (exact) or a larger matrix"
        ));
    }

    let total_cost: f64 = row_assign
        .iter()
        .enumerate()
        .filter_map(|(i, item)| item.map(|item| matrix[i][item]))
        .sum();

    Ok((total_cost, row_assign, col_assign))
}

/// Run one auction phase at a fixed `epsilon`, warm-started from the current
/// `prices`. Returns `true` if every row ended up assigned (i.e. the phase
/// converged), `false` if it exhausted `MAX_PHASE_BIDS` first.
///
/// The assignment vectors are reset at the start of the phase; the price
/// vector is kept, which is what carries the learning from coarser phases.
fn auction_pass(
    matrix: &[Vec<f64>],
    prices: &mut [f64],
    row_assign: &mut [Option<usize>],
    col_assign: &mut [Option<usize>],
    epsilon: f64,
) -> bool {
    let n = matrix.len();
    for slot in row_assign.iter_mut() {
        *slot = None;
    }
    for slot in col_assign.iter_mut() {
        *slot = None;
    }
    let mut unassigned: VecDeque<usize> = (0..n).collect();

    for _ in 0..MAX_PHASE_BIDS {
        let Some(bidder) = unassigned.pop_front() else {
            return true; // every row is assigned: phase converged.
        };

        // Minimization: best item is the one with the lowest (cost + price).
        let mut best_item = None;
        let mut best_val = f64::INFINITY;
        let mut second_best_val = f64::INFINITY;

        for item in 0..n {
            let val = matrix[bidder][item] + prices[item];
            if val < best_val {
                second_best_val = best_val;
                best_val = val;
                best_item = Some(item);
            } else if val < second_best_val {
                second_best_val = val;
            }
        }

        let best_item = match best_item {
            Some(item) => item,
            None => unreachable!("n >= 1 guarantees at least one item"),
        };

        // Raise the price of the best item so it becomes less attractive to others.
        let gamma = if second_best_val == f64::INFINITY {
            epsilon // n == 1 or all other items have the same best_val.
        } else {
            second_best_val - best_val + epsilon
        };
        prices[best_item] += gamma;

        // Displace the previous holder of best_item, if any.
        if let Some(prev) = col_assign[best_item] {
            unassigned.push_back(prev);
            row_assign[prev] = None;
        }

        row_assign[bidder] = Some(best_item);
        col_assign[best_item] = Some(bidder);
    }

    false // budget exhausted with rows still unassigned.
}

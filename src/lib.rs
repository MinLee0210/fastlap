#![allow(clippy::needless_range_loop)]
//! fastlap — High-performance LAP solver powered by Rust.
//!
//! Provides `solve_lap` for single matrices and `solve_lap_batch` for parallel
//! solving of many independent matrices.  All six algorithms (LAPJV, Hungarian,
//! LAPMOD, Dantzig, Auction, Subgradient) are exposed through a uniform API.

use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub mod lap;
pub mod matrix;
pub mod types;
pub mod utils;

use crate::matrix::{extract_matrix, extract_sparse_adjacency, is_csr, validate_matrix};
use crate::types::{LapSolution, SparseCost};
use crate::utils::{negate_matrix, recompute_cost, solve_with, supported_algorithms};

// ---------------------------------------------------------------------------
// Python ↔ Rust conversion helpers
// ---------------------------------------------------------------------------

/// A single batch entry, kept in its original form so sparse CSR inputs can be
/// solved without densification (see [`solve_lap_batch`]).
enum BatchMatrix {
    Dense(Vec<Vec<f64>>),
    Sparse(SparseCost),
}

// ---------------------------------------------------------------------------
// Public Python API
// ---------------------------------------------------------------------------

/// Solve a Linear Assignment Problem.
///
/// Parameters
/// ----------
/// cost_matrix : numpy.ndarray or scipy.sparse.csr_matrix
///     An (n x m) cost matrix. Square matrices are solved directly.
///     Rectangular matrices are padded internally and assignments to
///     padded rows/columns are reported as ``None``.
/// algorithm : str
///     One of: ``"lapjv"``, ``"hungarian"``, ``"lapmod"``,
///     ``"dantzig"``, ``"auction"``, ``"subgradient"``.
/// maximize : bool, optional
///     If ``True``, find the maximum-weight assignment instead of the
///     minimum-cost one. Defaults to ``False``.
///
/// Returns
/// -------
/// tuple[float, list[int | None], list[int | None]]
///     ``(total_cost, row_assignments, col_assignments)``.
///     ``row_assign[i]`` is the column assigned to row i, or ``None``.
///     ``col_assign[j]`` is the row assigned to column j, or ``None``.
///
/// Raises
/// ------
/// ValueError
///     If the matrix is empty, non-rectangular, contains NaN/Inf,
///     or the algorithm name is not recognised.
/// TypeError
///     If the input is not a NumPy ndarray or scipy CSR matrix.
///
/// Examples
/// --------
/// >>> import fastlap
/// >>> cost = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
/// >>> total, rows, cols = fastlap.solve_lap(cost, algorithm="lapjv")
/// >>> total, rows, cols = fastlap.solve_lap(cost, algorithm="lapjv", maximize=True)
#[pyfunction]
#[pyo3(signature = (cost_matrix, algorithm, maximize=false))]
fn solve_lap<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    algorithm: &str,
    maximize: bool,
) -> PyResult<LapSolution> {
    // True sparse path: for "lapmod" on a scipy CSR matrix, solve directly
    // on the sparse adjacency instead of densifying to nrows*ncols first.
    if algorithm == "lapmod" && is_csr(cost_matrix) {
        let sparse = extract_sparse_adjacency(cost_matrix)?;
        let target = if maximize {
            sparse.negate()
        } else {
            sparse.clone()
        };
        let (_, row_assign, col_assign) = crate::lap::lapmod::solve_sparse(&target);
        let total_cost = sparse.cost_of(&row_assign);
        return Ok((total_cost, row_assign, col_assign));
    }

    let matrix = extract_matrix(cost_matrix)?;
    let solve_matrix = if maximize {
        negate_matrix(&matrix)
    } else {
        matrix.clone()
    };
    let (_, row_assign, col_assign) = solve_with(solve_matrix, algorithm)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let total_cost = recompute_cost(&matrix, &row_assign);
    Ok((total_cost, row_assign, col_assign))
}

/// Solve multiple independent Linear Assignment Problems in parallel.
///
/// Parameters
/// ----------
/// cost_matrices : list of numpy.ndarray or scipy.sparse.csr_matrix
///     A list of cost matrices to solve. With ``algorithm="lapmod"``, CSR
///     matrices are solved directly on their sparse structure (never
///     densified); with any other algorithm they are densified first, exactly
///     as in :func:`solve_lap`.
/// algorithm : str
///     Algorithm name (same as :func:`solve_lap`).
/// maximize : bool, optional
///     If ``True``, find the maximum-weight assignment for every matrix
///     instead of the minimum-cost one. Defaults to ``False``.
///
/// Returns
/// -------
/// list of tuple[float, list[int | None], list[int | None]]
///     One ``(total_cost, row_assignments, col_assignments)`` per matrix.
#[pyfunction]
#[pyo3(signature = (cost_matrices, algorithm, maximize=false))]
fn solve_lap_batch<'py>(
    py: Python<'py>,
    cost_matrices: &Bound<'py, PyAny>,
    algorithm: &str,
    maximize: bool,
) -> PyResult<Vec<LapSolution>> {
    // Validate algorithm name once up-front.
    if !supported_algorithms().contains(&algorithm) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown algorithm '{}'. Supported: {}",
            algorithm,
            supported_algorithms().join(", ")
        )));
    }

    // Extract everything up-front (while holding the GIL): sparse CSR inputs
    // are kept sparse when lapmod can consume them directly, so a batch of
    // large mostly-empty matrices never gets densified.
    let sparse_path = algorithm == "lapmod";
    let items: Vec<BatchMatrix> = cost_matrices
        .extract::<Vec<Bound<'py, PyAny>>>()?
        .iter()
        .map(|m| {
            if sparse_path && is_csr(m) {
                Ok(BatchMatrix::Sparse(extract_sparse_adjacency(m)?))
            } else {
                Ok(BatchMatrix::Dense(extract_matrix(m)?))
            }
        })
        .collect::<PyResult<_>>()?;

    let results: Vec<LapSolution> = py.allow_threads(|| {
        items
            .into_par_iter()
            .map(|item| match item {
                BatchMatrix::Sparse(sc) => {
                    let target = if maximize { sc.negate() } else { sc.clone() };
                    let (_, row_assign, col_assign) = crate::lap::lapmod::solve_sparse(&target);
                    let total_cost = sc.cost_of(&row_assign);
                    (total_cost, row_assign, col_assign)
                }
                BatchMatrix::Dense(matrix) => {
                    let solve_matrix = if maximize {
                        negate_matrix(&matrix)
                    } else {
                        matrix.clone()
                    };
                    let (_, row_assign, col_assign) = solve_with(solve_matrix, algorithm).unwrap();
                    let total_cost = recompute_cost(&matrix, &row_assign);
                    (total_cost, row_assign, col_assign)
                }
            })
            .collect()
    });

    Ok(results)
}

/// Solve a Linear Assignment Problem with optional per-entry weights.
///
/// The effective cost is ``weight[i][j] * cost_matrix[i][j]``.
///
/// Parameters
/// ----------
/// cost_matrix : numpy.ndarray or scipy.sparse.csr_matrix
///     An (n x m) cost matrix.
/// weights : numpy.ndarray or scipy.sparse.csr_matrix
///     Per-entry weights of the same shape as *cost_matrix*.
/// algorithm : str
///     Algorithm name (same as :func:`solve_lap`).
/// maximize : bool, optional
///     If ``True``, find the maximum-weight assignment instead of the
///     minimum-cost one. Defaults to ``False``.
///
/// Returns
/// -------
/// tuple[float, list[int | None], list[int | None]]
///     ``(total_cost, row_assignments, col_assignments)`` where the
///     returned cost is the sum of the *original* (unweighted) costs.
#[pyfunction]
#[pyo3(signature = (cost_matrix, weights, algorithm, maximize=false))]
fn solve_lap_weighted<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    algorithm: &str,
    maximize: bool,
) -> PyResult<LapSolution> {
    let costs = extract_matrix(cost_matrix)?;
    let w = extract_matrix(weights)?;

    if costs.len() != w.len() || (costs.is_empty() && w.is_empty()) || costs[0].len() != w[0].len()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cost_matrix and weights must have the same shape",
        ));
    }

    let weighted: Vec<Vec<f64>> = costs
        .iter()
        .zip(w.iter())
        .map(|(row_c, row_w)| {
            row_c
                .iter()
                .zip(row_w.iter())
                .map(|(c, ww)| c * ww)
                .collect()
        })
        .collect();

    // Individual entries of `costs` and `weights` are finite (validated above),
    // but their product can still overflow to +/-inf (or NaN for 0 * inf-style
    // mixes). Feeding that into the solvers hangs the shortest-augmenting-path
    // loops, so reject it here with the same error the other entry points raise.
    let weighted = validate_matrix(weighted)?;

    let solve_matrix = if maximize {
        negate_matrix(&weighted)
    } else {
        weighted
    };
    let (_, row_assign, col_assign) = solve_with(solve_matrix, algorithm)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    // Recompute cost from original unweighted matrix
    let total_cost = recompute_cost(&costs, &row_assign);

    Ok((total_cost, row_assign, col_assign))
}

/// Return the list of supported algorithm names.
#[pyfunction]
fn get_supported_algorithms() -> Vec<&'static str> {
    supported_algorithms().to_vec()
}

/// High-performance LAP solver backed by Rust.
///
/// Provides:
/// - :func:`solve_lap` — solve a single assignment problem
/// - :func:`solve_lap_batch` — solve many in parallel
/// - :func:`solve_lap_weighted` — solve with per-entry cost scaling
/// - :func:`get_supported_algorithms` — list available algorithms
#[pymodule]
fn fastlap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_lap, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lap_batch, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lap_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(get_supported_algorithms, m)?)?;
    Ok(())
}

#![allow(clippy::needless_range_loop)]
//! fastlap — High-performance LAP solver powered by Rust.
//!
//! Provides `solve_lap` for single matrices and `solve_lap_batch` for parallel
//! solving of many independent matrices. All seven algorithms (LAPJV, Hungarian,
//! LAPMOD, Dantzig, Auction, Subgradient, Sinkhorn) plus Linear Bottleneck
//! Assignment (LBAP) are exposed through a uniform API, alongside drop-in
//! compatibility layers for SciPy and lap/lapx.

use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub mod lap;
pub mod matrix;
pub mod types;
pub mod utils;

use crate::matrix::{extract_matrix, extract_sparse_adjacency, is_csr, validate_matrix};
use crate::types::{LapSolution, SparseCost};
use crate::utils::{
    apply_cost_limit_dense, apply_cost_limit_sparse, negate_matrix, solve_with,
    supported_algorithms,
};

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
/// cost_matrix : numpy.ndarray or scipy.sparse.csr_matrix or nested list
///     An (n x m) cost matrix. Square matrices are solved directly.
///     Rectangular matrices are padded internally and assignments to
///     padded rows/columns are reported as ``None``.
/// algorithm : str, optional
///     One of: ``"lapjv"``, ``"hungarian"``, ``"lapmod"``,
///     ``"dantzig"``, ``"auction"``, ``"subgradient"``, ``"sinkhorn"``.
///     Defaults to ``"lapjv"``.
/// maximize : bool, optional
///     If ``True``, find the maximum-weight assignment instead of the
///     minimum-cost one. Defaults to ``False``.
/// cost_limit : float, optional
///     Gating threshold. Assignments with cost > cost_limit (or < cost_limit
///     when maximize=True) are rejected and set to None. Defaults to None.
///
/// Returns
/// -------
/// tuple[float, list[int | None], list[int | None]]
///     ``(total_cost, row_assignments, col_assignments)``.
///     ``row_assign[i]`` is the column assigned to row i, or ``None``.
///     ``col_assign[j]`` is the row assigned to column j, or ``None``.
#[pyfunction]
#[pyo3(signature = (cost_matrix, algorithm="lapjv", maximize=false, cost_limit=None))]
fn solve_lap<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    algorithm: &str,
    maximize: bool,
    cost_limit: Option<f64>,
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
        return Ok(apply_cost_limit_sparse(
            &sparse, row_assign, col_assign, cost_limit, maximize,
        ));
    }

    let matrix = extract_matrix(cost_matrix)?;
    let solve_matrix = if maximize {
        negate_matrix(&matrix)
    } else {
        matrix.clone()
    };
    let (_, row_assign, col_assign) = solve_with(solve_matrix, algorithm)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(apply_cost_limit_dense(
        &matrix, row_assign, col_assign, cost_limit, maximize,
    ))
}

/// Solve multiple independent Linear Assignment Problems in parallel.
///
/// Parameters
/// ----------
/// cost_matrices : list of numpy.ndarray or scipy.sparse.csr_matrix
///     A list of cost matrices to solve.
/// algorithm : str, optional
///     Algorithm name (same as :func:`solve_lap`). Defaults to ``"lapjv"``.
/// maximize : bool, optional
///     If ``True``, find the maximum-weight assignment for every matrix.
/// cost_limit : float, optional
///     Gating threshold per assignment.
///
/// Returns
/// -------
/// list of tuple[float, list[int | None], list[int | None]]
///     One ``(total_cost, row_assignments, col_assignments)`` per matrix.
#[pyfunction]
#[pyo3(signature = (cost_matrices, algorithm="lapjv", maximize=false, cost_limit=None))]
fn solve_lap_batch<'py>(
    py: Python<'py>,
    cost_matrices: &Bound<'py, PyAny>,
    algorithm: &str,
    maximize: bool,
    cost_limit: Option<f64>,
) -> PyResult<Vec<LapSolution>> {
    // Validate algorithm name once up-front.
    if !supported_algorithms().contains(&algorithm) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown algorithm '{}'. Supported: {}",
            algorithm,
            supported_algorithms().join(", ")
        )));
    }

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
                    apply_cost_limit_sparse(&sc, row_assign, col_assign, cost_limit, maximize)
                }
                BatchMatrix::Dense(matrix) => {
                    let solve_matrix = if maximize {
                        negate_matrix(&matrix)
                    } else {
                        matrix.clone()
                    };
                    let (_, row_assign, col_assign) = solve_with(solve_matrix, algorithm).unwrap();
                    apply_cost_limit_dense(&matrix, row_assign, col_assign, cost_limit, maximize)
                }
            })
            .collect()
    });

    Ok(results)
}

/// Solve a Linear Assignment Problem with optional per-entry weights.
///
/// The effective optimization cost is ``weight[i][j] * cost_matrix[i][j]``,
/// while the returned total cost is calculated from the unweighted matrix.
#[pyfunction]
#[pyo3(signature = (cost_matrix, weights, algorithm="lapjv", maximize=false, cost_limit=None))]
fn solve_lap_weighted<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    algorithm: &str,
    maximize: bool,
    cost_limit: Option<f64>,
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

    let weighted = validate_matrix(weighted)?;

    let solve_matrix = if maximize {
        negate_matrix(&weighted)
    } else {
        weighted
    };
    let (_, row_assign, col_assign) = solve_with(solve_matrix, algorithm)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    Ok(apply_cost_limit_dense(
        &costs, row_assign, col_assign, cost_limit, maximize,
    ))
}

/// Solve the Linear Bottleneck Assignment Problem (LBAP).
///
/// Finds a matching that minimizes the maximum cost assigned:
/// ``min_pi max_i C[i, pi(i)]``.
#[pyfunction]
#[pyo3(signature = (cost_matrix, maximize=false, cost_limit=None))]
fn solve_lbap<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    maximize: bool,
    cost_limit: Option<f64>,
) -> PyResult<LapSolution> {
    let matrix = extract_matrix(cost_matrix)?;
    let solve_matrix = if maximize {
        negate_matrix(&matrix)
    } else {
        matrix.clone()
    };
    let (mut b_cost, r_assign, c_assign) = crate::lap::bottleneck::solve(solve_matrix);
    if maximize {
        b_cost = -b_cost;
    }
    let (_, r_assign, c_assign) =
        apply_cost_limit_dense(&matrix, r_assign, c_assign, cost_limit, maximize);

    let final_b_cost = if cost_limit.is_some() {
        let mut best = if maximize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        for i in 0..matrix.len() {
            if let Some(j) = r_assign[i] {
                if j < matrix[0].len() {
                    let c = matrix[i][j];
                    best = if maximize { best.min(c) } else { best.max(c) };
                }
            }
        }
        if best.is_finite() {
            best
        } else {
            0.0
        }
    } else {
        b_cost
    };

    Ok((final_b_cost, r_assign, c_assign))
}

/// Solve multiple independent Linear Bottleneck Assignment Problems in parallel.
#[pyfunction]
#[pyo3(signature = (cost_matrices, maximize=false, cost_limit=None))]
fn solve_lbap_batch<'py>(
    py: Python<'py>,
    cost_matrices: &Bound<'py, PyAny>,
    maximize: bool,
    cost_limit: Option<f64>,
) -> PyResult<Vec<LapSolution>> {
    let items: Vec<Vec<Vec<f64>>> = cost_matrices
        .extract::<Vec<Bound<'py, PyAny>>>()?
        .iter()
        .map(|m| extract_matrix(m))
        .collect::<PyResult<_>>()?;

    let results: Vec<LapSolution> = py.allow_threads(|| {
        items
            .into_par_iter()
            .map(|matrix| {
                let solve_matrix = if maximize {
                    negate_matrix(&matrix)
                } else {
                    matrix.clone()
                };
                let (mut b_cost, r_assign, c_assign) = crate::lap::bottleneck::solve(solve_matrix);
                if maximize {
                    b_cost = -b_cost;
                }
                let (_, r_assign, c_assign) =
                    apply_cost_limit_dense(&matrix, r_assign, c_assign, cost_limit, maximize);

                let final_b_cost = if cost_limit.is_some() {
                    let mut best = if maximize {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    };
                    for i in 0..matrix.len() {
                        if let Some(j) = r_assign[i] {
                            if j < matrix[0].len() {
                                let c = matrix[i][j];
                                best = if maximize { best.min(c) } else { best.max(c) };
                            }
                        }
                    }
                    if best.is_finite() {
                        best
                    } else {
                        0.0
                    }
                } else {
                    b_cost
                };

                (final_b_cost, r_assign, c_assign)
            })
            .collect()
    });

    Ok(results)
}

type PyIndexArrays<'py> = (
    Bound<'py, numpy::PyArray1<i64>>,
    Bound<'py, numpy::PyArray1<i64>>,
);

/// Drop-in replacement for ``scipy.optimize.linear_sum_assignment``.
///
/// Returns
/// -------
/// tuple[numpy.ndarray, numpy.ndarray]
///     ``(row_indices, col_indices)`` matching SciPy's format.
#[pyfunction]
#[pyo3(signature = (cost_matrix, maximize=false))]
fn linear_sum_assignment<'py>(
    py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    maximize: bool,
) -> PyResult<PyIndexArrays<'py>> {
    let (_, row_assign, _) = solve_lap(py, cost_matrix, "lapjv", maximize, None)?;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for (i, opt_j) in row_assign.iter().enumerate() {
        if let Some(j) = opt_j {
            rows.push(i as i64);
            cols.push(*j as i64);
        }
    }
    let py_rows = numpy::PyArray1::from_vec(py, rows);
    let py_cols = numpy::PyArray1::from_vec(py, cols);
    Ok((py_rows, py_cols))
}

/// Drop-in replacement for ``lap.lapjv`` / ``lapx.lapjv``.
///
/// Parameters
/// ----------
/// cost : numpy.ndarray or array-like
///     Cost matrix.
/// extend_cost : bool, optional
///     Accepted for compatibility with ``lap.lapjv`` (handled automatically).
/// cost_limit : float, optional
///     Maximum allowed cost for valid assignments. Unassigned pairs return -1.
/// return_cost : bool, optional
///     Whether to return the optimal total cost. Defaults to True.
///
/// Returns
/// -------
/// tuple[float, numpy.ndarray, numpy.ndarray] or tuple[numpy.ndarray, numpy.ndarray]
///     ``(opt_cost, x, y)`` where x and y are int32 arrays (-1 for unassigned).
#[pyfunction]
#[pyo3(signature = (cost, _extend_cost=true, cost_limit=None, return_cost=true))]
fn lapjv<'py>(
    py: Python<'py>,
    cost: &Bound<'py, PyAny>,
    _extend_cost: bool,
    cost_limit: Option<f64>,
    return_cost: bool,
) -> PyResult<PyObject> {
    let (opt_cost, row_assign, col_assign) = solve_lap(py, cost, "lapjv", false, cost_limit)?;
    let x: Vec<i32> = row_assign
        .into_iter()
        .map(|opt_j| opt_j.map(|j| j as i32).unwrap_or(-1))
        .collect();
    let y: Vec<i32> = col_assign
        .into_iter()
        .map(|opt_i| opt_i.map(|i| i as i32).unwrap_or(-1))
        .collect();

    let py_x = numpy::PyArray1::from_vec(py, x);
    let py_y = numpy::PyArray1::from_vec(py, y);

    if return_cost {
        Ok(pyo3::IntoPyObjectExt::into_py_any(
            (opt_cost, py_x, py_y),
            py,
        )?)
    } else {
        Ok(pyo3::IntoPyObjectExt::into_py_any((py_x, py_y), py)?)
    }
}

/// Find the K-best (ranked) assignments using Murty's algorithm.
///
/// Parameters
/// ----------
/// cost_matrix : numpy.ndarray or array-like
///     An (n x m) cost matrix.
/// k : int, optional
///     Number of top ranked assignments to return. Defaults to 3.
/// maximize : bool, optional
///     If True, return assignments with highest profit. Defaults to False.
/// cost_limit : float, optional
///     Gating threshold per assignment.
///
/// Returns
/// -------
/// list of tuple[float, list[int | None], list[int | None]]
///     List of up to k solutions in increasing order of cost (or decreasing order of profit).
#[pyfunction]
#[pyo3(signature = (cost_matrix, k=3, maximize=false, cost_limit=None))]
fn solve_lap_kbest<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    k: usize,
    maximize: bool,
    cost_limit: Option<f64>,
) -> PyResult<Vec<LapSolution>> {
    let matrix = extract_matrix(cost_matrix)?;
    let solve_matrix = if maximize {
        negate_matrix(&matrix)
    } else {
        matrix.clone()
    };

    let solutions = crate::lap::murty::solve_kbest(solve_matrix, k);

    let filtered: Vec<LapSolution> = solutions
        .into_iter()
        .map(|(_, row_assign, col_assign)| {
            apply_cost_limit_dense(&matrix, row_assign, col_assign, cost_limit, maximize)
        })
        .collect();

    Ok(filtered)
}

/// Return the list of supported algorithm names.
#[pyfunction]
fn get_supported_algorithms() -> Vec<&'static str> {
    supported_algorithms().to_vec()
}

/// High-performance LAP solver backed by Rust.
#[pymodule]
fn fastlap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    // Top-level functions
    m.add_function(wrap_pyfunction!(solve_lap, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lap_batch, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lap_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lbap, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lbap_batch, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lap_kbest, m)?)?;
    m.add_function(wrap_pyfunction!(linear_sum_assignment, m)?)?;
    m.add_function(wrap_pyfunction!(lapjv, m)?)?;
    m.add_function(wrap_pyfunction!(get_supported_algorithms, m)?)?;

    // fastlap.lap submodule
    let lap_mod = PyModule::new(py, "lap")?;
    lap_mod.add_function(wrap_pyfunction!(lapjv, &lap_mod)?)?;
    m.add_submodule(&lap_mod)?;

    // fastlap.compat submodule
    let compat_mod = PyModule::new(py, "compat")?;
    compat_mod.add_function(wrap_pyfunction!(linear_sum_assignment, &compat_mod)?)?;
    m.add_submodule(&compat_mod)?;

    Ok(())
}

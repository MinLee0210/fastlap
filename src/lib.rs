#![allow(clippy::needless_range_loop)]
//! fastlap — High-performance LAP solver powered by Rust.
//!
//! Provides `solve_lap` for single matrices and `solve_lap_batch` for parallel
//! solving of many independent matrices (3D ndarray batches supported, with
//! optional thread-count control). Eleven algorithms (LAPJV, Hungarian, LAPMOD,
//! LAPJVsp, Subgradient, Auction, Dantzig, Sinkhorn, SSP, Cost-Scaling, Greedy)
//! plus Linear Bottleneck Assignment (LBAP), Murty k-best, optimal dual
//! extraction and cost-limit gating are exposed through a uniform API,
//! alongside drop-in compatibility layers for SciPy and lap/lapx.

use numpy::PyArrayMethods;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub mod lap;
pub mod matrix;
pub mod types;
pub mod utils;

use crate::matrix::{extract_matrix, extract_sparse_adjacency, is_csr, validate_matrix};
use crate::types::{LapSolution, SparseCost};
use crate::utils::{
    apply_cost_limit_dense, apply_cost_limit_sparse, dual_supported_algorithms, negate_matrix,
    sap_solve_duals_matrix, solve_with, supported_algorithms,
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

/// Extract one element of a batch: with `sparse_path` set, scipy CSR inputs
/// stay sparse; everything else is extracted/validated as a dense matrix.
fn extract_batch_entry<'py>(m: &Bound<'py, PyAny>, sparse_path: bool) -> PyResult<BatchMatrix> {
    if sparse_path && is_csr(m) {
        Ok(BatchMatrix::Sparse(extract_sparse_adjacency(m)?))
    } else {
        Ok(BatchMatrix::Dense(extract_matrix(m)?))
    }
}

/// Solve a single dense batch entry (used by the Rayon pool closure).
fn solve_batch_dense(
    matrix: Vec<Vec<f64>>,
    algorithm: &str,
    maximize: bool,
    cost_limit: Option<f64>,
) -> LapSolution {
    let solve_matrix = if maximize {
        negate_matrix(&matrix)
    } else {
        matrix.clone()
    };
    let (_, row_assign, col_assign) = solve_with(solve_matrix, algorithm).unwrap();
    apply_cost_limit_dense(&matrix, row_assign, col_assign, cost_limit, maximize)
}

/// Solve a single sparse batch entry via a sparse-aware solver.
fn solve_batch_sparse(
    sc: SparseCost,
    algorithm: &str,
    maximize: bool,
    cost_limit: Option<f64>,
) -> LapSolution {
    let target = if maximize { sc.negate() } else { sc.clone() };
    let (_, row_assign, col_assign) = if algorithm == "lapjvsp" {
        crate::lap::lapjvsp::solve_sparse(&target)
    } else {
        crate::lap::lapmod::solve_sparse(&target)
    };
    apply_cost_limit_sparse(&sc, row_assign, col_assign, cost_limit, maximize)
}

/// Solve a single dense Linear Bottleneck Assignment (used by the Rayon pool).
fn solve_lbap_entry(matrix: Vec<Vec<f64>>, maximize: bool, cost_limit: Option<f64>) -> LapSolution {
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
}

/// Build a Rayon pool of the requested size (None = global pool). Callers
/// must construct this while holding the GIL, then run the parallel work
/// inside `py.allow_threads` so other Python threads are not blocked.
fn build_thread_pool(n_threads: Option<usize>) -> PyResult<Option<rayon::ThreadPool>> {
    match n_threads {
        None => Ok(None),
        Some(0) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_threads must be >= 1",
        )),
        Some(t) => rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .map(Some)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "failed to create thread pool: {e}"
                ))
            }),
    }
}

/// Run a parallel closure on the given pool (or Rayon's global pool), used
/// from inside `py.allow_threads`.
fn run_in_pool<T, F>(pool: &Option<rayon::ThreadPool>, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
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
    // True sparse path: for the sparse-aware algorithms on a scipy CSR matrix,
    // solve directly on the sparse adjacency instead of densifying.
    if is_csr(cost_matrix) && (algorithm == "lapmod" || algorithm == "lapjvsp") {
        let sparse = extract_sparse_adjacency(cost_matrix)?;
        let target = if maximize {
            sparse.negate()
        } else {
            sparse.clone()
        };
        let (_, row_assign, col_assign) = if algorithm == "lapjvsp" {
            crate::lap::lapjvsp::solve_sparse(&target)
        } else {
            crate::lap::lapmod::solve_sparse(&target)
        };
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
/// cost_matrices : numpy.ndarray of shape (B, N, M) or list of numpy.ndarray / scipy.sparse.csr_matrix
///     A batch of cost matrices to solve. A 3D ndarray is treated as B stacked
///     N×M matrices; with ``algorithm="lapmod"``, CSR matrices are solved
///     directly on their sparse structure (never densified).
/// algorithm : str, optional
///     Algorithm name (same as :func:`solve_lap`). Defaults to ``"lapjv"``.
/// maximize : bool, optional
///     If ``True``, find the maximum-weight assignment for every matrix.
/// cost_limit : float, optional
///     Gating threshold per assignment.
/// n_threads : int, optional
///     Number of worker threads to use. Defaults to ``None`` (all cores).
///
/// Returns
/// -------
/// list of tuple[float, list[int | None], list[int | None]]
///     One ``(total_cost, row_assignments, col_assignments)`` per matrix.
#[pyfunction]
#[pyo3(signature = (cost_matrices, algorithm="lapjv", maximize=false, cost_limit=None, n_threads=None))]
fn solve_lap_batch<'py>(
    py: Python<'py>,
    cost_matrices: &Bound<'py, PyAny>,
    algorithm: &str,
    maximize: bool,
    cost_limit: Option<f64>,
    n_threads: Option<usize>,
) -> PyResult<Vec<LapSolution>> {
    // Validate algorithm name once up-front.
    if !supported_algorithms().contains(&algorithm) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown algorithm '{}'. Supported: {}",
            algorithm,
            supported_algorithms().join(", ")
        )));
    }

    let sparse_path = algorithm == "lapmod" || algorithm == "lapjvsp";

    // Fast path: a 3D float64 ndarray is a stack of dense matrices. Slicing
    // each (N, M) plane directly avoids per-plane Python object overhead.
    if let Ok(stack) = cost_matrices.downcast::<numpy::PyArray3<f64>>() {
        use numpy::ndarray::Axis;
        let ro = stack.readonly();
        let view = ro.as_array();
        let nb = view.shape()[0];
        let nrows = view.shape()[1];
        let ncols = view.shape()[2];
        let mut matrices = Vec::with_capacity(nb);
        for b in 0..nb {
            let plane = view.index_axis(Axis(0), b);
            let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(nrows);
            for i in 0..nrows {
                let mut row: Vec<f64> = Vec::with_capacity(ncols);
                for j in 0..ncols {
                    row.push(plane[[i, j]]);
                }
                matrix.push(row);
            }
            matrices.push(validate_matrix(matrix)?);
        }
        let pool = build_thread_pool(n_threads)?;
        let results = py.allow_threads(|| {
            run_in_pool(&pool, || {
                matrices
                    .into_par_iter()
                    .map(|matrix| solve_batch_dense(matrix, algorithm, maximize, cost_limit))
                    .collect()
            })
        });
        return Ok(results);
    }

    // Extract everything up-front (while holding the GIL): sparse CSR inputs
    // are kept sparse when lapmod can consume them directly, so a batch of
    // large mostly-empty matrices never gets densified.
    let items: Vec<BatchMatrix> = cost_matrices
        .extract::<Vec<Bound<'py, PyAny>>>()?
        .iter()
        .map(|m| extract_batch_entry(m, sparse_path))
        .collect::<PyResult<_>>()?;

    let pool = build_thread_pool(n_threads)?;
    let results = py.allow_threads(|| {
        run_in_pool(&pool, || {
            items
                .into_par_iter()
                .map(|item| match item {
                    BatchMatrix::Sparse(sc) => {
                        solve_batch_sparse(sc, algorithm, maximize, cost_limit)
                    }
                    BatchMatrix::Dense(matrix) => {
                        solve_batch_dense(matrix, algorithm, maximize, cost_limit)
                    }
                })
                .collect()
        })
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
#[pyo3(signature = (cost_matrices, maximize=false, cost_limit=None, n_threads=None))]
fn solve_lbap_batch<'py>(
    py: Python<'py>,
    cost_matrices: &Bound<'py, PyAny>,
    maximize: bool,
    cost_limit: Option<f64>,
    n_threads: Option<usize>,
) -> PyResult<Vec<LapSolution>> {
    // Fast path: a 3D float64 ndarray is a stack of dense matrices.
    if let Ok(stack) = cost_matrices.downcast::<numpy::PyArray3<f64>>() {
        use numpy::ndarray::Axis;
        let ro = stack.readonly();
        let view = ro.as_array();
        let nb = view.shape()[0];
        let nrows = view.shape()[1];
        let ncols = view.shape()[2];
        let mut matrices = Vec::with_capacity(nb);
        for b in 0..nb {
            let plane = view.index_axis(Axis(0), b);
            let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(nrows);
            for i in 0..nrows {
                let mut row: Vec<f64> = Vec::with_capacity(ncols);
                for j in 0..ncols {
                    row.push(plane[[i, j]]);
                }
                matrix.push(row);
            }
            matrices.push(validate_matrix(matrix)?);
        }
        let pool = build_thread_pool(n_threads)?;
        let results = py.allow_threads(|| {
            run_in_pool(&pool, || {
                matrices
                    .into_par_iter()
                    .map(|matrix| solve_lbap_entry(matrix, maximize, cost_limit))
                    .collect()
            })
        });
        return Ok(results);
    }

    let items: Vec<Vec<Vec<f64>>> = cost_matrices
        .extract::<Vec<Bound<'py, PyAny>>>()?
        .iter()
        .map(|m| extract_matrix(m))
        .collect::<PyResult<_>>()?;

    let pool = build_thread_pool(n_threads)?;
    let results = py.allow_threads(|| {
        run_in_pool(&pool, || {
            items
                .into_par_iter()
                .map(|matrix| solve_lbap_entry(matrix, maximize, cost_limit))
                .collect()
        })
    });

    Ok(results)
}

type PyIndexArrays<'py> = (
    Bound<'py, numpy::PyArray1<i64>>,
    Bound<'py, numpy::PyArray1<i64>>,
);

/// Turn a row assignment into two aligned int64 index arrays (SciPy/lapx
/// `lapjvx` style): `rows[k]` paired with `cols[k]`, sorted by row index.
fn aligned_index_arrays<'py>(
    py: Python<'py>,
    row_assign: &[Option<usize>],
) -> PyResult<PyIndexArrays<'py>> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for (i, opt_j) in row_assign.iter().enumerate() {
        if let Some(j) = opt_j {
            rows.push(i as i64);
            cols.push(*j as i64);
        }
    }
    Ok((
        numpy::PyArray1::from_vec(py, rows),
        numpy::PyArray1::from_vec(py, cols),
    ))
}

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
    aligned_index_arrays(py, &row_assign)
}

/// lapx-style ``lapjvx``: SciPy-compatible aligned index output from the
/// LAPJV solver, with an optional returned cost and cost gating.
///
/// Parameters
/// ----------
/// cost_matrix : numpy.ndarray or array-like
///     Cost matrix.
/// maximize : bool, optional
///     If True, solve a maximum-weight assignment. Defaults to False.
/// cost_limit : float, optional
///     Gating threshold; assignments beyond it are dropped from the output.
/// return_cost : bool, optional
///     Whether to return the optimal cost. Defaults to True.
///
/// Returns
/// -------
/// tuple[float, numpy.ndarray, numpy.ndarray] or tuple[numpy.ndarray, numpy.ndarray]
///     ``(cost, row_indices, col_indices)`` (or the two arrays without cost).
#[pyfunction]
#[pyo3(signature = (cost_matrix, maximize=false, cost_limit=None, return_cost=true))]
fn lapjvx<'py>(
    py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    maximize: bool,
    cost_limit: Option<f64>,
    return_cost: bool,
) -> PyResult<PyObject> {
    let (opt_cost, row_assign, _) = solve_lap(py, cost_matrix, "lapjv", maximize, cost_limit)?;
    let (py_rows, py_cols) = aligned_index_arrays(py, &row_assign)?;
    if return_cost {
        Ok(pyo3::IntoPyObjectExt::into_py_any(
            (opt_cost, py_rows, py_cols),
            py,
        )?)
    } else {
        Ok(pyo3::IntoPyObjectExt::into_py_any((py_rows, py_cols), py)?)
    }
}

/// lapx-style ``lapjvxa``: return the assignment directly as an (K, 2) array of
/// ``[row, col]`` pairs, with an optional returned cost.
///
/// Parameters
/// ----------
/// cost_matrix : numpy.ndarray or array-like
///     Cost matrix.
/// maximize : bool, optional
///     If True, solve a maximum-weight assignment. Defaults to False.
/// cost_limit : float, optional
///     Gating threshold; assignments beyond it are dropped from the output.
/// return_cost : bool, optional
///     Whether to return the optimal cost. Defaults to True.
///
/// Returns
/// -------
/// tuple[float, numpy.ndarray] or numpy.ndarray
///     ``(cost, pairs)`` where pairs has shape (K, 2), or pairs alone.
#[pyfunction]
#[pyo3(signature = (cost_matrix, maximize=false, cost_limit=None, return_cost=true))]
fn assignment_pairs<'py>(
    py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    maximize: bool,
    cost_limit: Option<f64>,
    return_cost: bool,
) -> PyResult<PyObject> {
    let (opt_cost, row_assign, _) = solve_lap(py, cost_matrix, "lapjv", maximize, cost_limit)?;
    let pairs: Vec<Vec<i64>> = row_assign
        .iter()
        .enumerate()
        .filter_map(|(i, opt_j)| opt_j.map(|j| vec![i as i64, j as i64]))
        .collect();
    let py_pairs = numpy::PyArray2::from_vec2(py, &pairs)?;
    if return_cost {
        Ok(pyo3::IntoPyObjectExt::into_py_any(
            (opt_cost, py_pairs),
            py,
        )?)
    } else {
        Ok(pyo3::IntoPyObjectExt::into_py_any(py_pairs, py)?)
    }
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

/// Solve a Linear Assignment Problem and return the optimal dual variables.
///
/// In addition to the usual ``(total_cost, row_assignments, col_assignments)``
/// this returns the optimal *dual* potentials ``u`` (one per row) and ``v``
/// (one per column) of the minimum-cost assignment LP. They are feasible —
/// ``u[i] + v[j] <= cost[i][j]`` for every entry — with equality on every
/// matched pair, and ``total_cost == sum(u) + sum(v)`` (strong duality).
/// Economically these are the shadow prices of the row/column resources.
///
/// Only the exact dual-convergent algorithms are supported (see
/// [`dual_supported_algorithms`]): ``"lapjv"`` (default), ``"subgradient"``,
/// ``"sinkhorn"``, ``"dantzig"``. Maximization is not supported because the
/// duals are only meaningfully defined for the minimum-cost form.
///
/// Parameters
/// ----------
/// cost_matrix : numpy.ndarray or scipy.sparse.csr_matrix or nested list
///     An (n x m) cost matrix.
/// algorithm : str, optional
///     One of the exact dual-convergent algorithms. Defaults to ``"lapjv"``.
///
/// Returns
/// -------
/// tuple[float, list[int | None], list[int | None], list[float], list[float]]
///     ``(total_cost, row_assignments, col_assignments, u, v)``.
#[pyfunction]
#[pyo3(signature = (cost_matrix, algorithm="lapjv"))]
fn solve_lap_duals<'py>(
    _py: Python<'py>,
    cost_matrix: &Bound<'py, PyAny>,
    algorithm: &str,
) -> PyResult<crate::types::LapSolutionWithDuals> {
    if !dual_supported_algorithms().contains(&algorithm) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Algorithm '{}' does not support dual extraction. Supported: {}",
            algorithm,
            dual_supported_algorithms().join(", ")
        )));
    }
    let matrix = extract_matrix(cost_matrix)?;
    let ((total_cost, row_assign, col_assign), u, v) = sap_solve_duals_matrix(&matrix);
    Ok((total_cost, row_assign, col_assign, u, v))
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
    m.add_function(wrap_pyfunction!(solve_lap_duals, m)?)?;
    m.add_function(wrap_pyfunction!(linear_sum_assignment, m)?)?;
    m.add_function(wrap_pyfunction!(lapjv, m)?)?;
    m.add_function(wrap_pyfunction!(lapjvx, m)?)?;
    m.add_function(wrap_pyfunction!(assignment_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(get_supported_algorithms, m)?)?;

    // fastlap.lap submodule
    let lap_mod = PyModule::new(py, "lap")?;
    lap_mod.add_function(wrap_pyfunction!(lapjv, &lap_mod)?)?;
    m.add_submodule(&lap_mod)?;

    // fastlap.compat submodule
    let compat_mod = PyModule::new(py, "compat")?;
    compat_mod.add_function(wrap_pyfunction!(linear_sum_assignment, &compat_mod)?)?;
    compat_mod.add_function(wrap_pyfunction!(lapjvx, &compat_mod)?)?;
    compat_mod.add_function(wrap_pyfunction!(assignment_pairs, &compat_mod)?)?;
    m.add_submodule(&compat_mod)?;

    Ok(())
}

use numpy::PyReadonlyArray1;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use sprs::CsMat;
use std::sync::OnceLock;

use crate::types::SparseCost;

/// Cached handle to the `numpy` module (import is cheap after the first call,
/// and caching avoids touching Python's import machinery on every matrix).
static NUMPY_MOD: OnceLock<Py<PyModule>> = OnceLock::new();

/// True if `cost_matrix` exposes the scipy.sparse CSR attribute quartet.
pub fn is_csr<'py>(cost_matrix: &Bound<'py, PyAny>) -> bool {
    ["indptr", "indices", "data", "shape"]
        .iter()
        .all(|&attr| cost_matrix.hasattr(attr).unwrap_or(false))
}

/// Read a NumPy integer array of unknown width (scipy CSR `indptr`/`indices`
/// default to `int32`, not the platform `usize`) as `Vec<usize>`, normalizing
/// via `astype(int64)` first so extraction never depends on the caller's dtype.
fn read_index_array<'py>(arr: &Bound<'py, PyAny>) -> PyResult<Vec<usize>> {
    let normalized = arr.call_method1("astype", ("int64",))?;
    let readonly: PyReadonlyArray1<i64> = normalized.extract()?;
    Ok(readonly.as_slice()?.iter().map(|&x| x as usize).collect())
}

/// Convert a dense NumPy array to Vec<Vec<f64>>
pub fn extract_dense_matrix<'py>(
    cost_matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    let matrix: Vec<Vec<f64>> = cost_matrix
        .readonly()
        .as_array()
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect::<Vec<f64>>())
        .collect();
    Ok(matrix)
}

/// Convert a scipy.sparse.csr_matrix to Vec<Vec<f64>>
pub fn extract_sparse_matrix<'py>(cost_matrix: &Bound<'py, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let indptr = read_index_array(&cost_matrix.getattr("indptr")?)?;
    let indices = read_index_array(&cost_matrix.getattr("indices")?)?;
    let data: PyReadonlyArray1<f64> = cost_matrix.getattr("data")?.extract()?;

    let shape: (usize, usize) = cost_matrix.getattr("shape")?.extract::<(usize, usize)>()?;

    let csr = CsMat::new(shape, indptr, indices, data.as_slice()?.to_vec());

    let dense: Vec<Vec<f64>> = (0..shape.0)
        .map(|i| {
            (0..shape.1)
                .map(|j| csr.get(i, j).copied().unwrap_or(f64::INFINITY))
                .collect()
        })
        .collect();

    Ok(dense)
}

/// Convert input (dense or CSR) to a validated dense matrix.
///
/// Accepts numpy arrays of any numeric dtype (integer, float32, ...) as well
/// as plain Python nested lists: anything `np.asarray(x, dtype=float64)`
/// understands. A float64 ndarray is taken directly; anything else is routed
/// through numpy so the accepted surface matches what users naturally pass in
/// (the README quick start, for example, uses a plain list).
pub fn extract_matrix<'py>(cost_matrix: &Bound<'py, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    // Try dense float64 first (fast path, no conversion).
    if let Ok(array) = cost_matrix.downcast::<PyArray2<f64>>() {
        let matrix = extract_dense_matrix(array)?;
        return validate_matrix(matrix);
    }

    // Try sparse (CSR).
    if is_csr(cost_matrix) {
        let matrix = extract_sparse_matrix(cost_matrix)?;
        return validate_matrix(matrix);
    }

    // Fall back to a generic conversion through numpy, so Python lists and
    // integer / float32 arrays work too. This only fires when the input was
    // not already a float64 ndarray.
    let py = cost_matrix.py();
    let np = match NUMPY_MOD.get() {
        Some(module) => module.clone_ref(py),
        None => {
            // numpy is a hard dependency of fastlap, so an import failure here
            // is fatal regardless of the specific matrix being converted.
            let module = PyModule::import(py, "numpy")
                .expect("numpy could not be imported (fastlap requires numpy)");
            let _ = NUMPY_MOD.set(module.clone().unbind());
            module.unbind()
        }
    };
    let asarray = np.bind(py).getattr("asarray")?;
    let dtype = np.bind(py).getattr("float64")?;
    let arr = asarray.call1((cost_matrix, dtype))?;
    let type_name = cost_matrix.get_type().name()?.to_string();
    let array = arr.downcast::<PyArray2<f64>>().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Input must be convertible to a 2D float64 array, got {type_name}"
        ))
    })?;
    let matrix = extract_dense_matrix(array)?;
    validate_matrix(matrix)
}

/// Convert a scipy.sparse.csr_matrix directly to a [`SparseCost`] adjacency
/// list, without materializing the dense `nrows * ncols` matrix.
///
/// This is what lets `lapmod` actually behave like a sparse-aware solver:
/// `extract_sparse_matrix` above densifies immediately (fine for the other
/// five dense algorithms, which have no sparse code path), but for a large,
/// sparse matrix that densification is exactly the cost this function avoids.
pub fn extract_sparse_adjacency<'py>(cost_matrix: &Bound<'py, PyAny>) -> PyResult<SparseCost> {
    let indptr = read_index_array(&cost_matrix.getattr("indptr")?)?;
    let indices = read_index_array(&cost_matrix.getattr("indices")?)?;
    let data: PyReadonlyArray1<f64> = cost_matrix.getattr("data")?.extract()?;
    let shape: (usize, usize) = cost_matrix.getattr("shape")?.extract::<(usize, usize)>()?;

    if shape.0 == 0 || shape.1 == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Matrix must not be empty",
        ));
    }

    let data = data.as_slice()?;

    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); shape.0];
    for i in 0..shape.0 {
        for k in indptr[i]..indptr[i + 1] {
            let val = data[k];
            let j = indices[k];
            if val.is_nan() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Matrix contains NaN at position [{i}, {j}]"
                )));
            }
            if val.is_infinite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Matrix contains infinite value at position [{i}, {j}]"
                )));
            }
            rows[i].push((j, val));
        }
    }

    Ok(SparseCost {
        nrows: shape.0,
        ncols: shape.1,
        rows,
    })
}

/// Ensure matrix is rectangular, non-empty, and contains no NaN/Inf values.
pub fn validate_matrix(matrix: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    if matrix.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Matrix must not be empty",
        ));
    }

    let ncols = matrix[0].len();
    if ncols == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Matrix rows must not be empty",
        ));
    }

    for (i, row) in matrix.iter().enumerate() {
        if row.len() != ncols {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Matrix must be rectangular: row 0 has {} columns but row {} has {}",
                ncols,
                i,
                row.len()
            )));
        }
        for (j, &val) in row.iter().enumerate() {
            if val.is_nan() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Matrix contains NaN at position [{i}, {j}]"
                )));
            }
            if val.is_infinite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Matrix contains infinite value at position [{i}, {j}]"
                )));
            }
        }
    }

    Ok(matrix)
}

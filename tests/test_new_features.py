import pytest
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment as scipy_lsa
import lap as ref_lap
import fastlap
from fastlap import compat, lap


def test_supported_algorithms_contains_sinkhorn():
    algos = fastlap.get_supported_algorithms()
    assert "sinkhorn" in algos
    assert "lapjv" in algos


# ── Cost limit tests ────────────────────────────────────────────────────────

def test_cost_limit_basic():
    # Matrix where optimal without limit is (0->0, 1->1, 2->2) with costs 1, 5, 20
    matrix = np.array([
        [1.0, 50.0, 50.0],
        [50.0, 5.0, 50.0],
        [50.0, 50.0, 20.0],
    ])
    cost_no_limit, rows_no_limit, cols_no_limit = fastlap.solve_lap(matrix, algorithm="lapjv")
    assert rows_no_limit == [0, 1, 2]
    assert abs(cost_no_limit - 26.0) < 1e-9

    # With cost_limit = 10.0, row 2 -> col 2 (cost 20) should be gated out (None)
    cost, rows, cols = fastlap.solve_lap(matrix, algorithm="lapjv", cost_limit=10.0)
    assert rows == [0, 1, None]
    assert cols == [0, 1, None]
    assert abs(cost - 6.0) < 1e-9


def test_cost_limit_maximize():
    profit = np.array([
        [100.0, 10.0],
        [10.0, 5.0],
    ])
    # Maximize optimal is (0->0: 100, 1->1: 5). With limit=50, pair 1->1 (profit 5 < 50) is gated out.
    cost, rows, cols = fastlap.solve_lap(profit, algorithm="lapjv", maximize=True, cost_limit=50.0)
    assert rows == [0, None]
    assert cols == [0, None]
    assert abs(cost - 100.0) < 1e-9


def test_cost_limit_batch():
    matrices = [
        np.array([[1.0, 100.0], [100.0, 2.0]]),
        np.array([[1.0, 100.0], [100.0, 50.0]]),
    ]
    results = fastlap.solve_lap_batch(matrices, algorithm="lapjv", cost_limit=10.0)
    # Matrix 0: pairs (0->0: 1, 1->1: 2) -> both <= 10 -> cost 3
    assert abs(results[0][0] - 3.0) < 1e-9
    assert results[0][1] == [0, 1]
    # Matrix 1: pair 1->1 (cost 50 > 10) is gated out -> cost 1
    assert abs(results[1][0] - 1.0) < 1e-9
    assert results[1][1] == [0, None]


def test_cost_limit_sparse_lapmod():
    matrix = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 3.0],
    ])
    csr = sp.csr_matrix(matrix)
    cost, rows, cols = fastlap.solve_lap(csr, algorithm="lapmod", cost_limit=10.0)
    assert rows == [0, None, 2]
    assert cols == [0, None, 2]
    assert abs(cost - 5.0) < 1e-9


def test_cost_limit_weighted():
    costs = np.array([[2.0, 50.0], [50.0, 20.0]])
    weights = np.ones_like(costs)
    cost, rows, cols = fastlap.solve_lap_weighted(costs, weights, algorithm="lapjv", cost_limit=10.0)
    assert rows == [0, None]
    assert cols == [0, None]
    assert abs(cost - 2.0) < 1e-9


# ── Sinkhorn algorithm tests ────────────────────────────────────────────────

@pytest.mark.parametrize("size", [2, 3, 5, 8])
def test_sinkhorn_correctness(size):
    np.random.seed(size)
    matrix = np.random.uniform(1.0, 50.0, (size, size))
    cost_sinkhorn, rows_s, cols_s = fastlap.solve_lap(matrix, algorithm="sinkhorn")
    ref_rows, ref_cols = scipy_lsa(matrix)
    ref_cost = matrix[ref_rows, ref_cols].sum()

    assert abs(cost_sinkhorn - ref_cost) < 1e-6
    assert rows_s == ref_cols.tolist()


def test_sinkhorn_rectangular():
    np.random.seed(42)
    matrix = np.random.uniform(1.0, 50.0, (3, 5))
    cost, rows, cols = fastlap.solve_lap(matrix, algorithm="sinkhorn")
    ref_rows, ref_cols = scipy_lsa(matrix)
    ref_cost = matrix[ref_rows, ref_cols].sum()
    assert abs(cost - ref_cost) < 1e-6
    assert sum(1 for r in rows if r is not None) == 3


# ── Linear Bottleneck Assignment (LBAP) tests ──────────────────────────────

def test_solve_lbap_square():
    # Matrix where total sum min assignment vs bottleneck assignment differ
    matrix = np.array([
        [1.0, 2.0, 10.0],
        [2.0, 1.0, 10.0],
        [10.0, 10.0, 5.0],
    ])
    # Total sum optimal: (0->1: 2, 1->0: 2, 2->2: 5) -> max edge is 5.0
    cost, rows, cols = fastlap.solve_lbap(matrix)
    assert cost <= 5.0
    assert len(rows) == 3
    assert None not in rows


def test_solve_lbap_rectangular():
    matrix = np.array([
        [1.0, 9.0, 3.0, 8.0],
        [7.0, 2.0, 6.0, 4.0],
    ])
    cost, rows, cols = fastlap.solve_lbap(matrix)
    assert cost == 2.0  # (0->0: 1.0, 1->1: 2.0) -> max edge = 2.0
    assert rows == [0, 1]
    assert cols == [0, 1, None, None]


def test_solve_lbap_batch():
    matrices = [
        np.array([[1.0, 5.0], [5.0, 2.0]]),
        np.array([[3.0, 8.0], [8.0, 4.0]]),
    ]
    results = fastlap.solve_lbap_batch(matrices)
    assert len(results) == 2
    assert results[0][0] == 2.0  # max(1, 2)
    assert results[1][0] == 4.0  # max(3, 4)


def test_solve_lbap_maximize():
    matrix = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])
    # Maximize bottleneck (max min-edge): (0->1: 20, 1->0: 30) -> min edge is 20
    cost, rows, cols = fastlap.solve_lbap(matrix, maximize=True)
    assert cost == 20.0
    assert rows == [1, 0]


# ── Compatibility layer tests ──────────────────────────────────────────────

def test_linear_sum_assignment_compat():
    matrix = np.array([
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
    ])
    # Top-level and submodule
    r1, c1 = fastlap.linear_sum_assignment(matrix)
    r2, c2 = compat.linear_sum_assignment(matrix)
    ref_r, ref_c = scipy_lsa(matrix)

    np.testing.assert_array_equal(r1, ref_r)
    np.testing.assert_array_equal(c1, ref_c)
    np.testing.assert_array_equal(r2, ref_r)
    np.testing.assert_array_equal(c2, ref_c)
    assert r1.dtype == np.int64
    assert c1.dtype == np.int64


def test_lapjv_compat_dropin():
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    # Test top-level and submodule lapjv
    cost1, x1, y1 = fastlap.lapjv(matrix)
    cost2, x2, y2 = lap.lapjv(matrix)
    ref_cost, ref_x, ref_y = ref_lap.lapjv(matrix, extend_cost=True)

    assert abs(cost1 - ref_cost) < 1e-6
    assert abs(cost2 - ref_cost) < 1e-6
    np.testing.assert_array_equal(x1, ref_x)
    np.testing.assert_array_equal(y1, ref_y)
    assert x1.dtype == np.int32
    assert y1.dtype == np.int32


def test_lapjv_compat_cost_limit():
    matrix = np.array([
        [0.1, 0.9],
        [0.9, 0.8],
    ])
    # With cost_limit=0.5, row 1 (cost 0.8 > 0.5) is unassigned (-1)
    cost, x, y = fastlap.lapjv(matrix, cost_limit=0.5)
    assert x[0] == 0
    assert x[1] == -1
    assert y[0] == 0
    assert y[1] == -1
    assert abs(cost - 0.1) < 1e-6


def test_lapjv_compat_return_cost_false():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = fastlap.lapjv(matrix, return_cost=False)
    assert isinstance(res, tuple)
    assert len(res) == 2
    x, y = res
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


# ── Additional Algorithm Tests (ssp, cost_scaling, greedy, murty) ─────────

@pytest.mark.parametrize("size", [2, 3, 5, 8])
def test_ssp_correctness(size):
    np.random.seed(10 + size)
    matrix = np.random.uniform(1.0, 50.0, (size, size))
    cost, rows, cols = fastlap.solve_lap(matrix, algorithm="ssp")
    ref_rows, ref_cols = scipy_lsa(matrix)
    ref_cost = matrix[ref_rows, ref_cols].sum()
    assert abs(cost - ref_cost) < 1e-6
    assert rows == ref_cols.tolist()


@pytest.mark.parametrize("size", [2, 3, 5, 8])
def test_cost_scaling_correctness(size):
    np.random.seed(20 + size)
    matrix = np.random.uniform(1.0, 50.0, (size, size))
    cost, rows, cols = fastlap.solve_lap(matrix, algorithm="cost_scaling")
    ref_rows, ref_cols = scipy_lsa(matrix)
    ref_cost = matrix[ref_rows, ref_cols].sum()
    assert abs(cost - ref_cost) < 1e-6
    assert rows == ref_cols.tolist()


def test_greedy_returns_valid_matching():
    matrix = np.array([
        [1.0, 5.0, 9.0],
        [8.0, 2.0, 6.0],
        [4.0, 7.0, 3.0],
    ])
    cost, rows, cols = fastlap.solve_lap(matrix, algorithm="greedy")
    assert rows == [0, 1, 2]
    assert cols == [0, 1, 2]
    assert abs(cost - 6.0) < 1e-6


def test_solve_lap_kbest_murty():
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    # 3x3 matrix has 3! = 6 possible assignments.
    # Request k=4 top solutions in sorted order.
    solutions = fastlap.solve_lap_kbest(matrix, k=4)
    assert len(solutions) == 4
    costs = [s[0] for s in solutions]
    # Verify non-decreasing order
    assert costs == sorted(costs)
    # The optimal cost is 15.0 (0->2: 3, 1->1: 5, 2->0: 7 or 0->0: 1, 1->1: 5, 2->2: 9)
    assert abs(costs[0] - 15.0) < 1e-6


# ── Optimal duals (solve_lap_duals) ────────────────────────────────────────

def test_solve_lap_duals_complementary():
    """Duals must be feasible (u+v <= cost), tight on matched pairs, and the
    dual objective sum(u) + sum(v) must equal the primal cost (strong duality)."""
    np.random.seed(11)
    for n in [2, 5, 10]:
        m = np.random.uniform(-10, 100, (n, n))
        for algo in ["lapjv", "subgradient", "sinkhorn", "dantzig"]:
            cost, rows, cols, u, v = fastlap.solve_lap_duals(m, algorithm=algo)
            for i in range(n):
                assert len(u) == n and len(v) == n
                for j in range(n):
                    assert u[i] + v[j] <= m[i, j] + 1e-9, f"{algo}: infeasible ({i},{j})"
                j = rows[i]
                assert abs(u[i] + v[j] - m[i, j]) < 1e-7, f"{algo}: not tight ({i},{j})"
                assert cols[j] == i
            assert abs(cost - (sum(u) + sum(v))) < 1e-7, f"{algo}: duality gap"


def test_solve_lap_duals_rectangular():
    np.random.seed(12)
    m = np.random.uniform(1, 50, (3, 7))
    cost, rows, cols, u, v = fastlap.solve_lap_duals(m)
    assert len(u) == 3 and len(v) == 7
    ref_rows, ref_cols = scipy_lsa(m)
    assert abs(cost - m[ref_rows, ref_cols].sum()) < 1e-9


def test_solve_lap_duals_unsupported_algorithm():
    with pytest.raises(ValueError, match="does not support dual"):
        fastlap.solve_lap_duals(np.ones((3, 3)), algorithm="greedy")


# ── 3D batch input + n_threads ─────────────────────────────────────────────

def test_solve_lap_batch_3d_ndarray():
    """A (B, N, M) 3D ndarray is treated as B stacked matrices and must agree
    with solving each plane individually."""
    np.random.seed(13)
    b, n = 4, 6
    stack = np.random.uniform(1, 50, (b, n, n))
    results = fastlap.solve_lap_batch(stack, algorithm="lapjv")
    assert len(results) == b
    for i in range(b):
        ref_rows, ref_cols = scipy_lsa(stack[i])
        assert abs(results[i][0] - stack[i][ref_rows, ref_cols].sum()) < 1e-9


def test_solve_lap_batch_3d_int_dtype():
    stack = np.array([
        [[1, 2], [3, 4]],
        [[4, 3], [2, 1]],
        [[2, 5], [5, 2]],
    ])  # (3, 2, 2) integers
    results = fastlap.solve_lap_batch(stack, algorithm="lapjv")
    assert abs(results[0][0] - 5.0) < 1e-9
    assert abs(results[1][0] - 5.0) < 1e-9
    assert abs(results[2][0] - 4.0) < 1e-9


def test_solve_lbap_batch_3d_ndarray():
    np.random.seed(14)
    stack = np.random.uniform(0, 20, (3, 4, 4))
    results_3d = fastlap.solve_lbap_batch(stack)
    results_list = fastlap.solve_lbap_batch([m for m in stack])
    assert len(results_3d) == 3
    for (a, b) in zip(results_3d, results_list):
        assert abs(a[0] - b[0]) < 1e-9


def test_solve_lap_batch_n_threads():
    np.random.seed(15)
    matrices = [np.random.uniform(1, 50, (8, 8)) for _ in range(8)]
    r1 = fastlap.solve_lap_batch(matrices, algorithm="lapjv")
    r2 = fastlap.solve_lap_batch(matrices, algorithm="lapjv", n_threads=1)
    r3 = fastlap.solve_lap_batch(matrices, algorithm="lapjv", n_threads=3)
    assert [r[0] for r in r1] == [r[0] for r in r2] == [r[0] for r in r3]
    with pytest.raises(ValueError, match="n_threads"):
        fastlap.solve_lap_batch(matrices, algorithm="lapjv", n_threads=0)


# ── LAPJVsp (true-sparse solver) ───────────────────────────────────────────

def test_lapjvsp_in_supported():
    assert "lapjvsp" in fastlap.get_supported_algorithms()


def test_lapjvsp_dense_matches_reference():
    np.random.seed(16)
    for n in [2, 3, 5, 9]:
        m = np.random.uniform(1, 100, (n, n))
        cost, rows, cols = fastlap.solve_lap(m, algorithm="lapjvsp")
        ref_rows, ref_cols = scipy_lsa(m)
        assert abs(cost - m[ref_rows, ref_cols].sum()) < 1e-9
        assert None not in rows and None not in cols


def test_lapjvsp_sparse_csr_matches_lapmod():
    np.random.seed(17)
    n = 14
    dense = np.random.uniform(1, 100, (n, n))
    mask = np.random.rand(n, n) < 0.35
    np.fill_diagonal(mask, True)
    csr = sp.csr_matrix(np.where(mask, dense, 0))
    ref = np.where(mask, dense, np.inf)
    ref_rows, ref_cols = scipy_lsa(ref)
    ref_cost = ref[ref_rows, ref_cols].sum()

    c_mod, rows, cols = fastlap.solve_lap(csr, algorithm="lapmod")
    c_jv, rows2, cols2 = fastlap.solve_lap(csr, algorithm="lapjvsp")
    assert abs(c_mod - ref_cost) < 1e-9
    assert abs(c_jv - ref_cost) < 1e-9
    for i, j in enumerate(rows2):
        assert j is not None and mask[i, j]


def test_lapjvsp_sparse_csr_rectangular():
    np.random.seed(18)
    dense = np.random.uniform(1, 50, (8, 5))
    csr = sp.csr_matrix(dense)
    cost, rows, cols = fastlap.solve_lap(csr, algorithm="lapjvsp")
    ref_rows, ref_cols = scipy_lsa(dense)
    assert abs(cost - dense[ref_rows, ref_cols].sum()) < 1e-9
    assert sum(1 for r in rows if r is not None) == 5


def test_lapjvsp_sparse_csr_maximize_and_batch():
    np.random.seed(19)
    n = 6
    dense = np.random.uniform(1, 100, (n, n))
    csr = sp.csr_matrix(dense)
    cost, rows, cols = fastlap.solve_lap(csr, algorithm="lapjvsp", maximize=True)
    ref_rows, ref_cols = scipy_lsa(-dense)
    assert abs(cost - dense[ref_rows, ref_cols].sum()) < 1e-9

    results = fastlap.solve_lap_batch([csr, csr], algorithm="lapjvsp", maximize=True)
    assert abs(results[0][0] - dense[ref_rows, ref_cols].sum()) < 1e-9


# ── lapx-style compat helpers ──────────────────────────────────────────────

def test_compat_lapjvx_matches_scipy():
    np.random.seed(20)
    m = np.random.uniform(1, 50, (5, 5))
    cost, rows, cols = compat.lapjvx(m, return_cost=True)
    ref_rows, ref_cols = scipy_lsa(m)
    assert abs(cost - m[ref_rows, ref_cols].sum()) < 1e-9
    np.testing.assert_array_equal(rows, ref_rows)
    np.testing.assert_array_equal(cols, ref_cols)
    assert rows.dtype == np.int64
    # return_cost=False -> (rows, cols) only
    out = compat.lapjvx(m, return_cost=False)
    assert isinstance(out, tuple) and len(out) == 2


def test_compat_assignment_pairs():
    np.random.seed(21)
    m = np.random.uniform(1, 50, (4, 6))
    cost, pairs = compat.assignment_pairs(m, return_cost=True)
    ref_rows, ref_cols = scipy_lsa(m)
    assert abs(cost - m[ref_rows, ref_cols].sum()) < 1e-9
    pairs = np.asarray(pairs)
    assert pairs.shape == (4, 2)
    assert set(pairs[:, 0].tolist()) == set(ref_rows.tolist())
    assert set(pairs[:, 1].tolist()) == set(ref_cols.tolist())
    # top-level aliases exist
    assert hasattr(fastlap, "lapjvx")
    assert hasattr(fastlap, "assignment_pairs")
    arr_only = compat.assignment_pairs(m, return_cost=False)
    assert isinstance(arr_only, np.ndarray)


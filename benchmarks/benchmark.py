"""
Repeatable benchmark harness for fastlap.

Measures best-of-N wall-clock times across:
  * dense square solves, every algorithm vs scipy/lapx when available
  * rectangular solves
  * true-sparse CSR solves (lapmod vs lapjvsp vs scipy csgraph)
  * 3D ndarray batches (incl. n_threads scaling)
  * K-best (Murty) solves

Every exact algorithm is cross-checked against `lapjv` (the reference), so a
benchmark run also acts as a quick correctness sweep. Results print as a
markdown table and can be dumped to JSON for regression tracking.

Usage:
    uv run python benchmarks/benchmark.py
    uv run python benchmarks/benchmark.py --quick
    uv run python benchmarks/benchmark.py --repeat 5 --json bench.json
    uv run python benchmarks/benchmark.py --sizes 200 500 1000
"""

import argparse
import json
import time

import numpy as np

import fastlap

EXACT = [a for a in fastlap.get_supported_algorithms() if a != "greedy"]

# Known-slow dense algorithms, measured only up to this size so a benchmark
# run stays tractable (cost_scaling's bounded relabel sweep is O(n^2) per
# phase and does not scale competitively on dense random matrices).
SLOW_UPTO = {"cost_scaling": 100}

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False

try:
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    HAS_CSGRAPH = True
except ImportError:  # pragma: no cover
    HAS_CSGRAPH = False

try:
    import lap  # noqa: F401
    HAS_LAP = True
except ImportError:  # pragma: no cover
    HAS_LAP = False

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


def best_time(fn, repeat=3):
    """Return the minimum wall-clock time (seconds) over `repeat` calls."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def ms(t):
    return t * 1e3


# --------------------------------------------------------------------------
# Dense square
# --------------------------------------------------------------------------

def bench_dense_square(sizes, repeat, rows):
    rng = np.random.default_rng(0)
    for n in sizes:
        m = rng.uniform(1.0, 100.0, (n, n))
        ref, *_ = fastlap.solve_lap(m, "lapjv")
        exact_ok = True
        for algo in EXACT:
            if SLOW_UPTO.get(algo, float("inf")) < n:
                rows.append(
                    ("dense", f"{n}x{n}", f"{algo} (not timed, n>{SLOW_UPTO[algo]})",
                     float("nan"), None)
                )
                continue
            t = best_time(lambda a=algo: fastlap.solve_lap(m, a), repeat)
            cost, *_ = fastlap.solve_lap(m, algo)
            tol = 1e-3 if algo == "auction" else 1e-7
            if abs(cost - ref) > tol:
                exact_ok = False
            rows.append(("dense", f"{n}x{n}", algo, ms(t), cost))
        if HAS_SCIPY:
            t = best_time(lambda: linear_sum_assignment(m), repeat)
            rows.append(("dense", f"{n}x{n}", "scipy.optimize", ms(t), ref))
        if HAS_LAP:
            t = best_time(lambda: lap.lapjv(m, extend_cost=True), repeat)
            rows.append(("dense", f"{n}x{n}", "lap.lapjv", ms(t), ref))
        if not exact_ok:
            raise SystemExit(f"correctness sweep FAILED on dense {n}x{n}")
    # A greedy timing row (approximate, no correctness tie-in).
    m = rng.uniform(1.0, 100.0, (sizes[0], sizes[0]))
    t = best_time(lambda: fastlap.solve_lap(m, "greedy"), repeat)
    rows.append(("dense", f"{sizes[0]}x{sizes[0]}", "greedy", ms(t), None))


# --------------------------------------------------------------------------
# Rectangular
# --------------------------------------------------------------------------

def bench_rectangular(sizes, repeat, rows):
    rng = np.random.default_rng(1)
    for n in sizes:
        m = rng.uniform(1.0, 100.0, (n, 2 * n))
        ref, *_ = fastlap.solve_lap(m, "lapjv")
        for algo in ["lapjv", "hungarian", "lapmod", "lapjvsp", "auction"]:
            t = best_time(lambda a=algo: fastlap.solve_lap(m, a), repeat)
            cost, *_ = fastlap.solve_lap(m, algo)
            tol = 1e-3 if algo == "auction" else 1e-7
            if abs(cost - ref) > tol:
                raise SystemExit(f"rect correctness FAILED {n}x{2*n} {algo}")
            rows.append(("rect", f"{n}x{2*n}", algo, ms(t), cost))


# --------------------------------------------------------------------------
# Sparse CSR (true sparse path)
# --------------------------------------------------------------------------

def make_csr(n, density, seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, True)  # guarantee a feasible perfect matching
    data = np.where(mask, rng.uniform(1.0, 100.0, (n, n)), 0.0)
    return sp.csr_matrix(data)


def bench_sparse(sizes, density, repeat, rows):
    if sp is None:
        return
    for n in sizes:
        csr = make_csr(n, density, seed=n)
        ref, _, _ = fastlap.solve_lap(csr, "lapmod")
        for algo in ["lapmod", "lapjvsp"]:
            t = best_time(lambda a=algo: fastlap.solve_lap(csr, a), repeat)
            cost, *_ = fastlap.solve_lap(csr, algo)
            if abs(cost - ref) > 1e-7:
                raise SystemExit(f"sparse correctness FAILED n={n} {algo}")
            rows.append(("sparse", f"n={n} d={density}", algo, ms(t), cost))
        if HAS_CSGRAPH:
            t = best_time(lambda: min_weight_full_bipartite_matching(csr), repeat)
            rows.append(("sparse", f"n={n} d={density}", "scipy csgraph", ms(t), ref))


# --------------------------------------------------------------------------
# 3D batches
# --------------------------------------------------------------------------

def bench_batch(batch_size, n, repeat, rows):
    rng = np.random.default_rng(2)
    stack = rng.uniform(1.0, 100.0, (batch_size, n, n))
    refs = fastlap.solve_lap_batch(stack, "lapjv")
    ref0 = refs[0][0]

    t1 = best_time(lambda: fastlap.solve_lap_batch(stack, "lapjv"), repeat)
    t2 = best_time(lambda: fastlap.solve_lap_batch(stack, "lapjv", n_threads=1), repeat)
    t4 = best_time(lambda: fastlap.solve_lap_batch(stack, "lapjv", n_threads=4), repeat)
    t8 = best_time(lambda: fastlap.solve_lap_batch(stack, "lapjv", n_threads=8), repeat)
    # 1D throughput = B solves / time; speedup vs single thread.
    rows.append(("batch", f"B={batch_size} n={n}", "lapjv (all cores)", ms(t1), ref0))
    rows.append(("batch", f"B={batch_size} n={n}", "lapjv (1 thread)", ms(t2), ref0))
    rows.append(("batch", f"B={batch_size} n={n}", "lapjv (4 threads)", ms(t4), ref0))
    rows.append(("batch", f"B={batch_size} n={n}", "lapjv (8 threads)", ms(t8), ref0))


# --------------------------------------------------------------------------
# K-best (Murty)
# --------------------------------------------------------------------------

def bench_kbest(n, k, repeat, rows):
    rng = np.random.default_rng(3)
    m = rng.uniform(1.0, 100.0, (n, n))
    t = best_time(lambda: fastlap.solve_lap_kbest(m, k=k), repeat)
    sols = fastlap.solve_lap_kbest(m, k=k)
    assert len(sols) == k
    rows.append(("kbest", f"n={n} k={k}", "murty", ms(t), sols[0][0]))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="fastlap benchmark harness")
    p.add_argument("--quick", action="store_true",
                   help="small problem sizes for a fast sanity run")
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="dense square sizes (default: 100 250 500 1000)")
    p.add_argument("--repeat", type=int, default=3, help="best-of-N runs")
    p.add_argument("--json", default=None, help="write results as JSON")
    args = p.parse_args()

    sizes = args.sizes or ([50, 100, 200] if args.quick else [100, 250, 500, 1000])
    sparse_sizes = ([200, 400] if args.quick else [500, 1000, 2000])
    batch = (20, 30) if args.quick else (200, 50)
    kbest_n, kbest_k = (40, 10) if args.quick else (80, 25)

    rows = []  # (kind, shape, solver, ms, cost)
    bench_dense_square(sizes, args.repeat, rows)
    bench_rectangular(sizes[: min(2, len(sizes))], args.repeat, rows)
    bench_sparse(sparse_sizes, density=0.01, repeat=args.repeat, rows=rows)
    bench_batch(batch[0], batch[1], args.repeat, rows)
    bench_kbest(kbest_n, kbest_k, args.repeat, rows)

    json_rows = [
        {"kind": k, "problem": s, "solver": a, "time_ms": round(t, 4), "cost": c}
        for (k, s, a, t, c) in rows
        if not np.isnan(t)
    ]
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(json_rows, fh, indent=2)
        print(f"wrote {args.json}")

    print("\nfastlap benchmarks (best of {} runs, wall-clock)\n".format(args.repeat))
    header = f"{'kind':<9}{'problem':<16}{'solver':<36}{'time (ms)':>11}{'cost':>16}"
    print(header)
    print("-" * len(header))
    for kind, prob, algo, t, cost in rows:
        cost_s = "" if cost is None else f"{cost:.6f}"
        t_s = "—" if np.isnan(t) else f"{t:.3f}"
        print(f"{kind:<9}{prob:<16}{algo:<36}{t_s:>11}{cost_s:>16}")

    print("\nNote: greedy cost intentionally omitted (1/2-approximation); "
          "cost_scaling is only timed up to n=100 because its bounded relabel "
          "sweep is orders of magnitude slower than the other exact solvers on "
          "dense random matrices.")


if __name__ == "__main__":
    main()

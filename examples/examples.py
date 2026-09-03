"""Interactive quick-start: time every fastlap algorithm and (optionally)
compare against scipy.optimize / lap on the same random matrix.

    uv run python examples/examples.py            # dense comparison
    uv run python examples/examples.py --size 200
    uv run python examples/examples.py --sparse   # also show CSR lapmod/lapjvsp
    uv run python examples/examples.py --batch    # show a 3D (B, N, M) batch
"""

import argparse
import time

import numpy as np

import fastlap

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import scipy.sparse as sp
    HAS_SP = True
except ImportError:
    HAS_SP = False

try:
    import lap
    HAS_LAP = True
except ImportError:
    HAS_LAP = False


def bench(fn, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sparse", action="store_true", help="also run CSR lapmod/lapjvsp")
    p.add_argument("--batch", action="store_true", help="also run a 3D batch")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.size
    m = rng.uniform(0, 100, (n, n))

    print(f"{n}x{n} random cost matrix, best of 3\n")
    rows = []
    for algo in fastlap.get_supported_algorithms():
        t = bench(lambda a=algo: fastlap.solve_lap(m, a), 3)
        cost, *_ = fastlap.solve_lap(m, algo)
        rows.append((algo, t * 1e3, cost))

    ref = rows[[a for a, *_ in rows].index("lapjv")][2]

    if HAS_SCIPY:
        t = bench(lambda: linear_sum_assignment(m), 3)
        rows.append(("scipy.optimize", t * 1e3, None))
    if HAS_LAP:
        t = bench(lambda: lap.lapjv(m, extend_cost=True), 3)
        rows.append(("lap.lapjv", t * 1e3, None))

    print(f"{'algorithm':<16}{'time (ms)':>12}{'cost':>18}{'match lapjv':>13}")
    print("-" * 60)
    for algo, t, cost in rows:
        match = "" if cost is None else ("yes" if abs(cost - ref) < 1e-6 else "no")
        cost_s = "" if cost is None else f"{cost:.6f}"
        print(f"{algo:<16}{t:>12.3f}{cost_s:>18}{match:>13}")

    if args.sparse and HAS_SP:
        print("\nSparse CSR (density 0.5%):")
        mask = rng.random((n, n)) < 0.005
        np.fill_diagonal(mask, True)
        data = np.where(mask, rng.uniform(0, 100, (n, n)), 0.0)
        csr = sp.csr_matrix(data)
        for algo in ("lapmod", "lapjvsp"):
            t = bench(lambda a=algo: fastlap.solve_lap(csr, a), 3)
            cost, *_ = fastlap.solve_lap(csr, algo)
            print(f"  {algo:<10}{t * 1e3:>9.3f} ms   cost {cost:.6f}")

    if args.batch:
        b = 100
        stack = rng.uniform(0, 100, (b, n, n))
        t = bench(lambda: fastlap.solve_lap_batch(stack, "lapjv"), 3)
        print(f"\n3D batch: {b} x ({n}x{n}) in {t * 1e3:.3f} ms "
              f"({b / t / 1e3:.1f} solves/s)")


if __name__ == "__main__":
    main()

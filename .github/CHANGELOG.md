# Changelog

All notable changes to fastlap are documented here.

## [0.4.0] — Unreleased

### Fixed
- **Sparse CSR input now works with every algorithm.** Structurally missing
  entries were densified to `f64::INFINITY`, which `validate_matrix` then
  rejected, so `solve_lap(csr, "lapjv")` (and every other non-sparse
  algorithm) raised `Matrix contains infinite value`. Missing entries are now
  filled with a dimension-scaled finite sentinel that no real assignment can
  beat, giving true forbidden-edge semantics.
- **Non-CSR sparse formats are rejected clearly.** `csc_matrix` shares the
  `indptr`/`indices`/`data`/`shape` quartet with CSR and was silently misread;
  `coo_matrix` failed with a cryptic numpy error. Both now raise
  `TypeError: Unsupported sparse format ..., convert with .tocsr()`.
- **Murty K-best no longer corrupts large-cost problems.** Forbidden/fixed-away
  edges were masked with the hard-coded sentinel `1e12`, which is cheaper than
  real costs once entries reach that magnitude. The sentinel now scales with
  the matrix's max magnitude and dimension.
- **`solve_lap_duals` honors its `algorithm` argument.** Each supported
  algorithm (`lapjv`, `subgradient`, `sinkhorn`, `dantzig`) now seeds the exact
  SAP dual recovery with its own native feasible potentials instead of the
  argument being ignored.
- **Auction no longer errors out.** If its ε-scaling budget is exhausted it now
  falls back to the exact LAPJV solver instead of returning an error.
- The common `solve_lap(..., maximize=False, cost_limit=None)` path skips an
  unnecessary full-matrix clone and cost recompute.
- Documented that `cost_limit` is post-filter gating and does **not** match
  `lap.lapjv`/`lapx`'s constrained `(N+M)×(N+M)` re-solve.

### Documentation
- Added a production-readiness breakdown (production defaults vs. niche-exact
  vs. approximate) and a measured-performance table to the algorithms page,
  with a note on the hardware and a reproducible benchmark command.

### Performance
- **Single-matrix solves release the GIL.** `solve_lap`, `solve_lap_weighted`,
  `solve_lbap`, `solve_lap_kbest`, and `solve_lap_duals` now run their pure-Rust
  solve under `py.allow_threads`, so other Python threads keep running during a
  large solve. (Batch entry points already did.)
- **`pad_to_square` no longer deep-copies square matrices.** It returns a
  `Cow`: already-square input is borrowed instead of paying a full `n²` copy on
  every solve.

### Added
- **`lapjvsp` algorithm** — true-sparse Jonker–Volgenant (sparse column reduction + reduction transfer, warm-started sparse SAP); never densifies `scipy.sparse` CSR input.
- **`solve_lap_duals`** — returns optimal dual potentials `(u, v)` alongside the assignment (`lapjv`, `subgradient`, `sinkhorn`, `dantzig`).
- **3D batch input** — `solve_lap_batch` / `solve_lbap_batch` accept a `(B, N, M)` ndarray as a stack of matrices.
- **`n_threads` parameter** — cap the Rayon worker count on `solve_lap_batch` / `solve_lbap_batch`.
- **lapx-style compat helpers** — `lapjvx` (aligned index arrays) and `assignment_pairs` (`(K, 2)` pairs), top-level and under `fastlap.compat`.
- **`lapjvx_batch` / `lapjvxa_batch`** — lapx-style parallel batch solvers returning NumPy index arrays (`(costs, rows_list, cols_list)` and `(costs, assignments)`), with `return_cost` and `n_threads`; also under `fastlap.lap` and `fastlap.compat`.
- **`examples/terminal_ui.py`** — ANSI block heatmap with assignment overlay + `compare` subcommand racing all algorithms (optional `rich` tables).
- **`examples/bipartite_assignment.py`** — bipartite-graph rendering of an assignment (matplotlib + networkx).
- Dual feasibility/complementary-slackness Rust tests, LAPJVsp brute-force tests, and Python tests for every new feature.
- Documentation pages for LAPJVsp, optimal duals, visualisation/demos, and refresh of batch/sparse/compat/API pages.
- **Independent brute-force oracle tests** for every exact algorithm (small
  integer matrices, all permutations) plus a maximize oracle — an oracle that
  does not depend on SciPy.
- **CI:** non-blocking `cargo audit` advisory report, and a quick benchmark
  sweep uploaded as a JSON artifact for regression tracking.
- The dense sparse-input test now passes the CSR matrix directly instead of
  `.toarray()`, so it actually exercises the sparse-extraction path.
- **`solve_lap_batch`** — solve many independent LAPs in parallel via Rayon.
- **`solve_lap_weighted`** — reweight entries before solving (tracking pipeline support).
- **NaN/Inf/empty input validation** — rejects invalid matrices with precise `[i,j]` error messages.
- **`Option<usize>` return type** — unassigned entries are `None` (was `usize::MAX`).
- **Auction max-iteration guard** — `MAX_ITERATIONS = 1_000_000` prevents infinite loops.
- **Module docstring** — `help(fastlap)` now shows API overview.
- **`solve_with()` dispatch** — single source of truth for algorithm routing (deduplicated).
- Non-square matrix tests (6 parametrized shapes).
- Edge-case tests: 1×1, 2×2 identity, duplicate costs, diagonal, large cost range.
- NaN/Inf/empty/error rejection tests + None sentinel test + module doc test.
- `CONTRIBUTING.md` with dev setup, testing, and algorithm-addition guide.
- Algorithm comparison table in README.
- `examples/visualize_assignment.py` — matplotlib heatmap with assignment overlay.
- `examples/__init__.py` re-exports `run_example()`.
- `scripts/test_all.sh` — chains Rust + Python checks.
- `Makefile` with `test`, `build`, `clean`, `fmt`, `clippy` targets.
- `.cargo/config.toml` — warnings-as-errors in local builds.
- `docs/content/journal.md` — replaced lorem-ipsum with real dev journal.
- `aarch64-unknown-linux-gnu` build target in CI publish workflow.
- `cargo audit` step in CI lint job.

### Changed
- Version bumped to 0.4.0.
- **Breaking:** `LapSolution` now uses `Vec<Option<usize>>` — unassigned entries are `None`.
- Python version requirement: `>=3.9`. Numpy requirement: `>=1.26`.
- Maturin build requirement lowered to `>=1.5` for broader compatibility.
- Auction/Subgradient now fall back to padded SAP for rectangular matrices.
- Improved PyPI metadata: 16 keywords, full Python 3.9–3.13 classifiers, topic categories.

---

## [0.3.0] — 2026-07-19

### Added
- `solve_lap_batch(matrices, algorithm)` — parallel batch solving.
- `solve_lap_weighted(cost_matrix, weights, algorithm)` — weighted cost support.
- Non-square matrix tests, edge-case tests, batch + weighted tests.
- `CONTRIBUTING.md`, algorithm comparison table, visualization example.
- `Makefile`, `scripts/test_all.sh`, `.cargo/config.toml`.
- Python docstrings on `solve_lap`, `solve_lap_batch`, `solve_lap_weighted`.

### Changed
- Version bumped to 0.3.0.
- Auction `best_item` uses `Option<usize>` guard.
- README: removed "in progress" badge, updated version badge.

---

## [0.2.0] — 2026-07-19

### Added
- Correctness benchmark: 1000 random matrices (sizes 2–50).
- Sparse input test (`scipy.sparse.csr_matrix` verified).
- `aarch64-unknown-linux-gnu` CI target.
- Python 3.11, 3.12, 3.13 in CI matrix.
- `cargo audit` in CI.
- `scripts/publish.sh` supports `testpypi` argument.

### Changed
- Python `>=3.9`, NumPy `>=1.26`.

---

## [0.1.1] — 2025-06-23

### Added
- Dantzig's algorithm, Auction algorithm, Subgradient algorithm.

---

## [0.1.0] — 2025-06-23

### Added
- Initial release: LAPJV, Hungarian, LAPMOD.
- PyO3 bindings via `solve_lap()`.

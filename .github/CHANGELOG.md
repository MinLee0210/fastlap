# Changelog

All notable changes to fastlap are documented here.

## [0.4.0] — Unreleased

### Added
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
- SEO-optimised README with FAQ, comparison table, use-case list, and rich keywords.

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

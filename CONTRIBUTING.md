# Contributing to fastlap

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

**Prerequisites:** Python ≥ 3.9, Rust toolchain, [uv](https://docs.astral.sh/uv/) (recommended).

```bash
git clone https://github.com/MinLee0210/fastlap.git
cd fastlap
uv sync               # creates venv + installs dev deps
uv run maturin develop # builds Rust extension in-place
```

## Running Tests

```bash
# Rust checks (fmt + clippy + unit tests)
./scripts/test_rust.sh

# Python tests
./scripts/test_python.sh

# Everything
./scripts/test_all.sh
# or
make test
```

## Code Style

- **Rust:** `cargo fmt` defaults. Clippy warnings are treated as errors (`.cargo/config.toml`).
- **Python:** Follow existing patterns in `tests/`. No formatter enforced, but consistent style preferred.

## Project Structure

```
src/
  lib.rs            # PyO3 module: solve_lap, solve_lap_batch, solve_lap_weighted,
                     # solve_lbap(_batch), solve_lap_kbest, compat shims
  types.rs          # LapSolution type alias, SparseCost adjacency struct
  matrix.rs         # NumPy / CSR → Vec<Vec<f64>> extraction + validation
  utils.rs          # solve_with() dispatch, sap_solve(), cost-limit gating, pad/trim helpers
  lap/
    mod.rs
    lapjv.rs        # LAPJV — column reduction + warm-started shortest augmenting path
    hungarian.rs    # Hungarian (Kuhn-Munkres) — star/prime zero-covering
    lapmod.rs       # LAPMOD — sparse-adjacency shortest augmenting path
    dantzig.rs      # Dantzig — primal network simplex on the assignment LP
    auction.rs      # Auction — iterative bidding with ε-scaling
    subgradient.rs  # Subgradient dual ascent + SAP recovery
    sinkhorn.rs      # Sinkhorn — entropic OT dual scaling + SAP recovery
    ssp.rs          # Successive Shortest Path / min-cost max-flow
    cost_scaling.rs # Goldberg–Kennedy push-relabel with ε-relaxation
    greedy.rs       # Greedy 1/2-approximation
    murty.rs        # Murty's algorithm — ranked K-best assignments
    bottleneck.rs   # LBAP — binary search + Hopcroft-Karp
tests/
  conftest.py           # Helpers: generate_test_matrix, scipy_execute, etc.
  test_correctness.py   # Correctness tests covering all algorithms + edge cases
  test_new_features.py  # Cost limit, LBAP, K-best, compat layer, sparse tests
  test_performance.py   # Timing benchmarks
docs/                # MkDocs Material site (see "Building the Docs" below)
```

## Pull Requests

1. Fork the repo and create a feature branch.
2. Make your changes and ensure `make test` passes.
3. Add tests for new algorithms or features.
4. Update `TODO.md` to check off completed items.
5. Open a PR against `main`.

## Building the Docs

Docs live in `docs/` and are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

```bash
make docs-install   # pip install -r requirements-docs.txt
make docs-serve      # live-reload at http://127.0.0.1:8000
make docs-build      # strict build, matches CI
```

The site auto-deploys to GitHub Pages on every push to `main` that touches `docs/`, `mkdocs.yml`, `README.md`, `CONTRIBUTING.md`, or `.github/CHANGELOG.md` (see `.github/workflows/docs.yml`).

## Adding a New Algorithm

1. Create `src/lap/your_algo.rs` with `pub fn solve(matrix: Vec<Vec<f64>>) -> LapSolution`.
2. Register it in `src/lap/mod.rs`.
3. Add a match arm in `src/utils.rs::solve_with()`.
4. Add the name to `supported_algorithms()` in `src/utils.rs`.
5. Add correctness tests in `tests/test_correctness.py` comparing against SciPy's `linear_sum_assignment`.
6. Document it in `docs/algorithms.md` (comparison table + a short section) and in `fastlap.pyi`'s `Algorithm` literal.

## Reporting Issues

Open a GitHub issue with:
- A minimal reproducing example
- Expected vs actual behavior
- Python and Rust versions (`python --version`, `rustc --version`)

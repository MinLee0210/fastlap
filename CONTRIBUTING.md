# Contributing to fastlap

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

**Prerequisites:** Python ≥ 3.9, Rust toolchain, [uv](https://docs.astral.sh/uv/) (recommended).

```bash
git clone https://github.com/LakoreAI/fastlap.git
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
  lib.rs          # PyO3 module: solve_lap, solve_lap_batch, solve_lap_weighted
  types.rs        # LapSolution type alias
  matrix.rs       # NumPy / CSR → Vec<Vec<f64>> extraction + validation
  utils.rs        # solve_with() dispatch, sap_solve(), pad/trim helpers
  lap/
    mod.rs
    lapjv.rs      # LAPJV — shortest augmenting path
    hungarian.rs  # Hungarian (Kuhn-Munkres)
    lapmod.rs     # LAPMOD — LAPJV with large sentinel padding
    dantzig.rs    # Dantzig — simplex on assignment polytope
    auction.rs    # Auction — iterative bidding (ε-optimal)
    subgradient.rs # Subgradient dual + SAP recovery
tests/
  conftest.py         # Helpers: generate_test_matrix, scipy_execute, etc.
  test_correctness.py # 73+ tests covering all algorithms + edge cases
  test_performance.py # Timing benchmarks
```

## Pull Requests

1. Fork the repo and create a feature branch.
2. Make your changes and ensure `make test` passes.
3. Add tests for new algorithms or features.
4. Update `docs/TODO.md` to check off completed items.
5. Open a PR against `main`.

## Adding a New Algorithm

1. Create `src/lap/your_algo.rs` with `pub fn solve(matrix: Vec<Vec<f64>>) -> LapSolution`.
2. Register it in `src/lap/mod.rs`.
3. Add a match arm in `src/utils.rs::solve_with()`.
4. Add the name to `supported_algorithms()` in `src/utils.rs`.
5. Add correctness tests in `tests/test_correctness.py` comparing against SciPy's `linear_sum_assignment`.

## Reporting Issues

Open a GitHub issue with:
- A minimal reproducing example
- Expected vs actual behavior
- Python and Rust versions (`python --version`, `rustc --version`)

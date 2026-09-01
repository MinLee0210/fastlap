+++
title = "About fastlap"
+++

## What is fastlap?

**fastlap** is a high-performance Python library for solving the **Linear Assignment Problem (LAP)** — the combinatorial optimisation task of finding a minimum-cost one-to-one matching between two sets. It is implemented in **Rust** and exposed to Python via [PyO3](https://pyo3.rs).

fastlap is designed for researchers, engineers, and data scientists who need fast, reliable assignment solutions without leaving the Python ecosystem.

## Why Rust?

The LAP is central to **object tracking** (Hungarian tracker, SORT, DeepSORT), **task scheduling**, **resource allocation**, and **combinatorial optimisation**. Python-native solvers are fine for prototyping, but production pipelines — real-time tracking, large-scale logistics, robotics — need sub-millisecond assignment. Rust gives fastlap:

- **Memory safety** without a garbage collector
- **Zero-cost abstractions** for algorithmic code
- **SIMD-friendly** data layouts
- **Parallelism** via [Rayon](https://docs.rs/rayon) for batch solving

## Algorithms

fastlap ships six algorithms behind a uniform API:

| Algorithm | Complexity | Reference |
|-----------|-----------|-----------|
| LAPJV | O(n³) | Jonker & Volgenant, 1987 |
| Hungarian | O(n³) | Kuhn, 1955 / Munkres, 1957 |
| LAPMOD | O(n³) | LAPJV variant with sparse-aware padding |
| Dantzig | O(n³) | Dantzig, 1963 (simplex on assignment polytope) |
| Auction | O(n²·k) | Bertsekas, 1988 (ε-optimal) |
| Subgradient | O(n³ + iters·n²) | Held & Karp, 1971 (dual warm-up) |

## Links

- [GitHub](https://github.com/LakoreAI/fastlap)
- [PyPI](https://pypi.org/project/fastlap/)
- [Report a bug](https://github.com/LakoreAI/fastlap/issues)

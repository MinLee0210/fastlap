---
title = "Development Journal"
date = 2025-06-23
draft = false

[extra]
display_published = false
+++

## 2026-07-19

Implemented all P0 and P1 items from the project roadmap. Added correctness
benchmark (1000 random matrices across 5 sizes), guarded auction algorithm
initialization, pinned Python 3.9–3.13 CI matrix, added aarch64 Linux build
target, and introduced cargo audit to CI. Version bumped to 0.2.0.

## 2025-06-23

Initial release of fastlap with LAPJV, Hungarian, LAPMOD, Dantzig's, Auction,
and Subgradient algorithms. PyO3 bindings expose a single `solve_lap()` entry
point accepting dense NumPy arrays and returning `(cost, row_assign, col_assign)`.

#!/bin/bash
set -e

echo "Running Rust fmt check..."
cargo fmt --check

echo "Running Rust tests..."
cargo test --all-features

echo "Running Rust clippy..."
cargo clippy --all-targets --all-features -- -D warnings

echo "Rust verification complete!"

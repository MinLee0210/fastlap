#!/bin/bash
set -e

echo "=== Rust checks ==="
./scripts/test_rust.sh

echo ""
echo "=== Python checks ==="
./scripts/test_python.sh

echo ""
echo "All checks passed!"

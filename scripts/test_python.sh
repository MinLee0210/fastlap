#!/bin/bash
set -e

echo "Building fastlap in develop mode..."
uv run maturin develop

echo "Running Python tests..."
uv run python -m pytest tests/

echo "Python verification complete!"

#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env if it exists
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

if [ -z "$PYPI_API_KEY" ]; then
    echo "Error: PYPI_API_KEY not set. Create .env with your key or export it."
    exit 1
fi

TARGET="${1:-pypi}"

if [ "$TARGET" = "testpypi" ]; then
    echo "Publishing to TestPyPI..."
    uv run twine upload --repository testpypi --username __token__ --password "$PYPI_API_KEY" dist/*
    echo "TestPyPI publish complete!"
elif [ "$TARGET" = "pypi" ]; then
    echo "Publishing to PyPI..."
    uv run twine upload --username __token__ --password "$PYPI_API_KEY" dist/*
    echo "PyPI publish complete!"
else
    echo "Usage: $0 [testpypi|pypi]"
    exit 1
fi

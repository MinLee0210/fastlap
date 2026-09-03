.PHONY: test test-rust test-python build clean fmt clippy bench docs-install docs-serve docs-build

test: test-rust test-python

test-rust:
	./scripts/test_rust.sh

test-python:
	./scripts/test_python.sh

build:
	uv run maturin build --release

bench:
	uv run python benchmarks/benchmark.py

clean:
	rm -rf dist/ target/ build/ *.egg-info

fmt:
	cargo fmt

clippy:
	cargo clippy -- -D warnings

docs-install:
	pip install -r requirements-docs.txt

docs-serve:
	mkdocs serve

docs-build:
	mkdocs build --strict

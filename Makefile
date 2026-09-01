.PHONY: test test-rust test-python build clean fmt clippy

test: test-rust test-python

test-rust:
	./scripts/test_rust.sh

test-python:
	./scripts/test_python.sh

build:
	uv run maturin build --release

clean:
	rm -rf dist/ target/ build/ *.egg-info

fmt:
	cargo fmt

clippy:
	cargo clippy -- -D warnings

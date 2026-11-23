.PHONY: install lint format typecheck test check demo

install:      ## dev environment (CPU torch)
	uv sync --extra dev --extra data --extra viz --extra serve

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest -m "not network" --cov

check: lint typecheck test   ## everything CI runs

demo:         ## synthetic end-to-end training run on CPU
	uv run systemone demo

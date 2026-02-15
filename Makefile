.PHONY: install format lint typecheck test check

install:
	poetry install

format:
	poetry run ruff format src tests
	poetry run ruff check --fix src tests

lint:
	poetry run ruff format --check src tests
	poetry run ruff check src tests

typecheck:
	poetry run mypy

test:
	poetry run pytest -v

check: lint typecheck test

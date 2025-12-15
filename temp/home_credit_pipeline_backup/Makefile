.PHONY: install lint format typecheck test run clean

install:
	@echo "📦 Installing project with uv..."
	uv sync --all-extras

lint:
	@echo "🧹 Linting..."
	uv run ruff check src tests

format:
	@echo "🎨 Formatting..."
	uv run black src tests

typecheck:
	@echo "🔍 Type checking..."
	uv run mypy src --ignore-missing-imports

test:
	@echo "🧪 Running tests..."
	uv run pytest -v --cov=src/home_credit --cov-report=term-missing

run:
	@echo "🚀 Running pipeline..."
	uv run python main.py

clean:
	@echo "🗑️ Cleaning artifacts..."
	rm -rf artifacts .pytest_cache .mypy_cache .ruff_cache .coverage

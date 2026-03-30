.PHONY: install test run lint format clean

install:
	pip install -e .
	pip install pytest black ruff pre-commit

test:
	pytest -q

run:
	python -m financial_ml_tick_imbalance.pipeline.run_pipeline

lint:
	ruff check src tests

format:
	black src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
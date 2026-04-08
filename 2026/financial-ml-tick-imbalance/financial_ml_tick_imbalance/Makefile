.PHONY: install test ingest bars train research lint format clean tree

PYTHONPATH=src

install:
	pip install -e .
	pip install pytest black ruff pre-commit

test:
	PYTHONPATH=$(PYTHONPATH) pytest -q tests

ingest:
	PYTHONPATH=$(PYTHONPATH) python -m financial_ml_tick_imbalance.pipeline.ingest_tick_data

bars:
	PYTHONPATH=$(PYTHONPATH) python -m financial_ml_tick_imbalance.pipeline.build_imbalance_bars

train:
	PYTHONPATH=$(PYTHONPATH) python -m financial_ml_tick_imbalance.pipeline.train_model

research:
	PYTHONPATH=$(PYTHONPATH) python -m financial_ml_tick_imbalance.pipeline.full_research_pipeline

lint:
	ruff check src tests

format:
	black src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

tree:
	tree src/financial_ml_tick_imbalance
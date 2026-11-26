VENV := .venv
BIN := $(VENV)/bin
export PYTHONPATH := src

.PHONY: help setup test lint demo serve docker-build

help: ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install dependencies
	python3 -m venv $(VENV)
	$(BIN)/pip install -q -r requirements-dev.txt

test: ## Run unit and integration tests
	$(BIN)/python -m pytest -q

lint: ## Run ruff linter
	$(BIN)/ruff check src tests scripts

demo: ## Run end-to-end demo (train, score, DuckDB, render PNG report)
	$(BIN)/python scripts/run_demo.py

serve: ## Start FastAPI inference server on port 8083
	$(BIN)/uvicorn iot_pipeline.api:app --host 0.0.0.0 --port 8083

docker-build: ## Build production Docker image
	docker build -t iot-anomaly-pipeline:latest .

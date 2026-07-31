# Canonical entry points. Local development, CI and the container images
# all run these same commands, so "works on my machine" and "works in CI"
# cannot drift apart.

VENV    ?= .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
API_PORT       ?= 8000
DASHBOARD_PORT ?= 8501
DATASET        ?= data/cml_sample_500.csv
CALIBRATE      ?=

.DEFAULT_GOAL := help
.PHONY: help setup api dashboard dev test test-fast lint typecheck audit format check train docker-up docker-down clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install all dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	@echo "Ready. Next: make test, then make dev"

api:  ## Run the API
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port $(API_PORT)

dashboard:  ## Run the Streamlit dashboard
	CML_API_URL=http://localhost:$(API_PORT) \
		$(VENV)/bin/streamlit run streamlit_app.py --server.port $(DASHBOARD_PORT)

dev:  ## Run API and dashboard together with auto-reload
	@echo "API       http://localhost:$(API_PORT)/docs"
	@echo "Dashboard http://localhost:$(DASHBOARD_PORT)"
	@trap 'kill 0' EXIT; \
	$(VENV)/bin/uvicorn app.main:app --reload --port $(API_PORT) & \
	CML_API_URL=http://localhost:$(API_PORT) \
		$(VENV)/bin/streamlit run streamlit_app.py --server.port $(DASHBOARD_PORT) & \
	wait

test:  ## Run the full test suite with coverage
	$(PY) -m pytest --cov --cov-report=term-missing

test-fast:  ## Run the test suite, skipping slow tests
	$(PY) -m pytest -m "not slow"

lint:  ## Check formatting and lint rules
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

typecheck:  ## Run static type checking
	$(VENV)/bin/mypy

audit:  ## Check dependencies for known vulnerabilities
	$(VENV)/bin/pip-audit

format:  ## Apply formatting and safe lint fixes
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

check: lint typecheck audit test  ## Everything CI runs

train:  ## Retrain the model from DATASET (overwrites models/). CALIBRATE=sigmoid|isotonic
	$(PY) ml/train_enhanced.py $(DATASET) $(if $(CALIBRATE),--calibrate $(CALIBRATE),)

docker-up:  ## Build and start the full stack
	docker compose up --build

docker-down:  ## Stop the stack
	docker compose down

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov *.egg-info
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +

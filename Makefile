.PHONY: help install install-dev test test-cov test-fast lint format type-check security clean docs serve-docs build publish metrics-collect metrics-update metrics-report

# Default target
help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install:  ## Install package in production mode
	pip install -e .

install-dev:  ## Install package in development mode with all dependencies
	pip install -e ".[dev,docs,benchmark]"
	pre-commit install

# Testing
test:  ## Run all tests
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-fast:  ## Run tests excluding slow ones
	pytest -m "not slow"

test-privacy:  ## Run privacy-specific tests
	pytest -m privacy

test-integration:  ## Run integration tests
	pytest -m integration

# Code Quality
lint:  ## Run all linting checks
	flake8 src/ tests/
	ruff check src/ tests/
	bandit -r src/

format:  ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

type-check:  ## Run type checking with mypy
	mypy src/

security:  ## Run security checks
	bandit -r src/ -f json
	safety check
	pip-audit

# Pre-commit
pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Documentation
docs:  ## Build documentation
	sphinx-build -b html docs/ docs/_build/html

serve-docs:  ## Serve documentation locally
	sphinx-autobuild docs/ docs/_build/html --host 0.0.0.0 --port 8000

# Build and publish
build:  ## Build distribution packages
	python -m build

publish-test:  ## Publish to Test PyPI
	python -m twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	python -m twine upload dist/*

# Cleaning
clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Development utilities
setup-dev:  ## Complete development environment setup
	python -m venv venv
	@echo "Activate the virtual environment with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"
	@echo "Then run: make install-dev"

benchmark:  ## Run benchmarks
	python -m dp_federated_lora.benchmarks.run_all

privacy-audit:  ## Audit privacy guarantees
	python -m dp_federated_lora.audit.privacy_checker

federated-demo:  ## Run federated learning demo
	python examples/federated_demo.py

check-all:  ## Run all checks (lint, type, security, test)
	@echo "Running complete code quality check..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security
	$(MAKE) test-fast
	@echo "✅ All checks passed!"

# CI/CD simulation
ci:  ## Simulate CI pipeline locally
	@echo "🔄 Simulating CI pipeline..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security
	$(MAKE) test-cov
	$(MAKE) docs
	@echo "✅ CI simulation completed successfully!"

# Docker
docker-build:  ## Build Docker image
	docker build -t dp-federated-lora:latest .

docker-run:  ## Run Docker container
	docker run --rm -it --gpus all dp-federated-lora:latest

docker-test:  ## Run tests in Docker
	docker run --rm dp-federated-lora:latest pytest

# Environment info
env-info:  ## Show environment information
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "PyTorch version: $$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
	@echo "GPU count: $$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"

# Maintenance
update-deps:  ## Update all dependencies to latest versions
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in

freeze-deps:  ## Freeze current dependency versions
	pip freeze > requirements-frozen.txt

# Performance profiling
profile:  ## Run performance profiling
	python -m cProfile -o profile.stats examples/profile_training.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Metrics and Automation
metrics-collect:  ## Collect comprehensive metrics
	python scripts/collect_metrics.py --update-file --verbose

metrics-update:  ## Update metrics with new values
	python scripts/update_metrics.py --visualize

metrics-report:  ## Generate comprehensive reports
	python scripts/generate_reports.py

metrics-all:  ## Run complete metrics pipeline
	@echo "🔄 Running complete metrics pipeline..."
	$(MAKE) metrics-collect
	$(MAKE) metrics-update
	$(MAKE) metrics-report
	@echo "✅ Metrics pipeline completed!"

metrics-dashboard:  ## Open metrics dashboard (requires reports)
	@echo "📊 Metrics dashboard files:"
	@ls -la reports/ 2>/dev/null || echo "No reports found. Run 'make metrics-all' first."
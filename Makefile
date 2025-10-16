# RAG Agent Makefile

.PHONY: help install install-dev test lint format check-quality setup-pre-commit clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run tests with coverage"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  check-quality    Run all quality checks"
	@echo "  setup-pre-commit Install pre-commit hooks"
	@echo "  clean            Clean up temporary files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v

# Code quality
lint:
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

check-quality:
	python scripts/check_code_quality.py --check

fix-formatting:
	python scripts/check_code_quality.py --fix

# Pre-commit setup
setup-pre-commit:
	pip install pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg

# Development setup
setup-dev: install-dev setup-pre-commit
	@echo "Development environment setup complete!"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf bandit-report.json

# Docker
docker-build:
	docker build -t rag-agent .

docker-run:
	docker run -d --name rag-agent -v $(PWD)/data:/app/data rag-agent

docker-stop:
	docker stop rag-agent
	docker rm rag-agent

# Documentation
docs:
	@echo "Generating documentation..."
	# Add documentation generation commands here

# All-in-one development workflow
dev: format lint test
	@echo "Development workflow complete!"

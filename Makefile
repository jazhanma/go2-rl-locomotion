# Go2 Quadruped Locomotion Framework Makefile

.PHONY: help install test train clean demo

help:  ## Show this help message
	@echo "Go2 Quadruped Locomotion Framework"
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -e .
	pip install pytest black flake8 mypy

test:  ## Run tests
	python -m pytest tests/ -v

test-policies:  ## Run policy tests
	python -m pytest tests/test_policies.py -v

train-ppo:  ## Train PPO baseline policy
	python train.py --policy ppo_baseline --timesteps 100000 --render

train-residual:  ## Train Residual RL policy
	python train.py --policy residual_rl --timesteps 100000 --render

train-bc:  ## Train BC pretrain policy
	python train.py --policy bc_pretrain --timesteps 100000 --render

train-asymmetric:  ## Train Asymmetric Critic policy
	python train.py --policy asymmetric_critic --timesteps 100000 --render

demo:  ## Run visualization demo
	python examples/visualization_demo.py

demo-basic:  ## Run basic training demo
	python examples/basic_training.py

clean:  ## Clean generated files
	rm -rf logs/ plots/ videos/ models/ __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean  ## Clean all files including dependencies
	rm -rf build/ dist/ *.egg-info/
	pip uninstall go2-locomotion -y

format:  ## Format code with black
	black src/ tests/ examples/ train.py

lint:  ## Lint code with flake8
	flake8 src/ tests/ examples/ train.py

type-check:  ## Type check with mypy
	mypy src/

check: format lint type-check test  ## Run all checks

build:  ## Build package
	python setup.py sdist bdist_wheel

upload:  ## Upload to PyPI
	twine upload dist/*

docs:  ## Generate documentation
	# Add documentation generation commands here
	@echo "Documentation generation not implemented yet"

setup: install  ## Setup development environment
	@echo "Setting up development environment..."
	@echo "Creating directories..."
	mkdir -p logs plots videos models data configs
	@echo "Development environment ready!"

run-example:  ## Run example training
	python train.py --config configs/ppo_baseline.yaml --render --record

run-eval:  ## Run evaluation
	python train.py --policy ppo_baseline --eval --rollout

run-all-policies:  ## Train all policy types
	@echo "Training all policy types..."
	$(MAKE) train-ppo
	$(MAKE) train-residual
	$(MAKE) train-bc
	$(MAKE) train-asymmetric

benchmark:  ## Run performance benchmark
	@echo "Running performance benchmark..."
	python -c "import time; start=time.time(); exec(open('examples/basic_training.py').read()); print(f'Benchmark completed in {time.time()-start:.2f}s')"

profile:  ## Profile training performance
	python -m cProfile -o profile.stats train.py --policy ppo_baseline --timesteps 10000
	python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Default target
.DEFAULT_GOAL := help


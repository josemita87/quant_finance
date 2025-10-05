POETRY ?= poetry
PROJECTS := DataScience FinancialMath GraphTheory MachineLearning PortfolioTracking

.PHONY: help install format lint test run run-data run-finance run-graph run-ml run-portfolio

help:
	@echo "Available targets:"
	@echo "  make install                      # Install dependencies for all Poetry projects"
	@echo "  make format                       # Run ruff formatter across projects"
	@echo "  make lint                         # Run ruff lint checks across projects"
	@echo "  make test                         # Run pytest for projects that define tests"
	@echo "  make run project=DIR module=...   # Run a module via Poetry (optional args=...)"
	@echo "  make run-data [args=...]          # Shortcut for DataScience CLI"
	@echo "  make run-finance                  # Shortcut for FinancialMath demo"
	@echo "  make run-graph module=...         # Shortcut for GraphTheory modules"
	@echo "  make run-ml [args=...]            # Shortcut for MachineLearning example"
	@echo "  make run-portfolio                # Shortcut for PortfolioTracking CLI"

install:
	@for project in $(PROJECTS); do \
		echo "Installing $$project"; \
		( cd $$project && $(POETRY) install ); \
	done

format:
	@for project in $(PROJECTS); do \
		echo "Formatting $$project"; \
		( cd $$project && $(POETRY) run ruff format . ); \
	done

lint:
	@for project in $(PROJECTS); do \
		echo "Linting $$project"; \
		( cd $$project && $(POETRY) run ruff check . ); \
	done

test:
	@for project in $(PROJECTS); do \
		if [ -d $$project/tests ] || [ -d $$project/src ]; then \
			echo "Testing $$project"; \
			( cd $$project && $(POETRY) run pytest ) || exit $$?; \
		fi; \
	done

args ?=

run:
	@if [ -z "$(project)" ] || [ -z "$(module)" ]; then \
		echo "Usage: make run project=ProjectDir module=python.module [args='--flags']"; \
		exit 1; \
	fi
	@echo "Running in $(project): python -m $(module) $(args)"
	@cd $(project) && $(POETRY) run python -m $(module) $(args)

run-data:
	@$(MAKE) -s run project=DataScience module=monte_carlo_methods.main args="$(args)"

run-finance:
	@$(MAKE) -s run project=FinancialMath module=financial_math.efficient_frontier args="$(args)"

run-graph:
	@if [ -z "$(module)" ]; then \
		echo "Usage: make run-graph module=gt_algorithms.<package>.<module> [args=...]"; \
		exit 1; \
	fi
	@$(MAKE) -s run project=GraphTheory module=$(module) args="$(args)"

run-ml:
	@$(MAKE) -s run project=MachineLearning module=digit_genetics.mnist_genetic_algorithm args="$(args)"

run-portfolio:
	@$(MAKE) -s run project=PortfolioTracking module=portfolio_tracking.portfolio args="$(args)"

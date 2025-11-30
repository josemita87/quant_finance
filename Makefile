UV ?= uv
PROJECTS := DataAnalysis DataEngineering DataScience FinancialMath GraphTheory LLM MachineLearning PortfolioTracking

.PHONY: help sync format lint test run run-data run-finance run-graph run-ml run-portfolio run-analysis run-llm

help:
	@echo "Available targets:"
	@echo "  make sync                         # Sync dependencies for all projects using uv"
	@echo "  make format                       # Run ruff formatter across projects"
	@echo "  make lint                         # Run ruff lint checks across projects"
	@echo "  make test                         # Run pytest for projects that define tests"
	@echo "  make run project=DIR module=...   # Run a module via uv (optional args=...)"
	@echo "  make run-analysis                 # Shortcut for DataAnalysis (Ecran Consulting)"
	@echo "  make run-data [args=...]          # Shortcut for DataScience CLI"
	@echo "  make run-finance                  # Shortcut for FinancialMath demo"
	@echo "  make run-graph module=...         # Shortcut for GraphTheory modules"
	@echo "  make run-llm                      # Shortcut for LLM (Patent MCP Server)"
	@echo "  make run-ml [args=...]            # Shortcut for MachineLearning example"
	@echo "  make run-portfolio                # Shortcut for PortfolioTracking CLI"

sync:
	@for project in $(PROJECTS); do \
		echo "Syncing $$project"; \
		( cd $$project && $(UV) sync ); \
	done

format:
	@for project in $(PROJECTS); do \
		echo "Formatting $$project"; \
		( cd $$project && $(UV) run ruff format . ); \
	done

lint:
	@for project in $(PROJECTS); do \
		echo "Linting $$project"; \
		( cd $$project && $(UV) run ruff check . ); \
	done

test:
	@for project in $(PROJECTS); do \
		if [ -d $$project/tests ]; then \
			echo "Testing $$project"; \
			( cd $$project && $(UV) run pytest ) || exit $$?; \
		else \
			echo "Skipping tests for $$project (no tests directory)"; \
		fi; \
	done

args ?=

run:
	@if [ -z "$(project)" ] || [ -z "$(module)" ]; then \
		echo "Usage: make run project=ProjectDir module=python.module [args='--flags']"; \
		exit 1; \
	fi
	@echo "Running in $(project): python -m $(module) $(args)"
	@cd $(project) && $(UV) run python -m $(module) $(args)

run-analysis:
	@$(MAKE) -s run project=DataAnalysis module=ecran_consulting.main args="$(args)"

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

run-llm:
	@$(MAKE) -s run project=LLM module=patent_mcp_server.main args="$(args)"

run-ml:
	@$(MAKE) -s run project=MachineLearning module=digit_genetics.mnist_genetic_algorithm args="$(args)"

run-portfolio:
	@$(MAKE) -s run project=PortfolioTracking module=portfolio_tracking.portfolio args="$(args)"

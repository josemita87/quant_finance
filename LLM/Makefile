# USPTO Patent MCP Server Makefile

.PHONY: help install run test clean

.DEFAULT_GOAL := help

help: ## Show available commands
	@echo "USPTO Patent MCP Server"
	@echo "======================="
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

run: ## Run the interactive chat server
	poetry run python -m src.main

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

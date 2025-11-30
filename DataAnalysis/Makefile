# ECRAN Strategic Consulting - Makefile
# Convenient commands to run the complete visualization pipeline

.PHONY: help install phase1 phase2 all clean clean-all view-output test

# Default target - show help
help:
	@echo "ECRAN Strategic Consulting - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "  make install      Install all dependencies"
	@echo "  make all          Run complete pipeline (Phase 1 + Phase 2)"
	@echo "  make phase1       Generate strategic narratives (3 charts)"
	@echo "  make phase2       Generate ML analysis (4 charts + data)"
	@echo "  make clean        Remove generated outputs (keep data)"
	@echo "  make clean-all    Remove all generated files including data"
	@echo "  make view-output  List all generated visualizations"
	@echo "  make test         Run a quick test of both phases"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install && make all"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	uv sync
	@echo "✅ Installation complete!"
	@echo ""

# Run Phase 1: Strategic Narratives
phase1:
	@echo "🎯 Running Phase 1: Strategic Narratives"
	@echo "========================================"
	uv run python main.py
	@echo ""
	@echo "✅ Phase 1 complete: 3 strategic charts generated"
	@echo ""

# Run Phase 2: ML Analysis
phase2:
	@echo "🧠 Running Phase 2: ML Analysis & Customer Insights"
	@echo "===================================================="
	uv run python main_phase2.py
	@echo ""
	@echo "✅ Phase 2 complete: Survey data + 4 ML charts generated"
	@echo ""

# Run complete pipeline
all: phase1 phase2
	@echo "=================================================="
	@echo "🎉 COMPLETE PIPELINE FINISHED"
	@echo "=================================================="
	@echo ""
	@echo "📊 Generated Assets:"
	@echo "  • 7 visualizations in output/"
	@echo "  • Survey data in data/survey_data.csv"
	@echo ""
	@echo "View outputs with: make view-output"
	@echo ""

# Clean generated outputs (keep data)
clean:
	@echo "🧹 Cleaning generated outputs..."
	@rm -f output/*.png
	@echo "✅ Output directory cleaned (data preserved)"
	@echo ""

# Clean all generated files including data
clean-all:
	@echo "🧹 Cleaning all generated files..."
	@rm -f output/*.png
	@rm -f data/survey_data.csv
	@rm -rf data/__pycache__ src/__pycache__
	@echo "✅ All generated files removed"
	@echo ""

# View generated outputs
view-output:
	@echo "📁 Generated Visualizations:"
	@echo "============================"
	@ls -lh output/*.png 2>/dev/null | awk '{printf "  • %-40s %6s\n", $$9, $$5}' || echo "  No visualizations generated yet. Run 'make all' first."
	@echo ""
	@if [ -f data/survey_data.csv ]; then \
		echo "📊 Survey Data:"; \
		echo "  • data/survey_data.csv" $$(ls -lh data/survey_data.csv | awk '{print $$5}'); \
		echo ""; \
	fi

# Quick test - run with minimal output
test:
	@echo "🧪 Testing pipeline..."
	@uv run python main.py > /dev/null 2>&1 && echo "✅ Phase 1 OK" || echo "❌ Phase 1 FAILED"
	@uv run python main_phase2.py > /dev/null 2>&1 && echo "✅ Phase 2 OK" || echo "❌ Phase 2 FAILED"
	@echo ""
	@echo "Generated files:"
	@ls -1 output/*.png 2>/dev/null | wc -l | awk '{printf "  • Visualizations: %d/7\n", $$1}'
	@if [ -f data/survey_data.csv ]; then echo "  • Survey data: ✓"; else echo "  • Survey data: ✗"; fi
	@echo ""

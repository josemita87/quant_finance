# Machine Learning

## Overview
Machine learning experiments outside of finance. The showcased module implements a simple genetic algorithm that promotes MNIST digits deemed prototypical by a KD-Tree density heuristic.

## Setup
```bash
uv sync
```

## Usage
```bash
uv run python -q - <<'PY'
from digit_genetics.mnist_genetic_algorithm import (
    GeneticAlgorithmParameters,
    download_mnist,
    evolve,
    kdtree_density_flags,
)

images, labels = download_mnist(200)
flags = kdtree_density_flags(images, labels)
params = GeneticAlgorithmParameters(generations=10, population_size=12)
_, _, _, history = evolve(images, labels, flags, params)
print("Best fitness per generation:", history)
PY
```

## Structure
```
MachineLearning/
├── src/digit_genetics/
│   ├── __init__.py
│   └── mnist_genetic_algorithm.py
├── pyproject.toml
└── README.md
```

## Highlights
- MNIST download helper with optional PNG export for quick inspection
- Histogram visualisation utilities for intensity distributions
- KD-Tree density heuristic to promote representative samples
- Minimal genetic algorithm with selection, crossover, and mutation

## Related Projects
- [DataScience](../DataScience/README.md)
- [GraphTheory](../GraphTheory/README.md)

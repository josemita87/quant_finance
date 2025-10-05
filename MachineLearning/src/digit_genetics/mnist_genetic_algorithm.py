"""Genetic algorithm exploration on MNIST digits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from sklearn.datasets import fetch_openml

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_mnist(sample_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Download a subset of MNIST digits for local experimentation.

    Args:
        sample_size: Number of images to download.

    Returns:
        Tuple of ``(images, labels)`` where images are flattened arrays.
    """

    mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
    images, labels = mnist.data[:sample_size], mnist.target.astype(int)[:sample_size]
    return images, labels


def save_digit_images(images: np.ndarray, labels: np.ndarray) -> None:
    """Persist images to ``DATA_DIR`` for quick inspection."""

    for index, (image, label) in enumerate(zip(images, labels)):
        path = DATA_DIR / f"digit_{label}_{index}.png"
        Image.fromarray(image.reshape(28, 28).astype(np.uint8)).save(path)


def crop_and_center(image: np.ndarray) -> np.ndarray:
    """Crop surrounding whitespace and re-centre a digit on a 28×28 canvas."""

    rows = np.any(image > 0, axis=1)
    cols = np.any(image > 0, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    if y_indices.size == 0 or x_indices.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    cropped = image[y_indices[0] : y_indices[-1] + 1, x_indices[0] : x_indices[-1] + 1]
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - cropped.shape[0]) // 2
    x_offset = (28 - cropped.shape[1]) // 2
    canvas[y_offset : y_offset + cropped.shape[0], x_offset : x_offset + cropped.shape[1]] = cropped
    return canvas


def plot_histograms(images: np.ndarray, labels: np.ndarray, samples_per_digit: int = 1) -> None:
    """Plot log-normalised histograms for each digit."""

    unique_labels = np.unique(labels)
    plt.figure(figsize=(12, 6))
    for label in unique_labels:
        idx = np.where(labels == label)[0][:samples_per_digit]
        for sample in idx:
            histogram, _ = np.histogram(images[sample], bins=64, range=(0, 255))
            histogram = np.where(histogram == 0, 1e-5, histogram)
            plt.plot(np.log(histogram), label=f"Digit {label}")
    plt.xlabel("Pixel intensity bin")
    plt.ylabel("Log count")
    plt.legend()
    plt.tight_layout()


def kdtree_density_flags(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mark samples considered dense (likely prototypical) using KDTree distances."""

    flattened = images.reshape(images.shape[0], -1)
    flags = np.zeros(flattened.shape[0], dtype=int)
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if idx.size < 6:
            continue
        tree = KDTree(flattened[idx])
        distances, _ = tree.query(flattened[idx], k=6)
        avg_distance = distances[:, 1:].mean(axis=1)
        threshold = np.median(avg_distance)
        flags[idx] = (avg_distance <= threshold).astype(int)
    return flags


@dataclass
class GeneticAlgorithmParameters:
    """Configuration for the simple genetic algorithm."""

    population_size: int = 30
    individual_length: int = 20
    selection_fraction: float = 0.2
    mutation_probability: float = 0.05
    generations: int = 50


def initialise_population(
    images: np.ndarray,
    labels: np.ndarray,
    flags: np.ndarray,
    params: GeneticAlgorithmParameters,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an initial population of sampled images."""

    generator = rng or np.random.default_rng()
    indices = generator.integers(0, images.shape[0], size=(params.population_size, params.individual_length))
    return images[indices], labels[indices], flags[indices]


def fitness_function(flags: np.ndarray) -> np.ndarray:
    """Assign higher fitness to individuals containing many dense digits."""

    return flags.sum(axis=1)


def select_elite(population: np.ndarray, labels: np.ndarray, flags: np.ndarray, scores: np.ndarray, params: GeneticAlgorithmParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the top-performing individuals based on fitness."""

    elite_count = max(1, int(params.population_size * params.selection_fraction))
    order = np.argsort(scores)[-elite_count:][::-1]
    return population[order], labels[order], flags[order]


def crossover(
    parents: Tuple[np.ndarray, np.ndarray, np.ndarray],
    params: GeneticAlgorithmParameters,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate offspring using single-point crossover."""

    parent_a, parent_b = rng.integers(0, parents[0].shape[0], size=2)
    split = rng.integers(1, params.individual_length)
    def combine(array: np.ndarray) -> np.ndarray:
        return np.concatenate((array[parent_a, :split], array[parent_b, split:]))
    images = np.stack([combine(parents[0]) for _ in range(params.population_size)])
    labels = np.stack([combine(parents[1]) for _ in range(params.population_size)])
    flags = np.stack([combine(parents[2]) for _ in range(params.population_size)])
    return images, labels, flags


def mutate(flags: np.ndarray, params: GeneticAlgorithmParameters, rng: np.random.Generator) -> np.ndarray:
    """Flip random flag values to inject diversity."""

    mask = rng.random(flags.shape) < params.mutation_probability
    mutated = flags.copy()
    mutated[mask] = 1 - mutated[mask]
    return mutated


def evolve(
    images: np.ndarray,
    labels: np.ndarray,
    flags: np.ndarray,
    params: GeneticAlgorithmParameters,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a simple genetic algorithm returning the best individual each generation."""

    generator = rng or np.random.default_rng()
    population_images, population_labels, population_flags = initialise_population(
        images, labels, flags, params, generator
    )
    history = []
    for _ in range(params.generations):
        scores = fitness_function(population_flags)
        elite = select_elite(population_images, population_labels, population_flags, scores, params)
        history.append(scores.max())
        offspring_images, offspring_labels, offspring_flags = crossover(elite, params, generator)
        population_images = offspring_images
        population_labels = offspring_labels
        population_flags = mutate(offspring_flags, params, generator)
    final_scores = fitness_function(population_flags)
    best_index = np.argmax(final_scores)
    return (
        population_images[best_index],
        population_labels[best_index],
        population_flags[best_index],
        np.array(history),
    )


__all__ = [
    "DATA_DIR",
    "GeneticAlgorithmParameters",
    "crop_and_center",
    "download_mnist",
    "evolve",
    "kdtree_density_flags",
    "plot_histograms",
    "save_digit_images",
]

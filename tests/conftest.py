"""Shared fixtures: a small synthetic stand-in for MNIST (no download needed)."""
import os
import sys

import numpy as np
import pytest

V2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v2")
sys.path.insert(0, V2)


def synthetic_digits(n_per_class, seed):
    """28x28 uint8 images: each class is a fixed random template plus pixel noise, so the classes are separable
    but not trivially so. Same layout as mnist.npz (x: uint8 images, y: int labels)."""
    templates = (np.random.default_rng(0).random((10, 28, 28)) < 0.25) * 200.0   # same classes for train and test
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.clip(templates[c] + rng.normal(0, 40, (n_per_class, 28, 28)), 0, 255)
                        for c in range(10)]).astype(np.uint8)
    y = np.repeat(np.arange(10), n_per_class)
    order = rng.permutation(len(y))
    return x[order], y[order]


@pytest.fixture
def fake_mnist():
    return {"train": synthetic_digits(40, 1), "test": synthetic_digits(10, 2)}

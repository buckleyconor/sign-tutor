"""Synthetic test fixtures."""
import numpy as np


def synthetic_hand(seed: int = 0, translate=(0, 0, 0), scale=1.0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1, 1, size=(21, 3)).astype(np.float32)
    base[0] = 0.0  # wrist at origin
    return base * scale + np.array(translate, dtype=np.float32)

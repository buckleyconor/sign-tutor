"""Landmark-space augmentation (applied at training time)."""
import numpy as np


def augment_one_hand(vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """vec: (63,) normalised landmark vector."""
    pts = vec.reshape(21, 3).copy()
    # Gaussian noise — MediaPipe is noisy in practice
    pts += rng.normal(0, 0.01, pts.shape).astype(np.float32)
    # In-plane rotation around Z axis (wrist tilt)
    theta = np.deg2rad(rng.uniform(-10, 10))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = pts @ R.T
    # Mirror flip (50%) — only safe for one-handed signs
    if rng.random() < 0.5:
        pts[:, 0] = -pts[:, 0]
    return pts.flatten()

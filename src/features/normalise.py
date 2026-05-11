import numpy as np


def normalise_one_hand(landmarks: np.ndarray) -> np.ndarray:
    """landmarks: (21, 3) array. Returns (63,) normalised vector.

    Normalisation:
    - Translate so wrist (landmark 0) is at origin
    - Scale so wrist-to-middle-finger-MCP (landmark 9) distance is 1.0
    - Flatten to (63,)
    """
    assert landmarks.shape == (21, 3)
    wrist = landmarks[0]
    centred = landmarks - wrist
    scale = np.linalg.norm(centred[9])
    if scale < 1e-6:
        scale = 1.0
    return (centred / scale).astype(np.float32).flatten()

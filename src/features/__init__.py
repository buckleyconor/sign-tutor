"""Feature builder — normalises hand landmarks into a model input vector."""

import numpy as np

from src.registry import Language
from .normalise import normalise_one_hand


def build_feature_vector(language: Language, detections) -> np.ndarray | None:
    """Build the model input vector for a language from hand detections.

    Args:
        language: A Language dataclass from the registry.
        detections: List of (handedness, landmarks_21x3) from HandTracker.

    Returns:
        A (63,) normalised landmark vector, or None if no hand was detected.
    """
    if not detections:
        return None
    return normalise_one_hand(detections[0][1])

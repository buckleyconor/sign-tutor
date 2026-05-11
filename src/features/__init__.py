import numpy as np
from src.registry import Language
from .normalise import normalise_one_hand
from .two_hand import build_two_hand_vector


def build_feature_vector(language: Language, detections) -> np.ndarray | None:
    """Dispatch to the correct feature builder based on language config."""
    if language.input_hands == 1:
        if not detections:
            return None
        # Take whichever hand was detected (handedness ignored for ASL/ISL)
        return normalise_one_hand(detections[0][1])
    elif language.input_hands == 2:
        return build_two_hand_vector(detections)
    raise ValueError(f"Unsupported input_hands: {language.input_hands}")

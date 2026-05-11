import numpy as np
from .normalise import normalise_one_hand


def build_two_hand_vector(detections, dominant: str = "Right") -> np.ndarray | None:
    """detections: list of (handedness, (21,3)) from HandTracker.
    Returns (126,) vector or None if both hands not present."""
    by_hand = {h: lm for h, lm in detections}
    if "Left" not in by_hand or "Right" not in by_hand:
        return None
    sub = "Left" if dominant == "Right" else "Right"
    dom_vec = normalise_one_hand(by_hand[dominant])
    sub_vec = normalise_one_hand(by_hand[sub])
    return np.concatenate([dom_vec, sub_vec])

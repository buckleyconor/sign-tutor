"""Unit tests for src/features/two_hand.py"""
import pytest
import numpy as np
from src.features.two_hand import build_two_hand_vector
from tests.fixtures.synthetic import synthetic_hand


def test_requires_both_hands():
    detections = [("Right", synthetic_hand(seed=1))]
    result = build_two_hand_vector(detections)
    assert result is None


def test_two_hand_output_shape():
    right = synthetic_hand(seed=1)
    left = synthetic_hand(seed=2)
    detections = [("Right", right), ("Left", left)]
    result = build_two_hand_vector(detections)
    assert result is not None
    assert result.shape == (126,)


def test_dominant_ordering():
    right = synthetic_hand(seed=1)
    left = synthetic_hand(seed=2)
    detections = [("Right", right), ("Left", left)]
    result_right_dom = build_two_hand_vector(detections, dominant="Right")
    result_left_dom = build_two_hand_vector(detections, dominant="Left")
    # Results should differ based on dominant hand ordering
    assert not np.allclose(result_right_dom, result_left_dom)

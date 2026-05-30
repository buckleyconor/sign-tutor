"""Unit tests for src/features/normalise.py"""
import pytest
import numpy as np
from src.features.normalise import normalise_one_hand
from tests.fixtures.synthetic import synthetic_hand


def test_output_shape():
    hand = synthetic_hand(seed=42)
    result = normalise_one_hand(hand)
    assert result.shape == (63,)
    assert result.dtype == np.float32


def test_translation_invariance():
    hand = synthetic_hand(seed=42)
    result1 = normalise_one_hand(hand)
    translated = hand + np.array([10.0, 20.0, 30.0], dtype=np.float32)
    result2 = normalise_one_hand(translated)
    np.testing.assert_allclose(result1, result2, atol=1e-5)


def test_scale_invariance():
    hand = synthetic_hand(seed=42)
    result1 = normalise_one_hand(hand)
    scaled = hand * 2.0
    result2 = normalise_one_hand(scaled)
    np.testing.assert_allclose(result1, result2, atol=1e-5)


def test_wrist_at_origin():
    hand = synthetic_hand(seed=42, translate=(5.0, 5.0, 5.0))
    result = normalise_one_hand(hand)
    wrist = result[:3]
    np.testing.assert_allclose(wrist, [0.0, 0.0, 0.0], atol=1e-6)


def test_zero_scale_handled():
    """Degenerate hand where wrist and middle MCP coincide."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[1:] = 1.0  # all other landmarks at same point
    result = normalise_one_hand(hand)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

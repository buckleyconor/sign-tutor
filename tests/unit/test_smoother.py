"""Unit tests for src/lesson/smoother.py"""
import pytest
from src.lesson.smoother import PredictionSmoother


def test_returns_none_until_warmup():
    s = PredictionSmoother(window=10)
    for i in range(4):
        s.update(0, 0.9)
    assert s.smoothed() is None


def test_modal_class_wins():
    s = PredictionSmoother(window=10)
    for _ in range(8):
        s.update(0, 0.9)  # class 0
    for _ in range(3):
        s.update(1, 0.95)  # class 1
    pred_idx, conf = s.smoothed()
    assert pred_idx == 0  # modal is 0 (8 votes vs 3)


def test_confidence_averages_modal_only():
    s = PredictionSmoother(window=10)
    for _ in range(6):
        s.update(0, 0.8)
    for _ in range(5):
        s.update(1, 0.99)
    pred_idx, conf = s.smoothed()
    assert pred_idx == 0
    assert pytest.approx(conf, abs=0.01) == 0.8


def test_window_eviction():
    s = PredictionSmoother(window=6)
    # Fill with class 0
    for _ in range(6):
        s.update(0, 0.5)
    # Add class 1 — oldest class 0 should be evicted
    for _ in range(6):
        s.update(1, 0.9)
    pred_idx, _ = s.smoothed()
    assert pred_idx == 1

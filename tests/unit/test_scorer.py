"""Unit tests for src/lesson/scorer.py"""
import pytest
import time
from unittest.mock import patch
from src.lesson.scorer import Light, TrafficLightScorer


TARGET = 0


def test_wrong_class_is_red():
    scorer = TrafficLightScorer(target_idx=TARGET)
    light, completed = scorer.evaluate(1, 0.99)
    assert light == Light.RED
    assert completed is False


def test_correct_low_conf_is_red():
    scorer = TrafficLightScorer(target_idx=TARGET)
    light, _ = scorer.evaluate(TARGET, 0.4)
    assert light == Light.RED


def test_correct_mid_conf_is_amber():
    scorer = TrafficLightScorer(target_idx=TARGET)
    light, _ = scorer.evaluate(TARGET, 0.65)
    assert light == Light.AMBER


def test_correct_high_conf_is_green():
    scorer = TrafficLightScorer(target_idx=TARGET)
    light, completed = scorer.evaluate(TARGET, 0.9)
    assert light == Light.GREEN
    assert completed is False


def test_completion_after_hold():
    scorer = TrafficLightScorer(target_idx=TARGET, hold_seconds=1.0)
    with patch("time.monotonic", side_effect=[0.0, 1.0]):
        scorer.evaluate(TARGET, 0.9)  # t=0, starts green timer
        light, completed = scorer.evaluate(TARGET, 0.9)  # t=1, should complete
        assert completed is True
        assert light == Light.GREEN


def test_amber_resets_green_timer():
    scorer = TrafficLightScorer(target_idx=TARGET, hold_seconds=1.0)
    with patch("time.monotonic", side_effect=[0.0, 0.5, 0.6]):
        scorer.evaluate(TARGET, 0.9)   # t=0, green
        scorer.evaluate(TARGET, 0.65)  # t=0.5, amber — resets timer
        light, completed = scorer.evaluate(TARGET, 0.9)  # t=0.6, green again
        assert completed is False  # not enough time since reset


def test_render_light():
    scorer = TrafficLightScorer(target_idx=TARGET)
    for light in [Light.RED, Light.AMBER, Light.GREEN]:
        html = scorer.render_light(light)
        assert "<div" in html
        assert "border-radius:50%" in html

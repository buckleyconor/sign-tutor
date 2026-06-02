"""Unit tests for src/lesson/scorer.py"""
import time
from unittest.mock import patch
from src.lesson.scorer import QualityScorer


TARGET = 0
GREEN_MIN = 0.90


def _scorer(**kwargs):
    return QualityScorer(target_idx=TARGET, green_min=GREEN_MIN, **kwargs)


def test_wrong_class_keeps_bar_at_zero():
    s = _scorer()
    s.update(pred_idx=1, conf=0.99)
    assert s._ema == 0.0
    assert not s.completed


def test_correct_low_conf_below_threshold():
    s = _scorer()
    s.update(pred_idx=TARGET, conf=0.40)
    assert s._ema < GREEN_MIN
    assert not s.completed


def test_correct_high_conf_reaches_full_green():
    s = _scorer()
    for _ in range(20):          # drive EMA to near-1.0
        s.update(pred_idx=TARGET, conf=1.0)
    assert s._ema >= GREEN_MIN


def test_completion_after_hold():
    s = _scorer(hold_seconds=1.0)
    s._ema = 1.0  # bypass EMA warm-up; test the hold timer only
    with patch("time.monotonic", side_effect=[0.0, 1.0, 1.0]):
        s.update(TARGET, 1.0)   # t=0 — starts green timer
        s.update(TARGET, 1.0)   # t=1
        assert s.completed      # t=1 — held >= 1.0s


def test_wrong_class_resets_green_timer():
    s = _scorer(hold_seconds=1.0)
    with patch("time.monotonic", side_effect=[0.0, 0.5, 0.6, 0.6]):
        s.update(TARGET, 1.0)   # t=0 — green timer starts
        s.update(1, 0.99)       # t=0.5 — wrong class resets timer
        s.update(TARGET, 1.0)   # t=0.6 — green again but not long enough
        assert not s.completed  # t=0.6 — only 0s since reset


def test_render_bar_returns_html():
    s = _scorer()
    s.update(TARGET, 0.5)
    html = s.render_bar()
    assert "<div" in html
    assert "%" in html


def test_render_bar_shows_zero_for_wrong_class():
    s = _scorer()
    s.update(pred_idx=1, conf=0.99)
    html = s.render_bar()
    assert "0%" in html


def test_render_bar_full_green_at_threshold():
    s = _scorer()
    for _ in range(40):
        s.update(TARGET, 1.0)
    html = s.render_bar()
    assert "width:100%" in html
    assert "hsl(120" in html

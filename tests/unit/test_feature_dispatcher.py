"""Unit tests for src/features/__init__.py (feature dispatcher)."""
import pytest
import numpy as np
from src.features import build_feature_vector
from src.registry import Language
from pathlib import Path
from tests.fixtures.synthetic import synthetic_hand


@pytest.fixture
def one_hand_lang():
    return Language(
        name="ASL", code="asl", input_hands=1,
        classes=[chr(i) for i in range(65, 91)],
        triton_model_name="asl_classifier",
        references_dir=Path("languages/asl/references"),
        notes={}
    )


def test_one_handed_no_detections(one_hand_lang):
    assert build_feature_vector(one_hand_lang, []) is None


def test_one_handed_single_detection(one_hand_lang):
    hand = synthetic_hand(seed=1)
    detections = [("Right", hand)]
    result = build_feature_vector(one_hand_lang, detections)
    assert result is not None
    assert result.shape == (63,)

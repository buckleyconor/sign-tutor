"""Unit tests for src/registry.py"""
import pytest
from pathlib import Path
import tempfile
import yaml

from src.registry import load_registry


@pytest.fixture
def lang_dir():
    """Create a temp directory with two language configs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        for code, hands in [("asl", 1), ("isl", 1)]:
            lang_dir = tmp / code
            lang_dir.mkdir(parents=True)
            cfg = {
                "name": f"{code.upper()} Sign Language",
                "code": code,
                "input_hands": hands,
                "classes": [chr(i) for i in range(65, 91)],  # A-Z
                "triton_model_name": f"{code}_classifier",
                "references_dir": "references",
                "notes": {},
            }
            (lang_dir / "config.yaml").write_text(yaml.dump(cfg))
            (lang_dir / "references").mkdir()

        yield tmp


def test_loads_supported_languages(lang_dir):
    registry = load_registry(lang_dir)
    assert set(registry.keys()) == {"asl", "isl"}


def test_asl_isl_one_handed(lang_dir):
    registry = load_registry(lang_dir)
    assert registry["asl"].input_hands == 1
    assert registry["isl"].input_hands == 1


def test_classes_have_26_letters(lang_dir):
    registry = load_registry(lang_dir)
    for code in ["asl", "isl"]:
        assert len(registry[code].classes) == 26
        assert registry[code].classes == [chr(i) for i in range(65, 91)]


def test_missing_config_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        registry = load_registry(Path(tmp))
        assert registry == {}

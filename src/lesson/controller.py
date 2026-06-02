"""Lesson controller — orchestrates the sign recognition flow."""
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from src.registry import Language, load_registry
from src.capture.hands import HandTracker
from src.features import build_feature_vector
from src.inference.triton_client import TritonClassifier
from src.lesson.smoother import PredictionSmoother
from src.lesson.scorer import QualityScorer


def load_thresholds(path: str = "configs/thresholds.yaml",
                    language_code: Optional[str] = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if language_code and language_code in cfg.get("overrides", {}):
        for k, v in cfg["overrides"][language_code].items():
            cfg[k] = v
    return cfg


class LessonController:
    def __init__(self, registry: Optional[dict[str, Language]] = None):
        self._registry = registry or load_registry()
        self._current_lang: Optional[Language] = None
        self._tracker = HandTracker()
        self._classifier: Optional[TritonClassifier] = None
        self._smoother: Optional[PredictionSmoother] = None
        self._scorer: Optional[QualityScorer] = None
        self._target_idx: int = 0
        self._thresholds = {}

    def switch_language(self, lang_code: str):
        lang = self._registry[lang_code]
        self._current_lang = lang
        self._classifier = TritonClassifier(model_name=lang.triton_model_name)
        self._thresholds = load_thresholds(language_code=lang_code)
        self._smoother = PredictionSmoother(
            window=self._thresholds.get("smoothing_window_frames", 15)
        )
        self.set_target(0)

    def set_target(self, idx: int):
        self._target_idx = idx
        self._scorer = QualityScorer(
            target_idx=idx,
            hold_seconds=self._thresholds.get("hold_seconds_for_complete", 1.0),
            green_min=self._thresholds.get("green_min_confidence", 0.90),
        )
        self._smoother = PredictionSmoother(
            window=self._thresholds.get("smoothing_window_frames", 15)
        )

    def process_frame(self, rgb_frame: np.ndarray, lang_code: str) -> dict:
        """Process a single webcam frame (RGB, as provided by Gradio streaming).
        Returns a dict of UI updates."""
        lang = self._registry.get(lang_code)
        if not lang or self._classifier is None:
            return {"annotated": rgb_frame, "status": "Select a language"}

        # Flip horizontally so hand orientation matches training images
        # (webcam sends non-mirrored frames; Kaggle dataset was photographed mirrored)
        flipped = np.ascontiguousarray(rgb_frame[:, ::-1, :])
        detections = self._tracker.process(flipped)

        feat = build_feature_vector(lang, detections)

        if feat is None:
            hint = "Show your hand" if lang.input_hands == 1 else "Show both hands"
            return {"status": hint, "completed": False}

        # Infer
        logits = self._classifier.infer(feat)
        probs = np.exp(logits) / np.exp(logits).sum()
        pred_idx = int(logits.argmax())
        confidence = float(probs[pred_idx])

        # Smooth
        self._smoother.update(pred_idx, confidence)
        smoothed = self._smoother.smoothed()

        if smoothed is None:
            return {"status": "Collecting data...", "completed": False}

        smooth_idx, smooth_conf = smoothed

        # Score
        self._scorer.update(smooth_idx, smooth_conf)
        predicted_letter = lang.classes[smooth_idx]
        target_letter = lang.classes[self._target_idx]

        return {
            "status": (
                f"Predicted: **{predicted_letter}** ({smooth_conf:.0%}) | "
                f"Target: **{target_letter}**"
            ),
            "completed": self._scorer.completed,
            "confidence": smooth_conf,
        }

    def get_reference_image(self, lang_code: str, idx: int) -> Optional[str]:
        lang = self._registry.get(lang_code)
        if not lang:
            return None
        letter = lang.classes[idx]
        ref_path = lang.references_dir / f"{letter}.png"
        return str(ref_path) if ref_path.exists() else None

    def get_letter(self, lang_code: str, idx: int) -> str:
        return self._registry[lang_code].classes[idx]

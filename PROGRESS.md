# Progress Log — Sign Language Tutor

> **Last updated:** 2026-05-11  
> **Status:** Data acquisition starting — Kaggle API configured, ASL dataset next  
> **Environment:** macOS dev machine (tests run locally), GB10 container pending deployment

---

## Completed

### 1. Repository & Infrastructure

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Project overview, quick-start, spec links | ✅ |
| `.gitignore` | Excludes datasets, checkpoints, ONNX, TensorRT engines, pycache | ✅ |
| `docker-compose.yml` | Two-service stack: Triton server (GPU) + tutor-app (GPU + webcam) | ✅ |
| `Dockerfile` | NVIDIA PyTorch 25.03 container; installs mediapipe, gradio, tritonclient | ✅ |
| `configs/thresholds.yaml` | Amber/green confidence thresholds, smoothing window, per-language overrides | ✅ |

### 2. Language Registry

| File | Purpose | Status |
|------|---------|--------|
| `src/registry.py` | `Language` dataclass + `load_registry()` — reads `languages/*/config.yaml` | ✅ |
| `languages/asl/config.yaml` | One-handed, 26 classes, `asl_classifier`, notes: dynamic letters J, Z | ✅ |
| `languages/isl/config.yaml` | One-handed, 26 classes, `isl_classifier`, notes: dynamic letters J, X, Z | ✅ |

Registry supports arbitrary language additions via new `config.yaml` + Triton model — no code changes required.

### 3. Capture Module

| File | Purpose | Status |
|------|---------|--------|
| `src/capture/hands.py` | `HandTracker` wrapper around MediaPipe Hands; configurable max_hands, detection/tracking confidence; returns `(handedness, landmarks_21x3)` tuples | ✅ |

### 4. Feature Extraction

| File | Purpose | Status |
|------|---------|--------|
| `src/features/__init__.py` | `build_feature_vector(language, detections)` — normalises and returns (63,) feature vector | ✅ |
| `src/features/normalise.py` | `normalise_one_hand()` — translate-to-wrist, scale-to-middle-MCP, flatten to `(63,)` | ✅ |

**Normalisation guarantees:**
- Translation invariant (wrist always at origin)
- Scale invariant (wrist→middle-MCP distance always 1.0)
- Degenerate hands (zero distance) handled safely

### 5. Inference Module

| File | Purpose | Status |
|------|---------|--------|
| `src/inference/triton_client.py` | `TritonClassifier` — Triton HTTP client wrapper; `infer(x)` returns raw logits | ✅ |
| `triton_repo/asl_classifier/config.pbtxt` | TensorRT plan, input `(63,)`, output `(26,)`, GPU | ✅ |
| `triton_repo/isl_classifier/config.pbtxt` | TensorRT plan, input `(63,)`, output `(26,)`, GPU | ✅ |

### 6. Lesson Logic

| File | Purpose | Status |
|------|---------|--------|
| `src/lesson/smoother.py` | `PredictionSmoother` — rolling window (default 15 frames), modal class vote, modal-only confidence average | ✅ |
| `src/lesson/scorer.py` | `TrafficLightScorer` — RED/AMBER/GREEN logic based on target match + confidence thresholds; hold-time completion detection; HTML render method | ✅ |
| `src/lesson/controller.py` | `LessonController` — orchestrates full pipeline: hand tracking → feature build → Triton inference → smoothing → scoring; `process_frame()` returns annotated image + UI state; `switch_language()`, `set_target()`, `get_reference_image()` | ✅ |

### 7. UI (Gradio)

| File | Purpose | Status |
|------|---------|--------|
| `src/ui/app.py` | Gradio Blocks app: language dropdown, live webcam stream with hand landmark overlay, reference image display, traffic-light indicator, progress dots, skip/next buttons | ✅ |

### 8. Training Pipeline

| File | Purpose | Status |
|------|---------|--------|
| `training/extract_landmarks.py` | Extract landmarks from image directories (Kaggle-style `letter/*.png` structure) → CSV with `label, source, frame_id, features` | ✅ |
| `training/model_one_hand.py` | `OneHandClassifier` — Linear(63,128) → ReLU → Dropout(0.3) → Linear(128,64) → ReLU → Linear(64,C) | ✅ |
| `training/augment.py` | `augment_one_hand()` — Gaussian noise, in-plane rotation (±10°), 50% mirror flip | ✅ |
| `training/train_classifier.py` | Full training loop: CSV loading, random split, augmentation, AdamW + cosine annealing, best-by-val-accuracy checkpointing, per-epoch CSV logging | ✅ |
| `training/export_onnx.py` | Convert PyTorch checkpoint → ONNX with dynamic batch axis | ✅ |
| `training/build_engine.sh` | `trtexec` wrapper: ONNX → TensorRT FP16 engine with min/opt/max batch shapes | ✅ |
| `training/capture.py` | Live webcam capture script: `--lang isl --letter J --duration 10` — drops frames with no hand detected, appends to landmarks CSV | ✅ |
| `training/check_quality.py` | Data quality gates: correct feature dimension, class balance ratio (min/max > 0.5), no NaN/Inf, ≥ 2 sources | ✅ |

### 9. Tests

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_registry.py` | Loads 2 languages, one-handed flags, 26-letter classes, empty dir handling | ✅ 4 assertions |
| `tests/unit/test_normalise.py` | Output shape, translation invariance, scale invariance, wrist at origin, zero-scale safety | ✅ 5 assertions |
| `tests/unit/test_feature_dispatcher.py` | One-handed no detections, one-handed single detection, unsupported input_hands raises ValueError | ✅ 3 assertions |
| `tests/unit/test_smoother.py` | Warmup period returns None, modal class wins, confidence averages modal only, window eviction | ✅ 4 assertions |
| `tests/unit/test_scorer.py` | Wrong class = RED, low confidence = RED, mid confidence = AMBER, high confidence = GREEN, completion after hold time, amber resets green timer, HTML rendering | ✅ 7 assertions |

**All 6 test modules pass** (tested on macOS with Python 3.14, numpy 2.4, pyyaml 6.0).

### 10. Triton Repository Skeleton

| Directory | Model | Input | Output | Platform |
|-----------|-------|-------|--------|----------|
| `triton_repo/asl_classifier/1/` | `config.pbtxt` | `(63,)` | `(26,)` | `onnxruntime_onnx` |
| `triton_repo/isl_classifier/1/` | `config.pbtxt` | `(63,)` | `(26,)` | `onnxruntime_onnx` |

Model weights (`model.plan`) are not yet built — this happens on the GB10 after training.

---

## Not Yet Done (Next Steps)

| Item | Spec Reference | Blocking On |
|------|----------------|-------------|
| SSH into GB10 (`promaxgb10`) and verify CUDA/container access | Spec 2, §1 | — |
| Download ASL Kaggle dataset | Spec 4, §2 | ✅ Kaggle API key ready | Dataset pending download |
| Download Rasband test set (held-out validation) | Spec 4, §3.3 | — | Dataset pending download |
| Clone DCU ISL-HS dataset | Spec 4, §3 | — |
| Run `extract_landmarks.py` on ASL dataset | Spec 2, §3 | Dataset available |
| Run `train_classifier.py` on ASL landmarks CSV | Spec 2, §5 | Extracted CSV, GPU container |
| Export ONNX → TensorRT engine for ASL | Spec 2, §6 | Trained checkpoint |
| Triton smoke test (send inference request) | Spec 3, §2 | Engine deployed |
| Gradio end-to-end demo (webcam → recognition) | Spec 3, §3 | Triton running |
| ISL self-capture via `training/capture.py` | Spec 4, §4 | Webcam, trained ISL model |
| Integration tests (real Triton, real webcam frame) | Spec 3, §2 | GB10 environment |

---

## Architecture Summary

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Webcam     │────▶│  HandTracker     │────▶│ Feature Extractor│
│  (OpenCV)    │     │  (MediaPipe)     │     │ (normalise)      │
└──────────────┘     │  returns 21×3 pts│     │ outputs (63)      │
                     └──────────────────┘                     ▼
                                   ┌─────────────────────────┐
                                   │  Triton Inference Server │
                                   │  (TensorRT plan, GPU)    │
                                   │  returns 26-class logits  │
                                   └───────────┬─────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Lesson Controller               │
                              │                                  │
                              │  PredictionSmoother (15-frame    │
                              │  rolling window, modal vote) ──▶ TrafficLightScorer ──▶ RED/AMBER/GREEN +
                              │                                  │     completion signal
                              └──────────────────────────────────┘
                                               │
                              ┌────────────────▼────────────────────────┐
                              │  Gradio UI                               │
                              │  ┌─────────┐  ┌─────────┐               │
                              │  │ Live    │  │ Target  │               │
                              │  │ Feed +  │  │ Sign +  │               │
                              │  │ overlay │  │ light   │  Language: [ASL v]
                              │  └─────────┘  └─────────┘  Letter: [A]
                              └────────────────────────────────────────┘

  Language registry (config.yaml) controls:
    • input_hands (1 or 2)
    • triton_model_name (which Triton model to call)
    • classes[] (letter mapping)
    • references_dir (target image location)
    • threshold overrides in configs/thresholds.yaml
```

## File Count

| Category | Files |
|----------|-------|
| Source code (`src/`) | 10 |
| Training pipeline (`training/`) | 9 |
| Tests (`tests/`) | 9 |
| Config / infra | 8 |
| Triton repo | 3 |
| Documentation | 5 (4 specs + 1 this file) |
| **Total** | **44** |

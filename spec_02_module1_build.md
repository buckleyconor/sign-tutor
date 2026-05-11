# Module 1 — Detailed Build Spec

**Multi-Language Fingerspelling Alphabet**
**Spec 2 of 3**

---

## 1. Scope of this spec

This document is the implementation plan for Module 1 — fingerspelled alphabet recognition for ASL, ISL, and BSL. It assumes the architecture in Spec 1. Each section below maps to a single, scoped deliverable. Every section ends with an exit criterion — something concrete you can demonstrate before moving on.

---

## 2. Repository Layout

```
sign-tutor/
├── README.md
├── docker-compose.yml          # triton + tutor-app
├── configs/
│   └── thresholds.yaml         # traffic-light cutoffs, smoothing window
├── languages/
│   ├── asl/
│   │   ├── config.yaml
│   │   ├── references/         # A.png ... Z.png
│   │   └── models/             # populated by training
│   ├── isl/
│   │   └── ...
│   └── bsl/
│       └── ...
├── src/
│   ├── capture/                # webcam + MediaPipe wrapper
│   ├── features/               # landmark normalisation, packing
│   ├── inference/              # Triton client wrapper
│   ├── lesson/                 # lesson controller, scoring
│   ├── ui/                     # Gradio app
│   └── registry.py             # language registry loader
├── training/
│   ├── extract_landmarks.py
│   ├── train_classifier.py
│   ├── export_onnx.py
│   └── build_engine.sh
├── triton_repo/                # Triton model repository
│   ├── asl_classifier/
│   │   ├── config.pbtxt
│   │   └── 1/model.plan
│   └── ...
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

> **Exit criterion:** empty repo with this layout committed and pushed.

---

## 3. Environment Setup

### 3.1 Containers

Two containers, orchestrated by `docker-compose`:

```yaml
# docker-compose.yml (excerpt)
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:25.03-py3
    runtime: nvidia
    ports: ["8000:8000", "8001:8001", "8002:8002"]
    volumes: ["./triton_repo:/models"]
    command: tritonserver --model-repository=/models --model-control-mode=poll

  tutor-app:
    build: .
    image: sign-tutor:latest
    runtime: nvidia
    ports: ["7860:7860"]
    devices: ["/dev/video0:/dev/video0"]
    depends_on: ["triton"]
```

### 3.2 tutor-app Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.03-py3
RUN pip install --no-cache-dir \
    mediapipe \
    opencv-python-headless \
    gradio \
    tritonclient[http,grpc] \
    pyyaml \
    pytest pytest-cov
WORKDIR /app
COPY . /app
CMD ["python", "-m", "src.ui.app"]
```

> **Note:** `opencv-python-headless` avoids GUI/X11 dependencies inside the container; Gradio handles all UI rendering in the browser.

> **Exit criterion:** `docker-compose up` brings both containers to a healthy state; Triton's `/v2/health/ready` returns 200; Gradio loads on `http://promaxgb10.local:7860`.

---

## 4. Language Registry

This is the keystone of the multi-language design. Every other component reads from the registry.

### 4.1 Per-language config schema

```yaml
# languages/asl/config.yaml
name: "American Sign Language"
code: "asl"
input_hands: 1            # 1 for ASL/ISL, 2 for BSL
classes: ["A","B","C","D","E","F","G","H","I","J","K","L","M",
          "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
triton_model_name: "asl_classifier"
references_dir: "references"
notes:
  dynamic_letters: ["J", "Z"]   # captured as static snapshots
  attribution: "Reference images: <source>"
```

### 4.2 Registry loader

```python
# src/registry.py
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Language:
    name: str
    code: str
    input_hands: int
    classes: list[str]
    triton_model_name: str
    references_dir: Path
    notes: dict

def load_registry(root: Path = Path("languages")) -> dict[str, Language]:
    registry = {}
    for cfg_path in root.glob("*/config.yaml"):
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        lang_dir = cfg_path.parent
        registry[data["code"]] = Language(
            name=data["name"],
            code=data["code"],
            input_hands=data["input_hands"],
            classes=data["classes"],
            triton_model_name=data["triton_model_name"],
            references_dir=lang_dir / data["references_dir"],
            notes=data.get("notes", {}),
        )
    return registry
```

> **Exit criterion:** `load_registry()` returns three Language objects (asl, isl, bsl) with the expected fields. Unit-tested.

---

## 5. Capture & Hand Tracking

### 5.1 MediaPipe wrapper

```python
# src/capture/hands.py
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_hands: int = 2,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_frame: np.ndarray):
        """Returns list of (handedness, landmarks_array) tuples.
           landmarks_array is shape (21, 3)."""
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return []
        out = []
        for hand_idx, hand_lms in enumerate(result.multi_hand_landmarks):
            handedness = result.multi_handedness[hand_idx].classification[0].label
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            out.append((handedness, arr))
        return out
```

### 5.2 Why max_hands = 2 always

Even though ASL/ISL only need one hand, we always run with `max_hands=2`. The cost is negligible and it lets the same capture layer serve all three languages without switching modes mid-session.

---

## 6. Feature Layer

### 6.1 Why normalise

Raw MediaPipe coordinates depend on where the hand is in the frame and how big it appears. To make the classifier robust to position and distance, we normalise so every input vector represents the same hand-shape in the same canonical space.

### 6.2 Single-hand normalisation (ASL/ISL)

- Translate so the wrist (landmark 0) is at the origin.
- Scale so the distance from wrist (0) to middle-finger MCP (9) is 1.0.
- Flatten (21, 3) → (63,) float32 vector.

```python
# src/features/normalise.py
import numpy as np

def normalise_one_hand(landmarks: np.ndarray) -> np.ndarray:
    """landmarks: (21, 3) array. Returns (63,) normalised vector."""
    assert landmarks.shape == (21, 3)
    wrist = landmarks[0]
    centred = landmarks - wrist
    scale = np.linalg.norm(centred[9])
    if scale < 1e-6:
        scale = 1.0
    return (centred / scale).astype(np.float32).flatten()
```

### 6.3 Two-hand feature builder (BSL)

BSL fingerspelling is two-handed and asymmetric — there is a "dominant" and a "subordinate" hand. We use MediaPipe's handedness label to consistently order them.

```python
# src/features/two_hand.py
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
```

### 6.4 Feature dispatcher

```python
# src/features/__init__.py
import numpy as np
from src.registry import Language
from .normalise import normalise_one_hand
from .two_hand import build_two_hand_vector

def build_feature_vector(language: Language, detections) -> np.ndarray | None:
    if language.input_hands == 1:
        if not detections:
            return None
        # Take whichever hand was detected (handedness ignored for ASL/ISL)
        return normalise_one_hand(detections[0][1])
    elif language.input_hands == 2:
        return build_two_hand_vector(detections)
    raise ValueError(f"Unsupported input_hands: {language.input_hands}")
```

> **Exit criterion:** feature builder returns the right shape per language; unit tests cover translation/scale invariance.

---

## 7. Training Pipeline

### 7.1 Dataset extraction

Source images go through MediaPipe to produce CSV rows of `(label, 63 floats)` for one-handed languages or `(label, 126 floats)` for BSL. Done once, then reused across many training runs.

```bash
# training/extract_landmarks.py (skeleton)
python extract_landmarks.py --src datasets/asl_alphabet \
                            --dst languages/asl/landmarks.csv \
                            --hands 1
```

### 7.2 Classifier (one-handed)

```python
# training/model_one_hand.py
import torch.nn as nn

class OneHandClassifier(nn.Module):
    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)
```

### 7.3 Classifier (two-handed)

```python
# training/model_two_hand.py
import torch.nn as nn

class TwoHandClassifier(nn.Module):
    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(126, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)
```

### 7.4 Training script outline

- Load CSV, split 80/10/10 train/val/test (stratified by class).
- Train for ~50 epochs, batch size 256, Adam lr=1e-3, cross-entropy loss.
- Augmentation: small Gaussian noise on coordinates (sigma=0.01) and ±5° rotations in the XY plane.
- Save best checkpoint by validation accuracy.
- Report per-class precision/recall and confusion matrix on test set.

### 7.5 ONNX export

```python
# training/export_onnx.py (excerpt)
dummy = torch.randn(1, 63)  # or (1, 126) for BSL
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
```

### 7.6 TensorRT engine build

```bash
# training/build_engine.sh
trtexec --onnx=model.onnx \
        --saveEngine=model.plan \
        --fp16 \
        --minShapes=input:1x63 \
        --optShapes=input:32x63 \
        --maxShapes=input:64x63
```

> We build the engine on the GB10 itself — TensorRT engines are tied to GPU architecture, and the GB10 is sm_121 (Blackwell, low-power variant). Engines built on an H100 won't run here, and vice versa. This is normal; document it for future you.

> **Exit criterion:** per-language `model.plan` file exists; `trtexec` validation passes; standalone Python harness can run inference on it.

---

## 8. Triton Deployment

### 8.1 Model repository layout

```
triton_repo/
├── asl_classifier/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan
├── isl_classifier/
│   ├── config.pbtxt
│   └── 1/model.plan
└── bsl_classifier/
    ├── config.pbtxt
    └── 1/model.plan
```

### 8.2 config.pbtxt (one-handed)

```
name: "asl_classifier"
platform: "tensorrt_plan"
max_batch_size: 64
input  [ { name: "input",  data_type: TYPE_FP32, dims: [63] } ]
output [ { name: "output", data_type: TYPE_FP32, dims: [26] } ]
instance_group [ { count: 1, kind: KIND_GPU } ]
```

### 8.3 config.pbtxt (BSL)

```
name: "bsl_classifier"
platform: "tensorrt_plan"
max_batch_size: 64
input  [ { name: "input",  data_type: TYPE_FP32, dims: [126] } ]
output [ { name: "output", data_type: TYPE_FP32, dims: [26] } ]
instance_group [ { count: 1, kind: KIND_GPU } ]
```

### 8.4 Triton client wrapper

```python
# src/inference/triton_client.py
import numpy as np
import tritonclient.http as httpclient

class TritonClassifier:
    def __init__(self, url: str = "triton:8000", model_name: str = "asl_classifier"):
        self._client = httpclient.InferenceServerClient(url=url)
        self._model_name = model_name

    def infer(self, x: np.ndarray) -> np.ndarray:
        """x: (D,) or (B, D). Returns (num_classes,) softmax-able logits."""
        if x.ndim == 1:
            x = x[None, :]
        inp = httpclient.InferInput("input", x.shape, "FP32")
        inp.set_data_from_numpy(x.astype(np.float32))
        resp = self._client.infer(self._model_name, [inp])
        return resp.as_numpy("output")[0]
```

> **Exit criterion:** from a Python REPL inside `tutor-app`, calling `TritonClassifier` on a known-good landmark vector returns the correct letter with high confidence.

---

## 9. Scoring & Smoothing

### 9.1 Smoothing

```python
# src/lesson/smoother.py
from collections import deque, Counter
import numpy as np

class PredictionSmoother:
    def __init__(self, window: int = 15):
        self._window = window
        self._preds = deque(maxlen=window)
        self._confs = deque(maxlen=window)

    def update(self, pred_idx: int, confidence: float):
        self._preds.append(pred_idx)
        self._confs.append(confidence)

    def smoothed(self) -> tuple[int, float] | None:
        if len(self._preds) < self._window // 2:
            return None
        modal, count = Counter(self._preds).most_common(1)[0]
        avg_conf = float(np.mean(
            [c for p, c in zip(self._preds, self._confs) if p == modal]
        ))
        return modal, avg_conf
```

### 9.2 Traffic-light scorer

```python
# src/lesson/scorer.py
from enum import Enum
import time

class Light(Enum):
    RED = "red"; AMBER = "amber"; GREEN = "green"

class TrafficLightScorer:
    def __init__(self, target_idx: int, hold_seconds: float = 1.0,
                 amber_min: float = 0.50, green_min: float = 0.80):
        self._target = target_idx
        self._hold = hold_seconds
        self._amber_min = amber_min
        self._green_min = green_min
        self._green_since: float | None = None

    def evaluate(self, pred_idx: int, conf: float) -> tuple[Light, bool]:
        """Returns (light, completed)."""
        now = time.monotonic()
        if pred_idx != self._target or conf < self._amber_min:
            self._green_since = None
            return Light.RED, False
        if conf < self._green_min:
            self._green_since = None
            return Light.AMBER, False
        # Green territory
        if self._green_since is None:
            self._green_since = now
        completed = (now - self._green_since) >= self._hold
        return Light.GREEN, completed
```

> **Exit criterion:** with a synthetic stream of `(pred, conf)` values, scorer transitions through RED → AMBER → GREEN → completed at the right thresholds. Unit-tested.

---

## 10. Gradio UI

### 10.1 Layout

Gradio's `gr.Blocks` API gives precise control over layout. Use a top Row for the language/lesson selector, then a Row containing two Columns for the split.

```python
# src/ui/app.py (skeleton)
import gradio as gr
from src.registry import load_registry
from src.lesson.controller import LessonController

LANGS = load_registry()

def build_app():
    controller = LessonController(LANGS)
    with gr.Blocks(title="Sign Language Tutor") as demo:
        with gr.Row():
            lang = gr.Dropdown(
                choices=[(l.name, l.code) for l in LANGS.values()],
                value="asl", label="Language")
            lesson = gr.Dropdown(
                choices=[("Alphabet", "alphabet")],
                value="alphabet", label="Lesson")
        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(sources=["webcam"], streaming=True,
                               label="Live feed")
            with gr.Column(scale=1):
                target = gr.Image(label="Sign this letter", interactive=False)
                light = gr.HTML(value=controller.render_light())
                status = gr.Markdown()
                with gr.Row():
                    skip_btn = gr.Button("Skip")
                    next_btn = gr.Button("Next letter")
        cam.stream(controller.on_frame,
                   inputs=[cam, lang],
                   outputs=[cam, target, light, status])
        # ... wire up button handlers
    return demo

if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)
```

### 10.2 Light rendering

```python
def render_light(light: Light) -> str:
    colours = {Light.RED: "#E74C3C",
               Light.AMBER: "#F39C12",
               Light.GREEN: "#27AE60"}
    return (f'<div style="width:60px;height:60px;'
            f'border-radius:50%;background:{colours[light]};'
            f'margin:auto;box-shadow:0 0 12px {colours[light]};"></div>')
```

### 10.3 Performance considerations

- Resize incoming webcam frames to 480p before MediaPipe — full HD is wasteful.
- Drop frames if the inference call is still in flight; never queue.
- Reuse the `HandTracker` and `TritonClassifier` instances across frames.
- Pre-load reference images into memory at startup (small enough to fit easily).

> **Exit criterion:** end-to-end demo with ASL — load page, select ASL alphabet, sign a letter, see GREEN, advance. 25+ FPS sustained.

---

## 11. Adding ISL (one-handed extension)

Adding ISL is the moment of truth for the framework. If we built the registry correctly, this is a data-only task.

- Capture ~200 images per letter using webcam (or use a small open ISL dataset if one is available).
- Run `extract_landmarks.py` to produce `languages/isl/landmarks.csv`.
- Train a `OneHandClassifier` with the same script — pass `--lang isl`.
- Export to ONNX, build TensorRT engine, drop into `triton_repo/isl_classifier/1/model.plan`.
- Add `languages/isl/config.yaml` and reference images.
- Restart Triton (or rely on poll mode) to load the new model.
- ISL appears in the language dropdown automatically — no UI code change.

> **Exit criterion:** switching the language dropdown to ISL works; sign 'A' in ISL; system gives correct GREEN.

---

## 12. Adding BSL (two-handed extension)

BSL is the other moment of truth — does the two-hand path work?

- Capture two-handed BSL alphabet samples with both hands visible.
- Use `--hands 2` in `extract_landmarks.py` to produce 126-feature CSV.
- Train `TwoHandClassifier`.
- Export ONNX with input shape `(1, 126)`, build engine.
- Drop into `triton_repo/bsl_classifier/1/model.plan` with the BSL `config.pbtxt`.
- Add `languages/bsl/config.yaml` with `input_hands: 2`.
- Verify UI handles "one hand visible, waiting for second" state with a hint.

> **Exit criterion:** BSL selection works; system requires both hands visible; correct GREEN on a known sign.

---

## 13. Configuration Summary

```yaml
# configs/thresholds.yaml
amber_min_confidence: 0.50
green_min_confidence: 0.80
hold_seconds_for_complete: 1.0
smoothing_window_frames: 15

# Per-language overrides allowed:
overrides:
  bsl:
    green_min_confidence: 0.75   # slightly more forgiving while two-hand model matures
```

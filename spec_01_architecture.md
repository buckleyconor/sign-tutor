# Multi-Language Sign Language Tutor — Master Architecture

**Spec 1 of 3**
**NVIDIA GB10 Demo Lab — Dell Customer Solution Center**

| | |
|---|---|
| Author | Conor (Dell CSC) |
| Version | 0.1 (Draft) |
| Date | May 2026 |

---

## 1. Executive Summary

This document specifies the architecture for a multi-language sign language tutor lab running on the NVIDIA GB10 (DGX Spark / `promaxgb10`). The system uses computer vision and a hand-landmark classifier to recognise fingerspelled letters in real time, compare them against a reference, and provide visual feedback to a learner via a split-screen interface.

The platform supports two sign languages from day one: American Sign Language (ASL) and Irish Sign Language (ISL). Both use one-handed alphabets with identical input shapes (63 normalised landmarks → 26 classes). The architecture is deliberately language-agnostic — adding additional sign languages requires only a new dataset and a per-language model, not a code rewrite.

The full ML stack is NVIDIA-centric: training in PyTorch on the GB10's Blackwell GPU, optimisation via TensorRT, and serving via Triton Inference Server. This makes the lab a credible enterprise reference implementation as well as a learning project.

### 1.1 Key Decisions

| Decision | Rationale |
|---|---|
| Hand tracking via MediaPipe Hands | CPU-only, fast, well-documented, runs natively on aarch64. Outputs 21 landmarks per hand, suitable for downstream classification. |
| Per-language classifier model | ASL and ISL alphabets differ at the letter level. Per-language models keep each one small, fast, and accurate. Both share the same input shape (63 features) since they are both one-handed. |
| PyTorch → ONNX → TensorRT pipeline | Standard NVIDIA inference path. Demonstrates real enterprise tooling rather than ad-hoc scripts. |
| Triton Inference Server | Provides HTTP/gRPC endpoint, model versioning, dynamic batching, and clean separation between UI and ML. Identical interface scales from GB10 to H100. |
| Gradio for UI | Python-native, fast to iterate, supports webcam input out of the box, easy to demo. Streamlit is an alternative but Gradio's webcam component is more mature. |
| Local-only first release | No cloud dependencies. Runs entirely on the GB10. Simplifies demo logistics and keeps customer data on-prem. |

---

## 2. Problem Statement & Scope

### 2.1 Problem

Sign language is a primary mode of communication for the Deaf and Hard-of-Hearing community. Recognition tools — even basic fingerspelling tutors — make learning more accessible and provide a foundation for accessibility products in healthcare, education, customer service, and video conferencing. There is no shortage of academic work in this area, but production-grade reference implementations on modern NVIDIA hardware are far rarer.

### 2.2 In Scope (Module 1 + Module 2)

- Real-time hand tracking via webcam input.
- Recognition of static fingerspelled alphabet letters in ASL and ISL.
- Split-screen UI: live feed + reference image + traffic-light score.
- Lesson progression: A→Z guided practice with feedback.
- Per-language model training and TensorRT optimisation.
- Triton-based inference serving.
- Module 2: small set of static or near-static common words per language.

### 2.3 Out of Scope (this version)

- Continuous signing / sentence-level translation.
- Dynamic letters (J, Z in ASL/ISL) — handled as static snapshots only.
- Facial expression analysis (a real component of full sign languages).
- Speech-to-sign avatar generation.
- Multi-user / cloud deployment.
- Accessibility certification — this is a learning tool, not a production assistive device.

### 2.4 Success Criteria

- Per-letter top-1 accuracy ≥ 90% on held-out test set, per language.
- End-to-end latency (frame in → classification out) < 100 ms on GB10.
- Stable 25+ FPS in the UI with no dropped frames during a 5-minute session.
- Adding a new sign language requires zero code changes — dataset and config only.

---

## 3. Sign Language Domain Notes

Sign languages are not universal. ASL and ISL are mutually unintelligible languages with distinct alphabets, even though they share the same one-handed fingerspelling approach.

| Language | Hands | Notes | Implication for our system |
|---|---|---|---|
| **ASL** | One | American Sign Language. Best-documented, most datasets available. J and Z involve motion. | Single-hand model. 21 landmarks input. Well-trodden ground. |
| **ISL** | One | Irish Sign Language. Officially recognised in Ireland (ISL Act 2017). Closer to ASL than to other sign languages but with letter-level differences. Limited public datasets. | Same input shape as ASL classifier. Custom dataset capture likely required. Strong local relevance for Cork-based CSC. |

---

## 4. System Architecture

### 4.1 High-level component diagram

```
+----------------------------------------------------------+
|                  GB10 (promaxgb10)                       |
|                                                          |
|   +---------+    +-----------------+    +-----------+    |
|   | Webcam  |--->| MediaPipe Hands |--->| Feature   |    |
|   |         |    | (CPU)           |    | builder   |    |
|   +---------+    +-----------------+    +-----+-----+    |
|                                               |          |
|                                               v          |
|   +-----------------+    +-------------+    +-------+    |
|   | Gradio UI       |<---| Score /     |<---|Triton |    |
|   | (split screen)  |    | Smoothing   |    |+TRT   |    |
|   +-----------------+    +-------------+    +-------+    |
|                                                          |
+----------------------------------------------------------+
```

### 4.2 Layered view

The system is organised into five layers:

- **Capture layer** — webcam frames via OpenCV at 30 FPS.
- **Perception layer** — MediaPipe Hands extracts 21 landmarks per detected hand.
- **Feature layer** — landmarks normalised into model input tensor (63 floats for one-handed).
- **Inference layer** — Triton serves the per-language TensorRT engine; gRPC/HTTP call returns class probabilities.
- **Application layer** — Gradio UI, lesson controller, scoring, traffic-light feedback.

### 4.3 Data flow per frame

1. OpenCV captures frame from webcam (BGR, 640×480 default).
2. Frame is converted to RGB and passed to MediaPipe Hands.
3. MediaPipe returns 0, 1 or 2 hand landmark sets.
4. Feature builder normalises landmarks (translation/scale invariant) and assembles the language-appropriate input vector.
5. If a complete input is available, it is sent to Triton with the model name matching the active language (e.g. `asl_classifier`).
6. Triton returns 26 class logits.
7. Score smoother applies a rolling average over the last N frames to suppress jitter.
8. Lesson controller compares smoothed prediction to target letter and emits a traffic-light score.
9. Gradio renders annotated webcam frame + reference image + score + feedback text.

### 4.4 Multi-language design

Languages are first-class configuration, not hard-coded. The system loads a registry of supported languages at startup, each with its own model, label set, reference imagery, and required input shape.

```
languages/
  asl/
    config.yaml          # input_hands: 1, classes: [A..Z]
    model.onnx
    model.engine         # TensorRT engine
    references/          # 26 reference images (PNG)
      A.png
      B.png ...
  isl/
    config.yaml          # input_hands: 1, classes: [A..Z]
    model.onnx
    model.engine
    references/
```

Adding a new sign language (e.g. Auslan, French Sign Language) requires only: a labelled dataset, a `config.yaml`, a trained model, and a reference image set. No pipeline or UI code changes.

### 4.5 ASL vs ISL

ASL and ISL share the input shape (one hand, 63 features). Their classifiers are structurally identical — only weights and label sets differ. This means the feature pipeline, normalisation, and model architecture are the same for both languages. The only per-language customisation is the model weights and label set.

The shared input shape is a key design advantage: the UI, lesson controller, and scoring code work identically for both languages with zero conditional logic.

---

## 5. NVIDIA Stack & GB10 Compatibility

### 5.1 Verified components

| Component | Status | Notes |
|---|---|---|
| MediaPipe Hands | ✅ OK | CPU-based, runs natively on aarch64. Pip install on Ubuntu 24.04 works. |
| PyTorch (NVIDIA container) | ✅ OK | Use `nvcr.io/nvidia/pytorch:25.xx-py3` — Blackwell-optimised, includes Triton 3.5 and FlashAttention 2. Do not use generic pip wheels. |
| TensorRT | ✅ OK | Native ARM64 support on DGX Spark. `trtexec` available. CUDA 13 / sm_121 target architecture. **Requires `tritonserver:26.04-py3` or later** — TRT 10.9 (25.03) has no kernel implementations for sm_121 and fails at engine build time. |
| Triton Inference Server | ✅ OK | ARM64 builds available. Use `26.04-py3` (TRT 10.16.01) — confirmed working on GB10. |
| ONNX Runtime GPU | ⚠️ Workaround | PyPI does not host aarch64 wheels. Use Ultralytics-hosted wheel: `onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl`. Only needed if doing ONNX-based inference outside Triton. |
| TAO Toolkit | Optional | Supports ARM64 since v6.0.0. Useful for image-based Module 2 experiments. For Module 1 (landmarks-only), direct PyTorch is simpler. |
| OpenCV / Gradio | ✅ OK | Pure Python, no aarch64 issues. |

### 5.2 Container strategy

The full system runs in two Docker containers on the GB10:

- **triton-server**: `nvcr.io/nvidia/tritonserver:26.04-py3` — hosts the per-language TensorRT engines on port 8000 (HTTP) and 8001 (gRPC). Must be 26.04+ for GB10 engine compilation (TRT 10.16).
- **tutor-app**: built from `nvcr.io/nvidia/pytorch:25.xx-py3` — runs MediaPipe, the Gradio UI, and the lesson controller; communicates with Triton over the local Docker network.

Training runs ad-hoc inside the pytorch container. Trained engines are copied into the Triton model repository and the server is signalled to reload.

---

## 6. UI / UX Specification

### 6.1 Layout

The UI is a single-page split-screen layout. Left half: live annotated webcam feed. Right half: reference imagery and feedback panel. Top bar: language and lesson selector.

```
+--------------------------------------------------------------+
| [Language: ASL ▾]  [Lesson: Alphabet ▾]   [Letter: D]  ●●●○○ |
+----------------------------+---------------------------------+
|                            |                                 |
|                            |   Sign the letter:              |
|     LIVE WEBCAM            |                                 |
|     (with landmarks        |          [reference image of D] |
|      overlaid)             |                                 |
|                            |   Quality:  [GREEN]             |
|                            |   Confidence: 96%               |
|                            |                                 |
|                            |   [ Skip ]   [ Next letter ]    |
+----------------------------+---------------------------------+
```

### 6.2 Traffic-light scoring

| Light | Trigger | UX behaviour |
|---|---|---|
| 🔴 **RED** | `conf < 0.50`, OR `predicted != target` | Show "try again" hint. Highlight key landmarks that differ from reference (stretch goal). |
| 🟠 **AMBER** | `0.50 ≤ conf < 0.80` AND `predicted == target` | Show "close" hint. Encourage user to hold the sign more clearly. |
| 🟢 **GREEN** | `conf ≥ 0.80` AND `predicted == target` sustained for ≥ 1.0 s | Mark letter as completed; auto-advance after a brief celebration animation. |

Thresholds are configurable via a single YAML file so demos can be tuned to room lighting / camera quality without code changes.

### 6.3 Smoothing

Raw frame-by-frame predictions jitter heavily. The UI uses a 15-frame (~0.5 s) rolling window: the displayed prediction is the modal class across the window, and the displayed confidence is the average for that class. This eliminates flicker without adding noticeable lag.

### 6.4 Lesson flow

1. User selects a language (ASL / ISL).
2. User selects a lesson (Module 1: Alphabet; Module 2: Words).
3. System presents target sign with reference image.
4. User signs; traffic-light updates in real time.
5. On sustained GREEN, system advances to next target.
6. Session ends after the full lesson; summary screen shows per-letter best score.

---

## 7. Risks & Mitigation

| Risk | Severity | Mitigation |
|---|---|---|
| Limited public ISL alphabet datasets | High | Self-capture with webcam, augmented with rotations/lighting; document data provenance carefully. Engage with Irish Deaf Society for guidance and review. |

| MediaPipe missed-detections under poor lighting | Medium | Document recommended demo lighting; UI to display "no hand detected" clearly; consider a fallback CV-CUDA pose model later. |
| aarch64 wheel availability for niche dependencies | Low | Stay inside the NVIDIA PyTorch container; avoid exotic dependencies. Already-known workaround for `onnxruntime-gpu`. |
| Cultural sensitivity / misrepresenting Deaf community | Medium | Frame the lab clearly as a learning aid, not an assistive product. Use reference imagery from authoritative sources. Acknowledge this is fingerspelling, not full sign language. |
| Demo audience confuses fingerspelling with full signing | Low | Add explicit on-screen note: "This lab teaches fingerspelling — one component of sign language." |

---

## 8. Deliverables & Timeline

*Indicative effort, single developer, part-time:*

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Environment & dataset | Containers running on GB10; ASL Kaggle dataset processed to landmark CSV. |
| 2 | ASL classifier | Trained PyTorch ASL model, ONNX export, TensorRT engine, Triton serving, smoke test. |
| 3 | UI + scoring | Gradio split-screen, traffic-light scoring, smoothing, ASL alphabet lesson playable end-to-end. |
| 4 | ISL | ISL dataset captured, classifier trained, plugged in via language registry. Adding a second language proves the framework. |
| 4 | ISL (one-handed) | ISL dataset captured, classifier trained, plugged in via language registry. Adding a second language proves the framework. |
| 6 | Module 2 + polish | Static-word lesson set per language. Demo-ready polish, latency tuning, documentation. |

---

## 9. Companion Specifications

- **Spec 2 — Module 1 Detailed Build Spec**: end-to-end implementation steps for the ASL alphabet classifier, ISL extension, and integration with Triton + Gradio.
- **Spec 3 — Test & Validation Plan**: unit, integration, and acceptance tests; data quality checks; performance benchmarks.

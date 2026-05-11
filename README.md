# Sign Language Tutor

Multi-language sign language fingerspelling tutor for the NVIDIA GB10 (DGX Spark).

## Languages

- **ASL** (American Sign Language) — one-handed alphabet
- **ISL** (Irish Sign Language) — one-handed alphabet  
- **BSL** (British Sign Language) — two-handed alphabet

## Architecture

- Hand tracking: MediaPipe Hands (CPU)
- Training: PyTorch → ONNX → TensorRT
- Inference: Triton Inference Server
- UI: Gradio (split-screen webcam + feedback)

## Quick Start

```bash
# 1. Start Triton + app
docker-compose up

# 2. Open http://localhost:7860
# 3. Select language, sign letters, get feedback
```

## Project Structure

See [Spec 1 — Architecture](spec_01_architecture.md) and [Spec 2 — Module 1 Build](spec_02_module1_build.md) for full details.

## Specs

| Spec | Description |
|------|-------------|
| [Spec 1](spec_01_architecture.md) | Master architecture |
| [Spec 2](spec_02_module1_build.md) | Module 1 detailed build |
| [Spec 3](spec_03_test_plan.md) | Test & validation plan |
| [Spec 4](spec_04_data_acquisition.md) | Data acquisition & preparation |

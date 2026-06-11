# Sign Tutor — Container Software Inventory

Generated: 2026-06-11
Host: promaxgb10 (aarch64)

Two containers comprise the running application:

| Service | Container | Image |
|---|---|---|
| Inference UI / app | `sign-tutor-tutor-app-1` | `sign-tutor:latest` (built from local Dockerfile) |
| Model server | `sign-tutor-triton-1` | `nvcr.io/nvidia/tritonserver:26.04-py3` |

---

## Container 1: tutor-app (`sign-tutor:latest`)

### Core stack
| Component | Version |
|---|---|
| OS | Ubuntu 24.04.1 LTS |
| Python | 3.12.3 |
| CUDA toolkit (nvcc) | 12.8.93 |
| PyTorch | 2.7.0a0+7c8ec84dab.nv25.3 |
| torchvision | 0.22.0a0 |
| Torch-TensorRT | 2.7.0a0 |
| TensorRT | 10.9.0.34 |
| ONNX | (built from pytorch third_party) |
| Gradio | 6.15.2 |
| MediaPipe | 0.10.18 |
| OpenCV (contrib / headless) | 4.11.0.86 |
| Triton client (tritonclient) | 2.51.0 |
| NumPy | 1.26.4 |
| scikit-learn | 1.6.1 |
| FastAPI / Uvicorn | 0.136.3 / 0.48.0 |

### Full pip package list
See `pip_freeze_tutor_app.txt` (generated alongside this report) — ~260 packages including the RAPIDS 25.2.0 suite (cudf, cuml, cugraph), JAX 0.7.1, spaCy 3.7.5, and the NVIDIA NGC PyTorch container stack.

---

## Container 2: triton (`tritonserver:26.04-py3`)

### Core stack
| Component | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Python | 3.12.3 |
| CUDA toolkit (nvcc) | 13.2.78 |
| Triton Server | 2.68.0 (tritonfrontend wheel) |
| CuPy | 13.6.0 (cuda13x) |
| NVIDIA DALI | 2.0.0 (cuda130) |
| NumPy | 1.26.4 |
| SciPy | 1.16.3 |
| FastAPI / Starlette | 0.121.2 / 0.49.3 |
| openai (client) | 1.107.3 |

### Full pip package list
```
absl-py==2.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
astunparse==1.6.3
attrs==26.1.0
certifi==2026.4.22
cupy-cuda13x==13.6.0
distlib==0.4.0
distro==1.9.0
dm-tree==0.1.9
fastapi==0.121.2
fastrlock==0.8.3
filelock==3.29.0
gast==0.7.0
h11==0.16.0
httpcore==1.0.9
httpx==0.27.2
idna==3.13
iniconfig==2.3.0
jiter==0.14.0
makefun==1.16.0
numpy==1.26.4
nvidia-dali-cuda130==2.0.0
nvidia-libnvcomp-cu13==5.1.0.21
nvidia-nvimgcodec-cu13==0.7.0.49
nvidia-nvjpeg==13.1.0.48
nvidia-nvjpeg2k-cu13==0.10.0.49
nvidia-nvtiff-cu13==0.7.0.79
nvtx==0.2.15
openai==1.107.3
packaging==25.0
partial-json-parser==0.2.1.1.post7
platformdirs==4.9.6
pluggy==1.6.0
pydantic==2.10.6
pydantic_core==2.27.2
Pygments==2.20.0
pytest==9.0.3
python-discovery==1.2.2
scipy==1.16.3
setuptools==68.1.2
six==1.17.0
sniffio==1.3.1
starlette==0.49.3
tqdm==4.67.3
tritonfrontend==2.68.0
tritonserver==0.0.0 (wheel)
typing_extensions==4.15.0
virtualenv==21.2.4
wheel==0.47.0
wrapt==2.1.2
```

---

---

## OS-level (apt/dpkg) packages

Full lists: `dpkg_tutor_app.txt` (410 packages) and `dpkg_triton.txt` (387 packages).
Notable system/runtime libraries below.

### Shared base (both containers)
| Component | tutor-app | triton |
|---|---|---|
| python3.12 | 3.12.3-1ubuntu0.5 | 3.12.3-1ubuntu0.13 |
| gcc / g++ | 13.2.0 (13.3.0 toolchain) | 13.2.0 (13.3.0 toolchain) |
| git | 2.43.0 | 2.43.0 |
| openssl / libssl3 | 3.0.13-0ubuntu3.5 | 3.0.13-0ubuntu3.7 |
| libglib2.0-0 | 2.80.0-6ubuntu3.2 | 2.80.0-6ubuntu3.8 |
| libgomp1 | 14.2.0 | 14.2.0 |

### CUDA / GPU stack (these DIFFER — see notes)
| Component | tutor-app | triton |
|---|---|---|
| CUDA toolkit | 12.8 (cuda-nvcc 12.8.93) | 13.2 (cuda-nvcc 13.2.78) |
| cuda-cudart | 12.8.90 | 13.2.75 |
| libcublas | 12.8.4.1 | 13.4.0.1 |
| libcudnn9 | 9.8.0.87 (cuda-12) | 9.21.0.82 (cuda-13) |
| libnccl2 | 2.25.1+cuda12.8 | 2.29.7+cuda13.2 |
| **libnvinfer (TensorRT)** | **10.9.0.34+cuda12.8** | **10.16.1.11+cuda13.2** |
| cuda-compat | 570.124.06 | 595.58.03 (+ orin compat) |

### App-specific media/imaging libs (tutor-app)
- libjpeg8 / libjpeg-turbo8 2.1.5, libpng16 1.6.43, libprotobuf 3.21.12 — backing OpenCV/MediaPipe image handling.

---

## Notes
- The two containers run **different CUDA toolkit versions** (12.8 in tutor-app, 13.2 in triton) and different CuPy/DALI CUDA lines (cu12 vs cu13). This is expected since they come from different base images; they communicate over the network (gRPC/HTTP), not by sharing CUDA libs.
- ⚠️ **TensorRT version skew between containers:** tutor-app ships TensorRT **10.9.0.34** (Python + libnvinfer) while triton ships **10.16.1.11**. A TensorRT engine (`.plan`) is **not portable across TRT minor versions** — it must be built with the same TensorRT that Triton loads it with. If engines are built inside tutor-app and served by triton, they will likely fail to load or silently fall back. Build the TRT engine in the triton container (or a matching 10.16.x environment). This compounds the existing model.onnx→engine drift risk already noted in project memory.
- The tutor-app is the heavy image — it carries the full NGC PyTorch + RAPIDS training/inference stack, while triton carries only the serving runtime.

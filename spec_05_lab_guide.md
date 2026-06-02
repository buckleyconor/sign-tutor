# Lab Guide — Add a New Sign Language

**Sign Language Tutor — Hands-on NVIDIA Stack Lab**
**Spec 5 of 5**
**Target duration: 3–4 hours**

---

## 0. Before you start

### 0.1 What this lab is

You will take a working sign-language recognition application that supports American Sign Language (ASL) and extend it to also support Irish Sign Language (ISL). Along the way you will touch every layer of NVIDIA's vision-AI stack: GPU-accelerated training in **PyTorch** on a **Blackwell** GPU, model optimisation with **TensorRT**, and production serving via **Triton Inference Server**.

The end deliverable is your trained ISL model running live in the same application. When you sign an ISL letter at the webcam, your model — running on the Triton server you just deployed to — recognises it and the traffic-light score turns green.

### 0.2 What this lab is *not*

This is not an app-development exercise. The application is already built; you are extending it. The technical learning happens in the NVIDIA tooling between data preparation and live inference. If you find yourself editing UI code, you have gone off-route — come back to the lab guide.

### 0.3 What you will learn

By the end you will have first-hand experience of:

- Extracting training features from raw images using MediaPipe on GPU.
- Understanding how landmark-space augmentation improves model robustness.
- Training a small classifier in the NVIDIA PyTorch container on Blackwell hardware.
- Exporting to ONNX and building a TensorRT engine for the GB10's `sm_121` architecture.
- Deploying a new model to a running Triton server without restarting it.
- Understanding how the same pipeline scales from a GB10 to a data-centre H100.

### 0.4 Prerequisites

You should be comfortable with:

- A Linux command line (`cd`, `cat`, `ls`, editing a YAML file).
- Running `docker` and `docker compose` commands.
- Reading Python (you will not need to write much).

You do **not** need prior experience with sign language, computer vision, or model training.

### 0.5 What is already running before you sit down

The facilitator has prepared:

- A GB10 (`promaxgb10`) running Docker, with two containers up:
  - `sign-tutor-triton-1`: hosts the ASL TensorRT model on port 8010 (HTTP) and 8011 (gRPC).
  - `sign-tutor-tutor-app-1`: the Gradio UI, reachable at `http://promaxgb10.local:7860`.
- A working ASL classifier already serving from Triton.
- A pre-collected dataset of ISL hand images at `datasets/isl_frames/` — 26 letter subdirectories, roughly 35 frames each (~910 images total).
- A clone of the `sign-tutor` repository at `/home/democenter/projects/sign-tutor`. The ISL language configuration and reference images are already in the repo; your job is to produce and deploy the trained model.

You will work entirely inside the project directory for this session:

```bash
cd /home/democenter/projects/sign-tutor
```

---

## 1. Orientation (15 minutes)

### 1.1 Verify the baseline works

Open a browser on the lab workstation and navigate to:

```
http://promaxgb10.local:7860
```

You should see the sign-language tutor UI with ASL selected. Sign a letter (the reference image shows you how). The quality bar should turn green when you match.

> ✅ **Checkpoint:** ASL recognition works.

### 1.2 Inspect what is running

In a terminal on the workstation, run:

```bash
cd /home/democenter/projects/sign-tutor
docker compose ps
```

You should see two services running. Now check what Triton is serving:

```bash
curl -s http://localhost:8010/v2/models/asl_classifier | python3 -m json.tool
```

You should see the ASL classifier with platform `tensorrt_plan`. There is no ISL classifier yet — you are about to create one.

```bash
# Confirm ISL is not yet serving (expect an error response)
curl -s http://localhost:8010/v2/models/isl_classifier
```

### 1.3 Understand the extension point

Open the languages directory:

```bash
ls languages/
```

You will see two folders: `asl/` and `isl/`. The ISL folder already contains the language configuration and reference images for the UI — these were prepared in advance. What is missing is a trained model. Look at what is there:

```bash
ls languages/isl/
cat languages/isl/config.yaml
ls languages/isl/references/
```

The `config.yaml` declares the language name, letter classes, and — critically — the `triton_model_name: "isl_classifier"`. The application will look for a Triton model with that name. Your job is to train that model and deploy it.

> ✅ **Checkpoint:** You understand that the application already knows about ISL. Your job is to produce the trained model artefact and deploy it to Triton.

---

## 2. Extract the training data (20 minutes)

### 2.1 Understand the dataset

The raw ISL data is a collection of hand images organised by letter:

```bash
ls datasets/isl_frames/        # 26 letter subdirectories (A–Z)
ls datasets/isl_frames/A/      # ~35 JPEG frames per letter
```

Each image shows a hand forming a letter of the ISL fingerspelling alphabet. Rather than training directly on images (which would require a larger model and much more data), this pipeline extracts 21 hand-landmark coordinates from each image using MediaPipe. The resulting 63-dimensional feature vector (21 landmarks × 3 coordinates, normalised relative to the wrist) is compact, pose-invariant, and fast to train on.

### 2.2 Extract landmarks

Run the landmark extraction from inside the tutor-app container (which has MediaPipe installed):

```bash
docker exec sign-tutor-tutor-app-1 python training/extract_landmarks.py \
    --src datasets/isl_frames \
    --dst datasets/isl_landmarks.csv \
    --source-tag isl_frames \
    --hands 1
```

This will take 1–2 minutes. MediaPipe processes each image and writes one CSV row per successfully detected hand. Frames where no hand is detected are silently skipped — this is normal; some frames are cropped or low-quality.

Check the output:

```bash
wc -l datasets/isl_landmarks.csv
```

Expect 600–850 rows (not all ~910 images will yield a detection).

### 2.3 Verify data quality

```bash
docker exec sign-tutor-tutor-app-1 \
    python training/check_quality.py datasets/isl_landmarks.csv 63
```

This checks that the feature dimension is correct (63) and that the class distribution is not severely imbalanced. You may see a warning about having only one source — this is expected for a single pre-collected dataset and is acceptable here.

> ✅ **Checkpoint:** You have a `datasets/isl_landmarks.csv` file with ISL landmark data.

---

## 3. Augmentation (15 minutes)

### 3.1 Why augment

Your dataset, though real, is limited in diversity: one signer, controlled lighting, consistent angle. A model trained on it alone will work well in lab conditions but may struggle with different hand sizes, wrist tilts, or lighting variations. Augmentation generates additional training variants without requiring more real data.

### 3.2 How augmentation works in this pipeline

Open the augmentation module:

```bash
cat training/augment.py
```

Unlike image-space augmentation (which would use a library like NVIDIA DALI to apply transforms to pixels), this pipeline augments in *landmark space* — directly perturbing the 63-dimensional feature vectors at training time. Three operations are applied to each training sample:

| Operation | Effect | Rationale |
|---|---|---|
| Gaussian noise | `± ~1%` jitter on each coordinate | MediaPipe itself is noisy; teaching the model to tolerate noise |
| In-plane rotation `± 10°` | Rotates landmarks around the wrist | Handles wrist tilt variation |
| Mirror flip (50% chance) | Flips the x-axis | Handles left/right hand variation |

### 3.3 Augmentation is automatic

You do not need to run a separate augmentation step. The training script applies these transforms to every training batch automatically — each epoch the model sees a slightly different version of every sample. This is why training for 50 epochs on ~700 samples produces a model that generalises beyond those 700 examples.

> ✅ **Checkpoint:** You understand what augmentation does and why the training script applies it automatically.

---

## 4. Train the classifier (30 minutes)

### 4.1 What you are training

A small Multi-Layer Perceptron (MLP) with three dense layers. Input: 63 floats. Output: 26 logits (one per letter A–Z). About 12,000 trainable parameters in total. On the GB10's Blackwell GPU, training takes 2–3 minutes.

This is deliberately tiny. For a well-bounded classification task with good features (which landmarks are), small models with clean data outperform large models with noisy data every time.

### 4.2 Run training

```bash
docker exec sign-tutor-tutor-app-1 python training/train_classifier.py \
    --dataset datasets/isl_landmarks.csv \
    --epochs 50 \
    --checkpoint-dir checkpoints/isl \
    --csv-file checkpoints/isl/train_log.csv
```

You will see per-epoch progress: training loss, validation loss, validation accuracy. The validation accuracy should rise quickly and plateau in the 80–95% range depending on dataset quality.

While it trains, open a second terminal and watch GPU activity:

```bash
nvidia-smi --loop-ms=500
```

You should see the GB10's GPU utilisation spike to 80–100% during the training loop. This is the Blackwell tensor cores running your matrix multiplications.

### 4.3 Inspect the result

```bash
cat checkpoints/isl/train_log.csv | column -ts,
```

Look at the final few epochs. You are looking for:

- **Validation accuracy ≥ 85%.** Below this, your model may struggle in the live demo.
- **Validation loss still decreasing (not diverging).** If it starts climbing while training loss falls, the model is overfitting — reduce `--epochs` or the dataset is too small.

If you fall short of 85%, see Section 9 (troubleshooting).

> ✅ **Checkpoint:** `checkpoints/isl/best.pt` exists and the training log shows validation accuracy ≥ 85%.

---

## 5. Export and optimise (20 minutes)

### 5.1 ONNX export

PyTorch is great for training, but for production inference NVIDIA has a faster path. The first step is exporting to ONNX, an open intermediate format that separates the model architecture from the training framework.

```bash
docker exec sign-tutor-tutor-app-1 python training/export_onnx.py \
    --checkpoint checkpoints/isl/best.pt \
    --output languages/isl/model.onnx
```

Verify the file was produced:

```bash
ls -lh languages/isl/model.onnx
```

### 5.2 Build the TensorRT engine

This is the NVIDIA-specific optimisation step. TensorRT compiles the ONNX graph into a binary engine tailored to the target GPU's architecture. On the GB10, that target is `sm_121` (Blackwell, low-power variant).

TensorRT is available inside the Triton container. Copy the ONNX in and build:

```bash
# Copy the ONNX into the Triton container
docker cp languages/isl/model.onnx sign-tutor-triton-1:/tmp/isl_model.onnx

# Create the Triton model version directory
mkdir -p triton_repo/isl_classifier/1

# Build the TensorRT engine (runs inside the container where trtexec lives)
docker exec sign-tutor-triton-1 \
    /usr/src/tensorrt/bin/trtexec \
        --onnx=/tmp/isl_model.onnx \
        --saveEngine=/models/isl_classifier/1/model.plan \
        --minShapes=input:1x63 \
        --optShapes=input:32x63 \
        --maxShapes=input:64x63 \
        --useCudaGraph
```

A few flags worth understanding:

- `--minShapes / --optShapes / --maxShapes` — tell TensorRT the range of batch sizes to expect. The engine is optimised for the `optShapes` size (32 samples) and handles the full 1–64 range. For the streaming UI, single-sample (batch=1) inference is the hot path.
- `--useCudaGraph` — captures kernel launch graphs for lower latency on repeated calls.

> **Why no `--fp16`?** FP16 (half-precision) is beneficial for large models where memory bandwidth is the bottleneck. For a 12,000-parameter MLP like this one, FP16 quantisation shifts logits enough to change the top-1 predicted class on a large fraction of inputs — effectively breaking the classifier. Use FP32 here; the latency difference at this model size is negligible (both are sub-millisecond).

> **A note worth remembering:** TensorRT engines are tied to GPU architecture. The `model.plan` you just built will run on the GB10 (sm_121). It will **not** run on an H100 (sm_90) or a 4090 (sm_89). To deploy to different hardware, you rebuild from the same ONNX on that hardware. The ONNX is the portable artefact; the engine is hardware-specific.

Check the engine was produced:

```bash
ls -lh triton_repo/isl_classifier/1/model.plan
```

> ✅ **Checkpoint:** `triton_repo/isl_classifier/1/model.plan` exists.

---

## 6. Deploy to Triton (15 minutes)

### 6.1 Triton's model repository

Triton serves models from a folder structure on disk. New models are deployed by adding folders; updates happen by adding numbered version subdirectories. There is no separate "deploy step" — the directory layout *is* the deployment.

Look at the current state:

```bash
ls -la triton_repo/
```

You will see `asl_classifier/` (the pre-existing model) and your new `isl_classifier/1/model.plan`. Now add the configuration file.

### 6.2 Write the config

Create `triton_repo/isl_classifier/config.pbtxt`:

```
name: "isl_classifier"
platform: "tensorrt_plan"
max_batch_size: 64
input  [ { name: "input",  data_type: TYPE_FP32, dims: [63] } ]
output [ { name: "output", data_type: TYPE_FP32, dims: [26] } ]
instance_group [ { count: 1, kind: KIND_GPU } ]
```

This declares: a TensorRT model called `isl_classifier`, taking 63-float inputs (one hand's landmarks) and producing 26-float outputs (one logit per letter), served on the GPU.

> **Note on `max_batch_size`:** Setting this to 64 lets Triton manage batching — it adds the batch dimension automatically and will queue up to 64 concurrent requests. For the streaming UI (which sends one frame at a time), Triton dispatches each request immediately; the batch headroom is there for when you scale to concurrent users.

### 6.3 Triton picks it up automatically

The Triton container is running with `--model-control-mode=poll`, which means it watches the repository for changes and loads new models automatically (poll interval: 5 seconds).

Wait a few seconds, then confirm:

```bash
curl -s http://localhost:8010/v2/models/isl_classifier | python3 -m json.tool
```

You should see `isl_classifier` with platform `tensorrt_plan`. The ISL model is now live.

> ✅ **Checkpoint:** Triton is serving your ISL model.

---

## 7. Activate in the application (10 minutes)

### 7.1 The language is already registered

The application reads `languages/*/config.yaml` at startup. The ISL config was pre-prepared — you can verify it:

```bash
cat languages/isl/config.yaml
```

Check that `triton_model_name: "isl_classifier"` matches the Triton model name you just deployed. The reference images for the UI are also already in place:

```bash
ls languages/isl/references/
```

### 7.2 Restart the application

```bash
docker compose restart tutor-app
```

The application reloads its language registry on startup. It takes about 10 seconds.

> ✅ **Checkpoint:** ISL is registered and the application has reloaded.

---

## 8. Test live (15 minutes)

### 8.1 Open the UI

Refresh `http://promaxgb10.local:7860` in your browser.

The language dropdown should now show two options: American Sign Language and Irish Sign Language. Select ISL.

### 8.2 Sign a few letters

Use the reference image as guidance. Hold each sign clearly and steadily for 1–2 seconds. The quality bar should rise as the model's confidence grows — the bar only turns green when the model consistently predicts the target letter above the 90% confidence threshold.

> 💡 **Tip:** The smoother accumulates predictions over 15 frames (about 3 seconds at the streaming rate). Hold the sign steady rather than moving — the system is looking for consistency, not speed.

> ✅ **Checkpoint:** Your trained ISL model is recognising your live signing through the Triton-hosted TensorRT engine.

That sentence is worth re-reading. *Your* model, *your* engine, on the same infrastructure that could scale up to a data centre, recognising *your* hand in real time. You have just deployed a working production inference pipeline on Blackwell.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `extract_landmarks.py` produces very few rows | Many images lack a detectable hand | Normal for some frames — proceed if you have ≥ 600 rows |
| Quality check fails on imbalance | One letter class has very few detections | Expected with ~35 frames/letter; proceed to training anyway |
| Training accuracy plateaus below 80% | Insufficient data diversity | Reduce `--val-split` to 0.1; or add more data via capture (see Appendix C) |
| `export_onnx.py` fails | Checkpoint path wrong | Verify `checkpoints/isl/best.pt` exists first |
| `trtexec` fails with "no implementation" | Wrong container or TRT version | Must run inside `sign-tutor-triton-1`; the host does not have trtexec |
| Triton reports model as `UNAVAILABLE` | `config.pbtxt` missing or typo | Check `docker logs sign-tutor-triton-1` for the actual error |
| UI doesn't show ISL in dropdown | Application has not reloaded | `docker compose restart tutor-app` |
| UI shows ISL but always "no hand detected" | Webcam permission denied | Check browser permissions; allow camera for `promaxgb10.local` |
| Quality bar stays red | Model accuracy too low or sign not held long enough | Hold sign steady for 2+ seconds; check training val accuracy |
| Predictions look random for every sign | Wrong model deployed | Verify `triton_model_name` in `config.yaml` matches the folder name in `triton_repo/` |

---

## 10. Going further (optional, 30+ minutes)

If you finish early, try one or more of:

### 10.1 Profile end-to-end latency

Instrument the lesson controller and measure: frame capture → MediaPipe → normalise → Triton → score. Identify the slowest stage. Hint: MediaPipe (CPU) is almost always the bottleneck — your Triton call is sub-millisecond.

### 10.2 Dynamic batching for concurrent users

The current config serves requests one at a time. Add dynamic batching to Triton for high-concurrency scenarios:

```
dynamic_batching {
  preferred_batch_size: [ 8, 16 ]
  max_queue_delay_microseconds: 5000
}
```

The `max_queue_delay_microseconds: 5000` (5 ms) cap means Triton won't hold a request longer than 5 ms waiting for a batch to fill — so single-user streaming latency is unaffected, while concurrent users get batched together automatically. Run the smoke test from multiple terminals simultaneously and watch Triton's throughput metrics. This is the capability that makes Triton relevant at data-centre scale.

### 10.3 Compare INT8 to FP32

Rebuild the engine with `--int8` and a calibration dataset. Measure the accuracy drop versus the latency gain. This is the kind of calibration enterprise teams do routinely for production inference.

### 10.4 Per-class confusion analysis

Look at the training log's per-epoch validation accuracy. Which letters does your model confuse most often? In ISL fingerspelling, certain pairs (M/N, R/U) share similar static shapes. Now you have hard evidence of where to focus your next data capture session.

---

## 11. Resetting for the next participant

Hand-off matters. The lab is shared infrastructure — leave it the way you found it.

There is a reset script that returns everything to a known-good state in under 30 seconds:

```bash
bash scripts/lab_reset.sh
```

This script:

1. Stops the `tutor-app` container.
2. Removes the `isl_classifier` model folder from `triton_repo/` (Triton drops it within the next poll cycle).
3. Removes participant-trained checkpoints and extracted datasets.
4. Restarts `tutor-app`.
5. Verifies the ASL baseline still works.

The script is idempotent — running it twice is safe. It leaves the Docker images, the pre-collected `datasets/isl_frames/` dataset, and the `languages/isl/` configuration intact.

If you want to keep your trained model, copy it before resetting:

```bash
cp checkpoints/isl/best.pt ~/my_isl_model.pt
```

---

## 12. What you take away

You have completed an end-to-end NVIDIA inference pipeline:

| Stage | Tool |
|---|---|
| Feature extraction | MediaPipe (CPU, in PyTorch container) |
| Data augmentation | Landmark-space augmentation (training loop) |
| Training | PyTorch in `nvcr.io/nvidia/pytorch` container on Blackwell |
| Format conversion | ONNX (`torch.onnx.export`) |
| Optimisation | TensorRT `trtexec` (sm_121, FP32) |
| Serving | Triton Inference Server (`tensorrt_plan`, poll-mode hot-reload) |
| Validation | Accuracy gate + live webcam test |

The application happens to be a sign-language tutor. The pipeline you ran is identical to the one a real enterprise team would use for production defect detection, medical imaging triage, retail shelf analytics, or any other vision-classification problem. Only the dataset and the label set change.

If you would like to talk about applying this to your own use case, the facilitator is happy to dig in.

---

## Appendix A — The reset script

```bash
#!/usr/bin/env bash
# scripts/lab_reset.sh — restore lab to baseline (ASL only) state.
# Idempotent. Safe to run multiple times.

set -euo pipefail

REPO=/home/democenter/projects/sign-tutor

log() { printf '[reset] %s\n' "$*"; }

# 1. Stop the tutor-app so it doesn't reload a half-reset state.
log "Stopping tutor-app..."
cd "$REPO"
docker compose stop tutor-app >/dev/null 2>&1 || true

# 2. Drop any non-ASL Triton models.
log "Removing non-baseline Triton models..."
for model_dir in "$REPO/triton_repo"/*/; do
    name=$(basename "$model_dir")
    if [[ "$name" != "asl_classifier" ]]; then
        log "  removing $name"
        rm -rf "$model_dir"
    fi
done

# 3. Remove participant-generated artefacts.
log "Clearing participant artefacts..."
rm -rf "$REPO/checkpoints/isl"
rm -f  "$REPO/datasets/isl_landmarks.csv"
rm -f  "$REPO/languages/isl/model.onnx"

# 4. Restart tutor-app.
log "Restarting tutor-app..."
docker compose up -d tutor-app >/dev/null

# 5. Verify the baseline is healthy.
log "Verifying baseline..."
sleep 8
for i in {1..10}; do
    if curl -sf http://localhost:8010/v2/models/asl_classifier/ready \
            >/dev/null 2>&1; then
        log "Triton: asl_classifier READY"
        break
    fi
    sleep 2
done

if ! curl -sf http://localhost:7860/ >/dev/null 2>&1; then
    log "WARNING: tutor-app not responding on :7860"
    exit 1
fi

log "Reset complete. Lab is at ASL-only baseline."
```

### A.1 What is *not* reset (and why)

- **Docker images.** Re-pulling wastes minutes. Images change rarely.
- **`datasets/isl_frames/`** — the pre-collected image dataset is a shared lab asset, not a participant artefact.
- **`languages/isl/`** (config.yaml, references/) — pre-prepared, not participant-created.
- **`triton_repo/asl_classifier/`** — the baseline ASL engine; never touched by participants.

---

## Appendix B — Facilitator notes

### B.1 Timing

| Section | Target | Common overrun |
|---|---|---|
| 0–1. Setup & orientation | 15 min | Workstation issues, browser permissions |
| 2. Data extraction | 20 min | Usually smooth; extraction takes ~2 min |
| 3. Augmentation | 15 min | Conceptual — no command to run, just reading and understanding |
| 4. Training | 30 min | Usually finishes early; encourage participants to watch `nvidia-smi` |
| 5. Export & TensorRT | 20 min | First-timers slow on the `trtexec` flags; have the answer ready |
| 6. Triton deploy | 15 min | Triton poll-mode confusion ("did it work?") — show them the `curl` |
| 7. Register | 10 min | Smooth — config already exists |
| 8. Test live | 15 min | Webcam lighting issues; bring a portable lamp |
| **Total** | **~140 min** | Plus 30 min buffer = ~2.8 hours |

### B.2 Common stuck points

- **Step 2 produces < 400 rows.** Some images in `isl_frames` are low quality or cropped. If fewer than ~600 rows are extracted, training accuracy will suffer — see Appendix C for adding self-captured data.
- **Step 4 accuracy below 80%.** Almost always insufficient data diversity. Suggest proceeding anyway and noting the per-class confusion in Step 10.4, or adding capture sessions (Appendix C).
- **Step 5 `trtexec` "no implementation" error.** Participant ran trtexec on the host instead of inside the container. The host does not have trtexec — it lives in `sign-tutor-triton-1`.
- **Step 6 model `UNAVAILABLE`.** 90% of the time `config.pbtxt` is missing or has a typo. Check `docker logs sign-tutor-triton-1`.
- **Step 8 quality bar stays red.** Usually the smoothing window not yet filled, or the model's accuracy is genuinely low. Tell participants to hold the sign steady for 2+ seconds.

### B.3 Pre-session setup checklist

- [ ] Both containers running: `docker compose ps`
- [ ] ASL classifier serving: `curl -s http://localhost:8010/v2/models/asl_classifier/ready`
- [ ] ISL classifier NOT present: `curl -s http://localhost:8010/v2/models/isl_classifier` should error
- [ ] `datasets/isl_frames/` present with 26 subdirectories
- [ ] `languages/isl/config.yaml` present
- [ ] `languages/isl/references/` contains 26 PNG files
- [ ] `checkpoints/isl/` does not exist (clean state)
- [ ] `datasets/isl_landmarks.csv` does not exist (clean state)
- [ ] Webcam functional, browser can access `promaxgb10.local:7860`
- [ ] Adequate lighting at the demo station

### B.4 Conversation starters for the wrap-up

- "Where in your environment do you do vision inference today?"
- "Have you looked at Triton for any of your inference serving?"
- "What does your training-to-deployment workflow look like at the moment?"
- "What's the biggest bottleneck in your current MLOps pipeline?"

The point of the lab is not to teach sign language. It is to give the participant first-hand experience of NVIDIA's vision-AI stack so they can have an informed conversation about applying it to their actual problems.

---

## Appendix C — Adding self-captured data (optional)

If the pre-collected `isl_frames` dataset yields insufficient accuracy (validation accuracy below 80%), adding self-captured data can help significantly. Each additional signer improves model generalisation.

### C.1 Prerequisites

A webcam attached to the lab workstation and the ISL alphabet reference open for guidance.

### C.2 Run a capture session

```bash
docker exec -it sign-tutor-tutor-app-1 \
    python training/capture.py \
        --lang isl \
        --output datasets/self_capture/isl \
        --frames-per-letter 30
```

The script walks through each letter: press Enter when ready, hold the sign for the duration, then move to the next. It captures landmark vectors directly (no images are saved).

> **Privacy note:** The capture script writes landmark coordinates only. No images are ever saved to disk — participants can run this without consent forms, and the resulting data can be shared without privacy implications.

### C.3 Merge and retrain

After capture, append the self-captured data to your existing CSV and re-run training:

```bash
cat datasets/isl_landmarks.csv datasets/self_capture/isl/landmarks.csv \
    > datasets/isl_landmarks_combined.csv

docker exec sign-tutor-tutor-app-1 python training/train_classifier.py \
    --dataset datasets/isl_landmarks_combined.csv \
    --epochs 50 \
    --checkpoint-dir checkpoints/isl \
    --csv-file checkpoints/isl/train_log.csv
```

Then re-export and rebuild the TRT engine as in Sections 5 and 6.

# Data Acquisition & Preparation

**Sign Language Tutor — Cross-cutting**
**Spec 4 of 4**

---

## 1. Why this is a separate spec

Data is the single biggest delivery risk for this project. The ML pipeline, the NVIDIA stack, and the UI are all well-trodden. Getting representative, balanced, high-quality landmark data for three sign languages — one of which (ISL) has only a single notable public dataset — is the part that has caused similar projects to stall.

This spec covers, per language: where the data comes from, how to obtain it, how to prepare it, the augmentation strategy, and a self-capture protocol for filling gaps. Read it before starting Phase 1 of the Module 1 build.

---

## 2. Strategy at a glance

| Language | Primary source | Top-up | Estimated effort |
|---|---|---|---|
| ASL | Kaggle "ASL Alphabet" (Akash Nagaraj) | None needed for Module 1 | ~1 hour download + extraction |
| ISL | DCU **ISL-HS** (Oliveira et al., 2017) | Self-capture for diversity | ~half a day |
| BSL | Self-capture (no clean public alphabet dataset) | Augmentation | ~2 days |

**Key insight:** because we use MediaPipe to extract 21-landmark vectors and **discard the original pixels**, the size of the source images stops mattering. A 200×200 ASL image and a 1080p self-capture both end up as the same 63 floats. This makes mixing data sources practical and dramatically reduces the per-language data burden.

---

## 3. ASL — Kaggle "ASL Alphabet" dataset

### 3.1 The dataset

- **Author:** Akash Nagaraj
- **URL:** https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **Size:** ~87,000 images, 200×200 pixels, colour
- **Classes:** 29 (A–Z plus `space`, `delete`, `nothing` — we use only A–Z for Module 1)
- **Per class:** ~3,000 images
- **License:** GPL-2.0 on Kaggle — fine for an internal lab/demo; check before any external publication.

This is the de-facto standard ASL alphabet dataset — every project linked from the search results uses it. Plenty enough data, well-balanced, and freely available.

### 3.2 Download (Kaggle CLI on the GB10)

```bash
# One-time setup inside the tutor-app container
pip install kaggle
mkdir -p ~/.kaggle
# Place kaggle.json (downloaded from kaggle.com/settings) in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download
mkdir -p datasets/asl_alphabet
cd datasets/asl_alphabet
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip
```

This pulls roughly 1 GB. Run it once; the resulting landmark CSV is what you'll keep — the raw images can then be deleted to save space.

### 3.3 Known caveats

- **Background uniformity.** All ~3,000 images per letter are very similar — same person, same room, same lighting. Models trained on this dataset alone often overfit and fail in real-world demo conditions.
- **Single signer.** No diversity in hand size, skin tone, or signing style.
- **Mitigation:** the [Dan Rasband ASL Alphabet Test set](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) provides 870 varied-background test images — use it as a held-out validation set, **never** as training data.

### 3.4 Validation expectation

A model trained purely on this dataset should hit 95%+ on a held-out split of the same data, and 80–90% on the Rasband test set. The latter is the more honest number for what you'll see in a live demo.

---

## 4. ISL — DCU ISL-HS dataset

### 4.1 The dataset

- **Author:** Marlon Oliveira et al., Dublin City University, 2017
- **GitHub:** https://github.com/marlondcu/ISL
- **Paper:** "A Dataset for Irish Sign Language Recognition" (Oliveira et al., 2017)
- **Size:** 58,114 images covering 23 hand-shapes (the static letters)
- **Subjects:** 6 people (3 male, 3 female), each performing each shape 3 times
- **Coverage:** 23 static letters (A–Z except J, X, Z) + 3 dynamic gestures (J, X, Z)
- **Format:** Greyscale, background-removed via pixel thresholding; both video and extracted frames provided
- **Capture method:** Each shape performed while moving the arm in an arc from vertical to horizontal — simulates rotation that occurs in real conversation
- **License:** Check the GitHub repo at download time. Always credit the authors in any presentation; mention the dataset in any public-facing demo materials.

### 4.2 Download

```bash
mkdir -p datasets/isl_hs
cd datasets/isl_hs
git clone https://github.com/marlondcu/ISL.git
# The frames are organised by signer and letter inside the repo
```

### 4.3 Why ISL-HS works for our pipeline

Despite being greyscale and background-removed, **MediaPipe still detects hands in greyscale images** (it operates on luminance internally for hand region proposals). And we throw away the pixels anyway — only the 21 landmarks survive. Test this on a handful of frames before doing the full extraction; if MediaPipe struggles on the background-removed frames, falling back to the original videos in the repo is the plan B.

### 4.4 Known caveats

- **Only 6 subjects.** Limited diversity in hand size, skin tone, and signing style. This is the largest published ISL dataset, so we work with what exists.
- **3 dynamic letters (J, X, Z).** ISL has more dynamic letters than ASL; we treat them the same way — capture as static snapshots at the most distinctive point of the motion. This is a known limitation we document on-screen in the UI.
- **Lab conditions.** All shots taken in similar lighting. Augmentation matters here.
- **Greyscale + background-removed.** Visually different from a webcam feed in a CSC demo room. Augmentation alone won't fully bridge this — self-capture top-up is recommended (Section 6).

### 4.5 Reported baselines from previous work

This dataset is well-studied. Published baselines:

- PCA + k-NN (Oliveira 2017): **95%** accuracy
- Random forest + engineered features: **96.7%**
- CNN (Oliveira follow-up): **99%**

Our landmark-based MLP should land in the 92–97% range on a held-out split. Exceeding the published numbers isn't the goal — generalising to live webcam input is.

---

## 5. BSL — what's actually available

### 5.1 The honest picture

There is **no clean, freely-downloadable BSL alphabet dataset** equivalent to ISL-HS or the ASL Kaggle set. The major BSL datasets are:

**BOBSL (Oxford VGG / BBC)**
- URL: https://www.robots.ox.ac.uk/~vgg/data/bobsl/
- ~1,400 hours of BSL-interpreted BBC broadcast footage
- Continuous signing, not isolated alphabet — overkill and wrong shape for our use case
- Requires a BBC R&D Terms of Use agreement and personal password approval
- Listed for completeness; not what we need

**FS23K (Oxford VGG)**
- arXiv: https://arxiv.org/abs/2603.19523
- Released 2026 by Chan, Kwon, and Zisserman
- 23,074 fingerspelling instances with letter-level annotations, derived from BOBSL
- This is fingerspelling-specific, but **continuous** — letter sequences, not isolated letters with clean labels — and inherits BOBSL's access requirements
- Conceivably useful for Module 3 (sequence recognition), not Module 1

**BSL alphabet reference charts** (free, useful for reference imagery in the UI):
- British Deaf Association: https://bda.org.uk/wp-content/uploads/2023/04/BSL-Fingerspelling-Chart.pdf
- Sign Language Week: https://signlanguageweek.org.uk/wp-content/uploads/2023/02/SLW-BSL-Fingerspelling-Chart-2023.pdf
- british-sign.co.uk: https://www.british-sign.co.uk/fingerspelling-alphabet-charts/ (right- and left-handed versions)

These are reference images, not training data.

### 5.2 Plan for BSL

**Self-capture is the realistic path.** The two-handed nature of BSL also means existing single-hand datasets are useless for it anyway. A capture protocol for self-collection is in Section 6.

The good news: BSL's two-handed alphabet is more visually distinctive per letter than ASL's, and a 126-feature input gives the model more to work with. Modest dataset sizes (200–400 samples per letter) should suffice.

---

## 6. Self-capture protocol

A single, reusable protocol covers ISL top-up and the entire BSL dataset. Designed to be runnable solo with a webcam and a few minutes per letter.

### 6.1 Goals of the protocol

- **Diversity over volume.** It is more valuable to have 100 samples across varied conditions than 1,000 in identical conditions.
- **Reproducible.** Anyone in the CSC could follow this script and produce data that mixes cleanly with existing captures.
- **Fast.** Aim for a full alphabet capture session in under 30 minutes per signer.

### 6.2 Per-letter target

- 200 frames per letter for ISL top-up (mixing with ISL-HS).
- 400 frames per letter for BSL (this is the primary source).

### 6.3 Capture conditions (cycle through during the session)

- 3 different backgrounds (plain wall, busy office, a doorway).
- 3 different lighting conditions (overhead room light, daylight from a window, dim/lamp).
- 2 distances from camera (~1 m and ~1.5 m).
- 2 angles (head-on and ~15° offset).
- Slow rotation of the wrist through ±20° during capture (mirrors the ISL-HS arc protocol).

For each letter: hold the sign and slowly rotate while the recorder runs for ~10 seconds at 30 FPS = 300 frames. Repeat with a different condition. Discard frames where MediaPipe failed to detect the hand(s).

### 6.4 Capture script

```python
# training/capture.py
import cv2, time, csv
from pathlib import Path
from src.capture.hands import HandTracker
from src.features import build_feature_vector
from src.registry import load_registry

def capture_letter(language_code: str, letter: str,
                   duration_seconds: float = 10.0,
                   condition_label: str = "default"):
    """Live capture: shows webcam, records landmarks, drops missed-detection frames."""
    langs = load_registry()
    lang = langs[language_code]
    tracker = HandTracker()
    out_path = Path(f"datasets/self_capture/{language_code}/landmarks.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    end = time.monotonic() + duration_seconds
    rows = []
    while time.monotonic() < end:
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = tracker.process(rgb)
        feat = build_feature_vector(lang, detections)
        if feat is None:
            cv2.putText(frame, "NO HAND" if lang.input_hands == 1
                        else "NEED BOTH HANDS",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)
        else:
            rows.append([letter, condition_label, *feat.tolist()])
            cv2.putText(frame, f"REC {letter}: {len(rows)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(out_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {len(rows)} samples for {letter} ({condition_label})")
```

The script writes directly to landmark CSV — no images stored, no privacy concerns about face data. This matters: webcam footage of signers is sensitive; storing only landmarks is a meaningful privacy improvement and removes any face data entirely.

### 6.5 Capture session structure

Run a shell wrapper that walks through the alphabet automatically:

```bash
# training/run_capture_session.sh
#!/usr/bin/env bash
set -euo pipefail
LANG_CODE=${1:-isl}
CONDITION=${2:-condition1}
for L in A B C D E F G H I J K L M N O P Q R S T U V W X Y Z; do
  read -p "Ready for letter ${L}? Press enter..."
  python -m training.capture --lang "$LANG_CODE" --letter "$L" \
                             --duration 10 --condition "$CONDITION"
done
```

A single signer producing one full pass of the alphabet across all conditions takes roughly 30–40 minutes.

### 6.6 Recommended signers

Capture from at least three different people if possible — a single signer overfits the model to that one person's hand geometry. For BSL specifically, prioritise getting at least one signer who is comfortable with the two-handed alphabet, even if it's just from a YouTube tutorial.

> **Cultural note:** for any data captured from members of the Deaf community, get explicit consent for the lab use, and offer to share the resulting model with them. This is both the right thing to do and meaningfully improves credibility if the demo is ever shown externally.

---

## 7. Landmark extraction pipeline

A single script handles all source datasets, producing a uniform per-language CSV.

### 7.1 Output schema

```
label,source,condition,f0,f1,...,fN
A,isl_hs,signer3_take2,0.0,0.0,0.0,0.123,...
A,self_capture,condition1,0.0,0.0,0.0,0.119,...
```

Where `N = 62` for one-handed (ASL/ISL) and `N = 125` for two-handed (BSL).

### 7.2 Extraction script

```python
# training/extract_landmarks.py
import argparse, csv
from pathlib import Path
import cv2
import numpy as np
from src.capture.hands import HandTracker
from src.features.normalise import normalise_one_hand
from src.features.two_hand import build_two_hand_vector

def extract_from_image_dir(src_dir: Path, hands: int, source_tag: str,
                           out_csv: Path):
    tracker = HandTracker(static_image_mode=True)
    written = skipped = 0
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        for letter_dir in sorted(src_dir.iterdir()):
            if not letter_dir.is_dir():
                continue
            letter = letter_dir.name.upper()
            for img_path in letter_dir.glob("*.[jp][pn]g"):
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped += 1
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                detections = tracker.process(rgb)
                if hands == 1:
                    if not detections:
                        skipped += 1
                        continue
                    feat = normalise_one_hand(detections[0][1])
                else:
                    feat = build_two_hand_vector(detections)
                    if feat is None:
                        skipped += 1
                        continue
                writer.writerow([letter, source_tag, img_path.stem,
                                 *feat.tolist()])
                written += 1
    print(f"Wrote {written}, skipped {skipped} (no hand detected)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--hands", type=int, choices=[1, 2], required=True)
    parser.add_argument("--source-tag", required=True)
    args = parser.parse_args()
    extract_from_image_dir(args.src, args.hands, args.source_tag, args.dst)
```

### 7.3 Per-language driver

```bash
# ASL
python -m training.extract_landmarks \
       --src datasets/asl_alphabet/asl_alphabet_train \
       --dst languages/asl/landmarks.csv \
       --hands 1 --source-tag kaggle_asl

# ISL — extract from ISL-HS frames (one folder per letter)
python -m training.extract_landmarks \
       --src datasets/isl_hs/Frames-Sentence-Level \
       --dst languages/isl/landmarks.csv \
       --hands 1 --source-tag isl_hs

# Append self-capture top-up (already in CSV format, just concat)
cat datasets/self_capture/isl/landmarks.csv >> languages/isl/landmarks.csv
```

### 7.4 Detection-rate sanity check

After extraction, log the detection rate per letter. Anything below ~85% is a smell:

```python
# training/check_detection_rate.py
import pandas as pd
df = pd.read_csv("languages/isl/landmarks.csv", header=None)
df.columns = ["label", "source", "frame_id"] + [f"f{i}" for i in range(63)]
counts = df.groupby(["source", "label"]).size().unstack(fill_value=0)
print(counts)
```

Letters with very low detection counts often have hand poses where MediaPipe loses track (heavily occluded fingers, hand turned away from camera). For these, prioritise self-capture top-up.

---

## 8. Augmentation

Landmark-based augmentation is much cleaner than pixel-based augmentation — no risk of distorting the hand into something unrealistic.

### 8.1 Recommended augmentations

Apply at training time, not pre-computed (avoids dataset bloat):

| Augmentation | Parameter | Why |
|---|---|---|
| Gaussian noise | σ = 0.01 on each coordinate | MediaPipe itself is noisy; teaches model robustness |
| In-plane rotation | ±10° around the wrist | Simulates wrist tilt during signing |
| Small scaling | ×0.95 to ×1.05 | Defensive — should already be neutralised by normalisation |
| Mirror flip (one-handed only) | x → -x with 50% probability | Doubles effective data; one-handed signs are mostly handedness-agnostic |

### 8.2 What NOT to augment

- **Don't mirror BSL data.** Two-handed signs depend on dominant/subordinate ordering — mirroring breaks the semantics.
- **Don't translate post-normalisation.** Normalisation already places the wrist at origin; adding translation would break that invariance.
- **Don't add per-finger jitter independent of the hand structure.** It produces unrealistic hand poses.

### 8.3 Implementation sketch

```python
# training/augment.py
import numpy as np

def augment_one_hand(vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """vec: (63,) normalised landmark vector."""
    pts = vec.reshape(21, 3).copy()
    # noise
    pts += rng.normal(0, 0.01, pts.shape).astype(np.float32)
    # in-plane rotation
    theta = np.deg2rad(rng.uniform(-10, 10))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = pts @ R.T
    # mirror with 50% probability
    if rng.random() < 0.5:
        pts[:, 0] = -pts[:, 0]
    return pts.flatten()
```

---

## 9. Data quality gates (must pass before training)

These are simple pandas checks. Wire them into the start of the training script so a bad dataset fails fast.

| Check | Threshold | Action if failed |
|---|---|---|
| Per-class sample count | min/max ratio ≤ 0.5 | Capture more for under-represented letters |
| NaN / Inf values | zero | Re-extract; debug normalisation |
| Vector length | all rows == expected dim | Mixing of one-hand and two-hand data — fix source tags |
| Duplicate rows | < 1% | Source has duplicate frames; deduplicate |
| Source diversity | ≥ 2 sources for ISL, ≥ 3 sessions for self-capture | Add more sources |
| MediaPipe detection rate per letter | ≥ 80% | Letter is hard to detect; prioritise top-up |

### 9.1 Quality check script

```python
# training/check_quality.py
import pandas as pd, sys
import numpy as np

def check(path: str, expected_dim: int) -> int:
    df = pd.read_csv(path, header=None)
    n_meta = 3  # label, source, frame_id
    feature_cols = df.shape[1] - n_meta
    failures = []

    if feature_cols != expected_dim:
        failures.append(f"Wrong dim: got {feature_cols}, expected {expected_dim}")

    counts = df.iloc[:, 0].value_counts()
    ratio = counts.min() / counts.max()
    if ratio < 0.5:
        failures.append(f"Imbalance: min/max class ratio = {ratio:.2f}")

    feats = df.iloc[:, n_meta:].astype(float)
    if feats.isna().any().any():
        failures.append("NaN values present")
    if np.isinf(feats.to_numpy()).any():
        failures.append("Inf values present")

    sources = df.iloc[:, 1].nunique()
    if sources < 2:
        failures.append(f"Only {sources} source(s); want >= 2")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("Data quality: OK")
    return 0

if __name__ == "__main__":
    sys.exit(check(sys.argv[1], int(sys.argv[2])))
```

---

## 10. Per-language data acquisition checklist

Tear-out checklist for tracking progress.

### ASL
- [ ] Kaggle credentials configured on GB10
- [ ] ASL Alphabet dataset downloaded and unzipped
- [ ] `extract_landmarks.py` run on the train split
- [ ] Rasband test set downloaded as held-out validation
- [ ] Quality check passes
- [ ] Per-letter detection rate ≥ 85%

### ISL
- [ ] ISL-HS GitHub repo cloned
- [ ] Sample frames sanity-checked through MediaPipe
- [ ] Full extraction completed
- [ ] Self-capture session 1 completed (one signer, three conditions)
- [ ] Self-capture session 2 completed (different signer or different day)
- [ ] Quality check passes
- [ ] Per-letter detection rate ≥ 85%
- [ ] J/X/Z dynamic letters handled (decision: skip or capture as static)
- [ ] Attribution recorded for any external contributors

### BSL
- [ ] Decision made on dynamic letters (H, J — both involve motion in BSL)
- [ ] Self-capture session 1 completed
- [ ] Self-capture session 2 completed (different signer if possible)
- [ ] Self-capture session 3 completed (additional conditions)
- [ ] Quality check passes
- [ ] Per-letter detection rate ≥ 80% (lower bar — two hands harder to track)
- [ ] Reference imagery sourced from BDA chart (under their licensing)

---

## 12. Augmentation tooling

### 12.1 Two augmentation strategies, two reasons

There are two distinct places in the pipeline where augmentation can happen, and they solve different problems. **Use both, for different reasons.**

| Strategy | Where it sits | What it solves | Tooling |
|---|---|---|---|
| **Image-space** | Before MediaPipe extraction | Diversifies the kinds of frames MediaPipe sees, producing more varied landmark vectors out the other side. Catches MediaPipe failure modes. | NVIDIA DALI (preferred) or Albumentations |
| **Landmark-space** | At training time, on each batch | Cheap regularisation — small jitters, rotations, mirror flips applied to the 63-float vector. | Plain NumPy (Spec 4 Section 8) |

The reason to do image-space augmentation specifically — even though we're throwing the pixels away — is that **MediaPipe is fixed**. You can't make it smarter, but if you feed it varied input images, you get more varied landmarks out, which is exactly what's missing from a 6-signer dataset.

### 12.2 Why this matters more for ISL-HS than for ASL Kaggle

ASL Kaggle has 87,000 samples per signer's room. ISL-HS has 6 signers, similar conditions across all of them, greyscale, background-removed. Image-space augmentation is the cheapest way to bridge the gap to a real demo room — varying lighting, contrast, and slight geometry until the landmark distribution looks more like what you'll capture live on the GB10 webcam.

### 12.3 NVIDIA DALI

The NVIDIA-native option. Strong fit for the project's NVIDIA narrative.

- **Docs:** https://developer.nvidia.com/dali
- **Auto-augment reference:** https://docs.nvidia.com/deeplearning/dali/user-guide/docs/auto_aug/augmentations.html
- **Status on GB10:** Pre-installed in `nvcr.io/nvidia/pytorch` containers. Nothing to install.
- **Strengths:** GPU-accelerated, including JPEG decode; 70+ operators; integrates cleanly with PyTorch/TensorFlow data loaders.
- **Weaknesses:** Operators are rigid — you can't easily compose custom geometric transforms the way you can in OpenCV. For our use case (rotation, brightness, contrast, noise) this is fine.

### 12.4 Recommended DALI augmentation set for ISL-HS

```python
# training/augment_isl_hs.py
from nvidia.dali import pipeline_def, fn, types

@pipeline_def(batch_size=64, num_threads=4, device_id=0)
def augmentation_pipe(file_root: str):
    """Image-space augmentation. Output images are then fed through
       MediaPipe to produce landmark vectors as normal."""
    images, labels = fn.readers.file(
        file_root=file_root,
        random_shuffle=True,
        name="Reader"
    )
    images = fn.decoders.image(images, device="mixed",
                               output_type=types.RGB)

    # Geometric — careful: must not distort the hand structure
    angle = fn.random.uniform(range=(-15.0, 15.0))
    images = fn.rotate(images, angle=angle, fill_value=0,
                       keep_size=True)

    # Photometric — still useful on greyscale; helps simulate lighting
    images = fn.brightness_contrast(
        images,
        brightness=fn.random.uniform(range=(0.8, 1.2)),
        contrast=fn.random.uniform(range=(0.85, 1.15))
    )

    # Mild noise — simulates webcam sensor noise
    images = fn.noise.gaussian(images, stddev=0.02)

    return images, labels
```

The augmented images **are not saved to disk**. They are streamed straight into MediaPipe, which produces the landmark vectors that get appended to the language's CSV. The original ISL-HS images remain untouched.

### 12.5 Augmentation budget

A reasonable target for ISL-HS: produce **3 augmented frames per original frame**, distributed across different augmentation parameter draws. With 58,114 originals, that gives ~232,000 input frames into MediaPipe. After accounting for some MediaPipe detection failures, you should land at around 200,000 landmark rows — very comfortable for the size of model we're training.

For the smaller self-capture top-up (a few hundred per letter), 5× augmentation is reasonable.

### 12.6 What NOT to augment

These are the operations to actively avoid — they will quietly degrade your model rather than improve it.

| Operation | Why not |
|---|---|
| Elastic transforms / grid distortion | Distorts the hand structure itself; the label no longer describes what's in the image. |
| Heavy rotation (>30°) | A hand at 45° is a different sign in some cases (W vs M, etc.). Stay within ±15°. |
| Colour shifts on ISL-HS | Dataset is greyscale and background-removed. Colour augmentation is a no-op or worse. |
| Horizontal flip on BSL data | Two-handed signs depend on dominant/subordinate ordering — flipping breaks the semantics. |
| Vertical flip (any language) | No real-world signing happens upside down. Pure noise. |
| Random crop tight enough to clip the hand | Cropped fingers = lost label information. |
| Heavy blur | MediaPipe needs to be able to find the hand. Blur it past recognition and you just lose data. |

### 12.7 Albumentations as an alternative

If you find DALI's rigidity painful, **Albumentations** (https://albumentations.ai/) is the leading alternative.

- **Strengths:** Simpler Python API, 70+ transforms, very actively maintained, fastest CPU-side library, has explicit support for keypoints which is useful if you ever do hybrid image-and-landmark training.
- **Weaknesses:** CPU-only by default (Kornia is the GPU equivalent). Less of an "NVIDIA story" for the demo.
- **License:** Dual AGPL/Commercial. Fine for internal lab use. Check before any external distribution.

For this project, **start with DALI** — it's already in your container and fits the narrative. Reach for Albumentations only if you hit a specific limitation.

### 12.8 Validating that augmentation actually helped

Augmentation isn't free — it adds training time and can occasionally hurt if overdone. Run two training runs and compare:

- **Run A:** ISL-HS landmarks only, no augmentation.
- **Run B:** ISL-HS landmarks + DALI image-space + landmark-space augmentation.

Measure both on the **same held-out test set**, ideally including some self-capture frames the model has never seen. If Run B is not at least 2 percentage points better on the held-out set, the augmentation is either misconfigured or hitting diminishing returns. Document the result in `docs/augmentation_ablation.md` for the project record.

### 12.9 Summary recommendation

For ISL on the GB10:

1. Image-space augmentation with DALI before MediaPipe extraction (3× volume).
2. Landmark-space augmentation in NumPy at training time (free, always on).
3. Run the ablation in 12.8 once, document the lift.
4. Don't over-augment. 3× image-space + per-batch landmark jitter is a strong baseline; bigger numbers rarely help.

---

## 13. Honest summary

ASL is essentially solved — Kaggle dataset gets you to a working model in a day. ISL is well-supported by the DCU dataset for a baseline, but live demo quality will require self-capture top-up and DALI-based augmentation. BSL is the hard one and is best treated as primarily a self-capture exercise, with the public continuous-signing datasets (BOBSL/FS23K) reserved for any future Module 3 work.

If time pressure forces a cut, **drop BSL from Module 1 phase 1**, ship ASL + ISL solidly, then return for BSL once the framework is proven on two languages. The architecture supports this — that's the whole point of the language registry.

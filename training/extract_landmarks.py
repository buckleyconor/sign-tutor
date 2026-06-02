"""Landmark extraction from image directories.

Handles Kaggle-style (letter/*/*.jpg) and ISL-HS-style (person/PersonX-A-T-F.jpg) layouts.
Outputs a CSV: label, source, frame_id, feat_0, ..., feat_62

Usage:
  python training/extract_landmarks.py \\
    --src datasets/asl_dataset/a \\
    --dst languages/asl/landmarks.csv \\
    --hands 1 \\
    --source-tag kaggle_asl
"""
import argparse, csv
from pathlib import Path
import cv2
import numpy as np
from src.capture.hands import HandTracker
from src.features.normalise import normalise_one_hand

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def iter_images(directory: Path):
    """Yield image files matching common extensions."""
    for ext in IMAGE_EXTENSIONS:
        yield from directory.glob(f"*{ext}")


def _extract_letter_from_filename(filename: str) -> str | None:
    """Try to extract a single letter from an image filename.

    Handles ISL-HS style: Person1-A-1-45.jpg → 'A'
    """
    stem = Path(filename).stem  # e.g. 'Person1-A-1-45'
    parts = stem.split("-")
    for part in parts:
        if len(part) == 1 and part.isalpha() and part.isupper():
            return part
    return None


def _is_letter_dir(name: str) -> bool:
    """Check if a directory name looks like a letter class (A-Z)."""
    return len(name) == 1 and name.upper().isalpha()


def extract_from_image_dir(src_dir: Path, hands: int, source_tag: str,
                           out_csv: Path):
    tracker = HandTracker(static_image_mode=True)
    written = skipped = 0

    # Detect format: letter-named subdirs (Kaggle) or person-named subdirs (ISL-HS)
    children = [d for d in sorted(src_dir.iterdir()) if d.is_dir()]
    if children and all(_is_letter_dir(c.name) for c in children):
        # Kaggle-style: letter/*/*.jpg
        written, skipped = _extract_from_letter_dirs(src_dir, hands,
                                                     source_tag, out_csv, tracker)
    else:
        # ISL-HS-style: Person*/PersonX-A-T-F.jpg
        written, skipped = _extract_from_flat_dirs(src_dir, hands,
                                                   source_tag, out_csv, tracker)

    print(f"Wrote {written}, skipped {skipped} (no hand detected)")


def _extract_from_letter_dirs(src_dir: Path, hands: int, source_tag: str,
                              out_csv: Path, tracker: HandTracker) -> tuple[int, int]:
    """Extract from letter-named subdirectories (Kaggle-style)."""
    written = skipped = 0
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        for letter_dir in sorted(src_dir.iterdir()):
            if not letter_dir.is_dir():
                continue
            letter = letter_dir.name.upper()
            if not _is_letter_dir(letter):
                continue
            for img_path in iter_images(letter_dir):
                if _process_image(img_path, hands, letter, source_tag,
                                  writer, tracker):
                    written += 1
                else:
                    skipped += 1
    return written, skipped


def _extract_from_flat_dirs(src_dir: Path, hands: int, source_tag: str,
                            out_csv: Path, tracker: HandTracker) -> tuple[int, int]:
    """Extract from flat image files under person dirs (ISL-HS-style)."""
    written = skipped = 0
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        for person_dir in sorted(src_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            for img_path in iter_images(person_dir):
                letter = _extract_letter_from_filename(img_path.name)
                if letter is None:
                    skipped += 1
                    continue
                if _process_image(img_path, hands, letter, source_tag,
                                  writer, tracker):
                    written += 1
                else:
                    skipped += 1
    return written, skipped


def _add_background(img: np.ndarray) -> np.ndarray:
    """Add a solid background to background-removed frames (ISL-HS).

    ISL-HS frames are thresholded grayscale silhouettes — MediaPipe can't
    detect hands without surrounding texture. This overlays the hand on a
    neutral gray background so MediaPipe can locate it.
    """
    if img.ndim == 3 and img.shape[2] >= 3:
        # Already RGB
        return img
    # Grayscale → RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Create background from median pixel color (should be the removed bg)
    bg_color = np.median(img, axis=(0, 1)).astype(np.uint8)
    # Overlay hand (non-background pixels) onto solid background
    mask = np.any(img != bg_color, axis=2).astype(np.uint8)
    result = np.full_like(img, bg_color)
    result[mask == 1] = img[mask == 1]
    return result


def _process_image(img_path: Path, hands: int, letter: str, source_tag: str,
                   writer: csv.writer, tracker: HandTracker) -> bool:
    """Process a single image and write its landmarks to the CSV.

    Returns True if successfully written, False otherwise.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    rgb = _add_background(img)
    detections = tracker.process(rgb)
    if hands == 1:
        if not detections:
            return False
        feat = normalise_one_hand(detections[0][1])
    else:
        return False  # Only one-handed extraction supported
    writer.writerow([letter, source_tag, img_path.stem, *feat.tolist()])
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--hands", type=int, choices=[1], default=1,
                        help="Number of input hands (only 1 supported)")
    parser.add_argument("--source-tag", required=True)
    args = parser.parse_args()
    extract_from_image_dir(args.src, args.hands, args.source_tag, args.dst)

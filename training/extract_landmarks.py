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

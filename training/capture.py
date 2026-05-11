import argparse, csv
from pathlib import Path
import cv2
import time
import numpy as np
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="isl")
    parser.add_argument("--letter", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--condition", default="default")
    args = parser.parse_args()
    capture_letter(args.lang, args.letter, args.duration, args.condition)

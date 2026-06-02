"""Extract frames from ISL-HS .mov videos to Kaggle-style letter directories.

The ISL-HS videos are in datasets/isl_hs/videos/Person{1-6}/{letter} ({1-3}).mov
This script extracts frames from each video and saves them to a temporary
Kaggle-style directory (e.g. /tmp/isl_frames/A/*.jpg) so the existing
extract_landmarks.py pipeline can process them.

Usage:
    python -m training.extract_videos \
        --src datasets/isl_hs/videos \
        --dst /tmp/isl_frames \
        --fps 6               # frames per second to extract
"""

import argparse
from pathlib import Path

import cv2


MOV_EXTENSIONS = (".mov",)


def extract_video_frames(video_path: Path, output_dir: Path,
                         fps: int = 6) -> int:
    """Extract frames from a single .mov video.

    Args:
        video_path: Path to the .mov file (e.g. Person1/a (1).mov).
        output_dir: Base output directory (e.g. /tmp/isl_frames).
        fps: Frames per second to extract from the video.

    Returns:
        Number of frames extracted.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARN: Could not open {video_path.name}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # fallback
    frame_interval = max(1, int(video_fps / fps))

    letter = _extract_letter_from_video_name(video_path)
    letter_dir = output_dir / letter
    letter_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            out_path = letter_dir / f"{video_path.stem}-{count:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            count += 1
        frame_idx += 1

    cap.release()
    return count


def _extract_letter_from_video_name(video_path: Path) -> str:
    """Extract the letter from a video name like 'a (1).mov'."""
    stem = video_path.stem  # e.g. "a (1)"
    parts = stem.split()
    if parts:
        return parts[0].upper()
    return "?"


def extract_all_videos(src_dir: Path, dst_dir: Path, fps: int = 6) -> int:
    """Extract frames from all .mov videos in person subdirectories.

    Args:
        src_dir: Top-level directory with Person{1-6}/ subdirs.
        dst_dir: Output directory for all extracted frames (letter/*.jpg).
        fps: Frames per second to extract.

    Returns:
        Total number of frames extracted.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for person_dir in sorted(src_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        print(f"Processing {person_dir.name}...")
        videos = []
        for ext in MOV_EXTENSIONS:
            videos.extend(person_dir.glob(f"*{ext}"))

        # Sort by letter then shot number for consistent ordering
        videos.sort(key=lambda p: (p.stem.split()[0], p.stem.split()[1]))

        for video_path in videos:
            count = extract_video_frames(video_path, dst_dir, fps)
            letter = _extract_letter_from_video_name(video_path)
            print(f"  {video_path.name} -> {letter}: {count} frames")
            total += count

    print(f"\nExtracted {total} frames to {dst_dir}")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from ISL-HS .mov videos"
    )
    parser.add_argument("--src", type=Path, required=True,
                        help="Directory with Person{1-6}/ subdirs")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Output directory for frames (letter/*.jpg)")
    parser.add_argument("--fps", type=int, default=6,
                        help="Frames per second to extract (default: 6)")
    args = parser.parse_args()
    extract_all_videos(args.src, args.dst, args.fps)

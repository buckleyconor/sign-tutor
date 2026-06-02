#!/usr/bin/env bash
# Extract frames from ISL-HS .mov videos using ffmpeg.
# Usage: bash /app/training/extract_frames.sh
#
# Reads videos from /app/datasets/isl_hs/videos/Person{1-6}/*.mov
# Outputs frames to /app/datasets/isl_frames/{A-Z}/*.jpg at 6fps

SRC="/app/datasets/isl_hs/videos"
DST="/app/datasets/isl_frames"
FPS=6

mkdir -p "$DST"

total=0
skipped=0

for person_dir in "$SRC"/Person*; do
    [ -d "$person_dir" ] || continue
    person=$(basename "$person_dir")
    echo "=== $person ==="
    
    for video in "$person_dir"/*.mov; do
        [ -f "$video" ] || continue
        # Extract letter from filename: "a (1).mov" -> "A"
        stem=$(basename "$video" .mov)
        letter=$(echo "$stem" | awk '{print toupper($1)}')
        
        # Create output dir
        out_dir="$DST/$letter"
        mkdir -p "$out_dir"
        
        # Count frames in output dir
        frame_count=$(ls "$out_dir"/*.jpg 2>/dev/null | wc -l)
        if [ "$frame_count" -gt 0 ]; then
            echo "  SKIP $video (already have $frame_count frames in $letter/)"
            continue
        fi
        
        # Extract frames at 6fps
        frames=$(ffmpeg -i "$video" -vf "fps=$FPS" -q:v 2 "$out_dir/%04d.jpg" 2>/dev/null)
        new_count=$(ls "$out_dir"/*.jpg 2>/dev/null | wc -l)
        
        if [ "$new_count" -gt 0 ]; then
            echo "  $video -> $letter/: $new_count frames"
            total=$((total + new_count))
        else
            echo "  WARN: Could not extract from $video"
            skipped=$((skipped + 1))
        fi
    done
done

echo ""
echo "Done: $total frames extracted, $skipped videos skipped"

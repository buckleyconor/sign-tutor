#!/usr/bin/env bash
# Extract frames from all ISL-HS .mov videos, one person at a time,
# then merge into a single Kaggle-style directory.
#
# Usage: bash /app/training/extract_all_frames.sh

SRC="/app/datasets/isl_hs/videos"
TEMP="/tmp/isl_person_frames"
DST="/app/datasets/isl_frames"
FPS=6
mkdir -p "$DST"

# Clean up any previous partial extraction
rm -rf "$TEMP"
mkdir -p "$TEMP"

total=0
total_skipped=0

for person_dir in "$SRC"/Person*; do
    [ -d "$person_dir" ] || continue
    person=$(basename "$person_dir")
    echo "=== Extracting $person ==="
    
    person_temp="$TEMP/$person"
    mkdir -p "$person_temp"
    
    extracted=0
    
    for video in "$person_dir"/*.mov; do
        [ -f "$video" ] || continue
        stem=$(basename "$video" .mov)
        letter=$(echo "$stem" | awk '{print toupper($1)}')
        
        out_dir="$person_temp/$letter"
        mkdir -p "$out_dir"
        
        # Extract frames at 6fps
        new_frames=$(ffmpeg -i "$video" -vf "fps=$FPS" -q:v 2 "$out_dir/%04d.jpg" 2>/dev/null)
        
        count=$(ls "$out_dir"/*.jpg 2>/dev/null | wc -l)
        echo "  $video -> $letter/: $count frames"
        extracted=$((extracted + count))
    done
    
    echo "  $person done: $extracted frames"
    total=$((total + extracted))
done

echo ""
echo "=== Merging into Kaggle-style directory ==="
# Merge all person directories into the shared letter directories
for letter_dir in "$TEMP"/*/; do
    [ -d "$letter_dir" ] || continue
    for letter in A B C D E F G H I J K L M N O P Q R S T U V W X Y Z; do
        src="$letter_dir/$letter"
        dst="$DST/$letter"
        mkdir -p "$dst"
        
        if [ -d "$src" ]; then
            for f in "$src"/*.jpg; do
                [ -f "$f" ] || continue
                basename_f=$(basename "$f")
                # Check if already exists (shouldn't, but safety)
                if [ ! -f "$dst/$basename_f" ]; then
                    cp "$f" "$dst/"
                    total_skipped=$((total_skipped + 1))
                fi
            done
        fi
    done
done

echo "Done: $total frames extracted, $total_skipped merged"
echo "Output: $(find $DST -name '*.jpg' | wc -l) total frames in $DST"

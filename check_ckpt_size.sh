#!/bin/bash

BASE="/home/mvu9/folder_04_ma/latent_visual"
FOLDERS=("checkpoints_dimv" "checkpoints_rerun")

echo "=== Checkpoint Folder Sizes ==="
total_bytes=0

for folder in "${FOLDERS[@]}"; do
    path="$BASE/$folder"
    if [ -d "$path" ]; then
        size=$(du -sb "$path" | awk '{print $1}')
        size_gb=$(echo "scale=2; $size / 1073741824" | bc)
        echo "$folder: ${size_gb} GB"
        total_bytes=$((total_bytes + size))
    else
        echo "$folder: NOT FOUND"
    fi
done

total_gb=$(echo "scale=2; $total_bytes / 1073741824" | bc)
echo "------------------------------"
echo "Total: ${total_gb} GB"

echo ""
echo "=== Per-subdirectory breakdown ==="
for folder in "${FOLDERS[@]}"; do
    path="$BASE/$folder"
    if [ -d "$path" ]; then
        echo ""
        echo "[$folder]"
        du -sh "$path"/*/  2>/dev/null | sort -rh | head -20
    fi
done

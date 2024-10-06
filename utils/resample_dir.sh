#!/bin/bash

# Directory containing the original WAV files
INPUT_DIR="origin_data"

# Directory to save the resampled WAV files
OUTPUT_DIR="origin_data_16k"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each WAV file in the input directory
for file in "$INPUT_DIR"/*.wav; do
    # Extract filename without extension
    filename=$(basename "$file" .wav)

    # Resample and save with the same name in the output directory
    sox "$file" -r 16000 "$OUTPUT_DIR/$filename.wav"
done
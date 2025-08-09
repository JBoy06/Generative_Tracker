#!/bin/bash
# Diagnostic version of GPU-Optimized Parallel Runner
# Shows what's actually happening with better error reporting

# --- Configuration ---
if [ -z "$1" ]; then
    echo "Usage: $0 input_video.mp4 [output_video.mp4] [--cores C] [--end-frame N] [--gpu-mode MODE]"
    echo "GPU modes: auto, single, multi, cpu-only"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-final_output.mp4}"
CORES=""
END_FRAME=""
GPU_MODE="auto"
DEBUG_MODE=1  # Enable debugging

# Parse remaining arguments
shift 2
while [[ $# -gt 0 ]]; do
    case $1 in
        --cores) CORES="$2"; shift 2 ;;
        --end-frame) END_FRAME="$2"; shift 2 ;;
        --gpu-mode) GPU_MODE="$2"; shift 2 ;;
        --debug) DEBUG_MODE=1; shift ;;
        *) shift ;;
    esac
done

echo "=== DIAGNOSTIC MODE ENABLED ==="
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Debug: $DEBUG_MODE"

# --- Test generative_tracker.py first ---
echo ""
echo "=== Testing generative_tracker.py ==="
if [ ! -f "generative_tracker.py" ]; then
    echo "❌ ERROR: generative_tracker.py not found in current directory"
    exit 1
fi

echo "✅ generative_tracker.py found"

# Test if Python can import required modules
echo "Testing Python dependencies..."
python3 -c "
import sys
try:
    import cv2
    print('✅ OpenCV imported successfully')
    print(f'   OpenCV version: {cv2.__version__}')
    
    # Test OpenCL
    print(f'   OpenCL available: {cv2.ocl.haveOpenCL()}')
    if cv2.ocl.haveOpenCL():
        print(f'   OpenCL enabled: {cv2.ocl.useOpenCL()}')
    
    import numpy as np
    print('✅ NumPy imported successfully')
    
    import argparse, math, time, logging
    from collections import deque
    from typing import List, Tuple, Optional, Deque
    from tqdm import tqdm
    print('✅ All required modules available')
    
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
" || {
    echo "❌ Python dependency check failed"
    echo "Install missing packages with:"
    echo "pip install opencv-python numpy tqdm"
    exit 1
}

# Test basic functionality with a very short segment
echo ""
echo "=== Testing Short Video Segment ==="
echo "Testing generative_tracker.py with first 10 frames..."

# Create test output directory
mkdir -p test_output

# Run a quick test
timeout 30 python3 generative_tracker.py \
    --input "$INPUT" \
    --output "test_output/test_chunk.mp4" \
    --start-frame 0 \
    --end-frame 10 \
    --verbose 2>&1 | tee test_output/test_log.txt

TEST_EXIT_CODE=$?
echo "Test exit code: $TEST_EXIT_CODE"

if [ $TEST_EXIT_CODE -eq 124 ]; then
    echo "⚠️  Test timed out after 30 seconds"
    echo "This suggests the generative_tracker.py is very slow or hanging"
elif [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "❌ Test failed with exit code $TEST_EXIT_CODE"
    echo "Last few lines of output:"
    tail -10 test_output/test_log.txt 2>/dev/null || echo "No log output"
else
    echo "✅ Basic test completed successfully"
    if [ -f "test_output/test_chunk.mp4" ]; then
        echo "✅ Test output file created"
        ls -la test_output/test_chunk.mp4
    else
        echo "❌ No test output file created"
    fi
fi

# GPU Detection (simplified)
echo ""
echo "=== GPU Detection ==="
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
    GPU_TYPE="nvidia"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
else
    echo "No NVIDIA GPU detected"
    GPU_TYPE="cpu"
    GPU_COUNT=0
fi

# System info
echo ""
echo "=== System Information ==="
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Disk space: $(df -h . | awk 'NR==2 {print $4}')"

# Video info
echo ""
echo "=== Video Information ==="
if command -v ffprobe &> /dev/null; then
    echo "Input video details:"
    ffprobe -v quiet -show_entries format=duration,size -show_entries stream=width,height,nb_frames,r_frame_rate -of default=nokey=1:noprint_wrappers=1 "$INPUT" 2>/dev/null || echo "Could not get video info"
else
    echo "ffprobe not available"
fi

# Now ask user if they want to continue
echo ""
echo "=== Analysis Complete ==="
echo "Based on the tests above, do you want to:"
echo "1. Continue with full processing (may be slow)"
echo "2. Try CPU-only mode"
echo "3. Process a smaller segment first"
echo "4. Exit and fix issues"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Continuing with full processing..."
        RUN_MODE="full"
        ;;
    2)
        echo "Switching to CPU-only mode..."
        GPU_MODE="cpu-only"
        RUN_MODE="full"
        ;;
    3)
        echo "Processing smaller segment (first 50 frames)..."
        END_FRAME=50
        RUN_MODE="small"
        ;;
    4)
        echo "Exiting for troubleshooting..."
        exit 0
        ;;
    *)
        echo "Invalid choice, exiting..."
        exit 1
        ;;
esac

# Simplified processing function
process_chunk_debug() {
    local chunk_id=$1
    local start_frame=$2
    local end_frame=$3
    local chunk_file="$TMP_DIR/chunk_${chunk_id}.mp4"
    
    echo "Processing chunk $chunk_id: frames $start_frame-$end_frame"
    
    # Run with visible output for debugging
    if python3 generative_tracker.py \
        --input "$INPUT" \
        --output "$chunk_file" \
        --start-frame $start_frame \
        --end-frame $end_frame \
        --verbose; then
        echo "✅ Chunk $chunk_id completed successfully"
        if [ -f "$chunk_file" ]; then
            local file_size=$(du -h "$chunk_file" | cut -f1)
            echo "   Output: $chunk_file ($file_size)"
            return 0
        else
            echo "❌ Chunk $chunk_id: Output file not created"
            return 1
        fi
    else
        local exit_code=$?
        echo "❌ Chunk $chunk_id failed with exit code $exit_code"
        return $exit_code
    fi
}

# Setup directories
TMP_DIR="chunks"
mkdir -p "$TMP_DIR"
rm -rf "$TMP_DIR"/* 2>/dev/null || true

# Get total frames (simplified)
if command -v ffprobe &> /dev/null; then
    TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$INPUT" 2>/dev/null)
    if [ -z "$TOTAL_FRAMES" ] || [ "$TOTAL_FRAMES" -eq 0 ]; then
        # Fallback estimation
        duration=$(ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$INPUT" 2>/dev/null)
        fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nokey=1:noprint_wrappers=1 "$INPUT" 2>/dev/null)
        if [ -n "$duration" ] && [ -n "$fps" ]; then
            TOTAL_FRAMES=$(echo "scale=0; $duration * ($fps)" | bc 2>/dev/null || echo "300")
        else
            TOTAL_FRAMES=300  # Default fallback
        fi
    fi
else
    TOTAL_FRAMES=300  # Default fallback
fi

if [ -n "$END_FRAME" ] && [ "$END_FRAME" -lt "$TOTAL_FRAMES" ]; then
    TOTAL_FRAMES="$END_FRAME"
fi

echo ""
echo "=== Starting Processing ==="
echo "Total frames to process: $TOTAL_FRAMES"
echo "GPU mode: $GPU_MODE"

# Process in small chunks for better monitoring
if [ "$TOTAL_FRAMES" -le 100 ]; then
    NUM_CHUNKS=2
    CHUNK_SIZE=$((TOTAL_FRAMES / 2))
else
    NUM_CHUNKS=4
    CHUNK_SIZE=$((TOTAL_FRAMES / 4))
fi

if [ "$CHUNK_SIZE" -lt 1 ]; then CHUNK_SIZE=1; fi

echo "Processing in $NUM_CHUNKS chunks of ~$CHUNK_SIZE frames each"
echo ""

# Process chunks sequentially with detailed output
SUCCESSFUL_CHUNKS=""
for chunk_id in $(seq 0 $((NUM_CHUNKS - 1))); do
    start_frame=$((chunk_id * CHUNK_SIZE))
    if [ "$chunk_id" -eq $((NUM_CHUNKS - 1)) ]; then
        end_frame=$TOTAL_FRAMES
    else
        end_frame=$(((chunk_id + 1) * CHUNK_SIZE))
    fi
    
    echo "=== Chunk $((chunk_id + 1))/$NUM_CHUNKS ==="
    
    if process_chunk_debug $chunk_id $start_frame $end_frame; then
        SUCCESSFUL_CHUNKS="$SUCCESSFUL_CHUNKS $chunk_id"
    else
        echo "⚠️  Chunk $chunk_id failed, continuing with remaining chunks..."
    fi
    echo ""
done

# Check results
if [ -z "$SUCCESSFUL_CHUNKS" ]; then
    echo "❌ No chunks were processed successfully"
    echo "Available files in chunks directory:"
    ls -la "$TMP_DIR"/ 2>/dev/null || echo "No files found"
    exit 1
fi

echo "=== Merging Results ==="
echo "Successfully processed chunks:$SUCCESSFUL_CHUNKS"

# Simple merging
MERGE_LIST="$TMP_DIR/merge_list.txt"
> "$MERGE_LIST"

VALID_CHUNKS=0
for chunk_id in $SUCCESSFUL_CHUNKS; do
    chunk_file="$TMP_DIR/chunk_${chunk_id}.mp4"
    if [ -f "$chunk_file" ] && [ -s "$chunk_file" ]; then
        echo "file '$PWD/$chunk_file'" >> "$MERGE_LIST"
        VALID_CHUNKS=$((VALID_CHUNKS + 1))
        echo "✅ Added chunk $chunk_id to merge list"
    else
        echo "⚠️  Chunk $chunk_id file is missing or empty"
    fi
done

if [ "$VALID_CHUNKS" -eq 0 ]; then
    echo "❌ No valid chunks found for merging"
    exit 1
fi

echo "Merging $VALID_CHUNKS chunks..."

# Merge without audio first
if ffmpeg -y -f concat -safe 0 -i "$MERGE_LIST" -c copy "$TMP_DIR/merged_no_audio.mp4" 2>&1; then
    echo "✅ Video chunks merged successfully"
    
    # Add audio
    echo "Adding audio..."
    if ffmpeg -y -i "$TMP_DIR/merged_no_audio.mp4" -i "$INPUT" \
        -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? \
        -shortest "$OUTPUT" 2>&1; then
        echo "✅ Final output created: $OUTPUT"
        
        # Show final stats
        if [ -f "$OUTPUT" ]; then
            final_size=$(du -h "$OUTPUT" | cut -f1)
            echo "📊 Final file size: $final_size"
            
            if command -v ffprobe &> /dev/null; then
                final_duration=$(ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$OUTPUT" 2>/dev/null)
                if [ -n "$final_duration" ]; then
                    echo "📊 Final duration: ${final_duration}s"
                fi
            fi
        fi
    else
        echo "⚠️  Audio merge failed, but video-only output available: $TMP_DIR/merged_no_audio.mp4"
        cp "$TMP_DIR/merged_no_audio.mp4" "$OUTPUT"
    fi
else
    echo "❌ Video merge failed"
    exit 1
fi

# Cleanup
echo ""
read -p "Remove temporary files? (y/n): " cleanup_choice
if [ "$cleanup_choice" = "y" ]; then
    rm -rf "$TMP_DIR"
    rm -rf test_output
    echo "✅ Cleanup completed"
fi

echo ""
echo "🎉 Processing completed!"
echo "Output: $OUTPUT"
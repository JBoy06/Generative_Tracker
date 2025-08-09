#!/bin/bash
# Parallel runner using GNU Parallel for optimal dynamic scheduling

# --- Configuration ---
if [ -z "$1" ]; then
    echo "Usage: $0 input_video.mp4 [output_video.mp4] [--cores C] [--end-frame N]"
    exit 1
fi

INPUT="$1"
OUTPUT=""
DEFAULT_CORES=3
CORES="$DEFAULT_CORES"
END_FRAME=""

# Parse arguments properly
shift # Remove first argument (input file)
while [[ $# -gt 0 ]]; do
    case $1 in
        --cores)
            CORES="$2"
            shift 2
            ;;
        --end-frame)
            END_FRAME="$2"
            shift 2
            ;;
        *)
            if [ -z "$OUTPUT" ]; then
                OUTPUT="$1"
            fi
            shift
            ;;
    esac
done

# Set default output if not specified
if [ -z "$OUTPUT" ]; then
    OUTPUT="final_output.mp4"
fi

# --- Setup ---
TOTAL_SYS_CORES=$(nproc)
if [ "$CORES" -gt "$TOTAL_SYS_CORES" ]; then
    CORES=$TOTAL_SYS_CORES
fi

# Check dependencies
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU Parallel is not installed"
    exit 1
fi

if ! command -v ffprobe &> /dev/null || ! command -v ffmpeg &> /dev/null; then
    echo "Error: FFmpeg is not installed"
    exit 1
fi

if ! command -v bc &> /dev/null; then
    echo "Error: bc (calculator) is not installed. Please install: sudo apt-get install bc"
    exit 1
fi

# DYNAMIC QUEUE LOGIC: Create more chunks than cores
NUM_CHUNKS=$((CORES * 4))
TMP_DIR="chunks"
mkdir -p "$TMP_DIR"
# Clean directory but preserve structure
find "$TMP_DIR" -name "*.mp4" -delete 2>/dev/null || true
find "$TMP_DIR" -name "*.txt" -delete 2>/dev/null || true

# --- Get total frames ---
TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames \
               -show_entries stream=nb_read_frames \
               -of default=nokey=1:noprint_wrappers=1 "$INPUT")

if [ -z "$TOTAL_FRAMES" ] || [ "$TOTAL_FRAMES" -eq 0 ]; then
    echo "Error: Could not determine total frames from input video"
    exit 1
fi

if [ -n "$END_FRAME" ] && [ "$END_FRAME" -lt "$TOTAL_FRAMES" ]; then
    TOTAL_FRAMES="$END_FRAME"
fi

CHUNK_SIZE=$((TOTAL_FRAMES / NUM_CHUNKS))
if [ "$CHUNK_SIZE" -eq 0 ]; then CHUNK_SIZE=1; fi

echo "----------------------------------------"
echo "Total frames: $TOTAL_FRAMES, Target Cores: $CORES"
echo "Dynamic Queue: Creating $NUM_CHUNKS small chunks..."
echo "----------------------------------------"

# --- Build and run jobs with GNU Parallel with Custom Progress ---
# Export variables so they're available in parallel subshells
export INPUT TOTAL_FRAMES NUM_CHUNKS CHUNK_SIZE TMP_DIR PROGRESS_DIR

# Create progress tracking files
PROGRESS_DIR="$TMP_DIR/progress"
mkdir -p "$PROGRESS_DIR"
echo "0" > "$PROGRESS_DIR/completed_frames"
echo "$(date +%s)" > "$PROGRESS_DIR/start_time"
echo "0" > "$PROGRESS_DIR/monitor_active"
# Clear the chunk completion log
> "$TMP_DIR/progress/completed_chunks.txt"

# Create a function to process each chunk
process_chunk() {
    local chunk_id=$1
    local START=$((chunk_id * CHUNK_SIZE))
    local END
    local FRAMES_IN_CHUNK
    
    # Calculate END frame and chunk size
    if [ "$chunk_id" -eq $((NUM_CHUNKS - 1)) ]; then
        END=$TOTAL_FRAMES
        FRAMES_IN_CHUNK=$((TOTAL_FRAMES - START))
    else
        END=$(((chunk_id + 1) * CHUNK_SIZE))
        FRAMES_IN_CHUNK=$CHUNK_SIZE
    fi
    
    # Run the python script for the chunk (suppress individual progress)
    python3 generative_tracker.py \
        --input "$INPUT" \
        --output "$TMP_DIR/chunk_${chunk_id}.mp4" \
        --start-frame $START \
        --end-frame $END > /dev/null 2>&1
    
    # Update progress if successful
    if [ $? -eq 0 ]; then
        # Simple progress update - write chunk completion to a file
        echo "$chunk_id $FRAMES_IN_CHUNK" >> "$TMP_DIR/progress/completed_chunks.txt"
        echo "$chunk_id"
    else
        echo "Error processing chunk $chunk_id (frames $START-$END)" >&2
        return 1
    fi
}

# Progress monitoring function (runs only once)
monitor_progress() {
    local start_time=$(cat "$PROGRESS_DIR/start_time")
    
    # Mark monitor as active
    echo "1" > "$PROGRESS_DIR/monitor_active"
    
    while true; do
        # Calculate completed frames by reading chunk completions
        local completed_frames=0
        if [ -f "$TMP_DIR/progress/completed_chunks.txt" ]; then
            while read -r chunk_id frames_in_chunk; do
                if [ -n "$chunk_id" ] && [ -n "$frames_in_chunk" ]; then
                    completed_frames=$((completed_frames + frames_in_chunk))
                fi
            done < "$TMP_DIR/progress/completed_chunks.txt"
        fi
        
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ "$completed_frames" -ge "$TOTAL_FRAMES" ]; then
            break
        fi
        
        # Calculate progress percentage
        local progress=0
        if [ "$TOTAL_FRAMES" -gt 0 ]; then
            progress=$((completed_frames * 100 / TOTAL_FRAMES))
        fi
        
        # Calculate frames per second
        local fps="0.00"
        if [ "$elapsed" -gt 0 ] && [ "$completed_frames" -gt 0 ]; then
            fps=$(echo "scale=2; $completed_frames / $elapsed" | bc -l 2>/dev/null || echo "0.00")
        fi
        
        # Calculate ETA
        local eta_seconds=0
        local eta_display="--:--"
        if [ "$completed_frames" -gt 0 ] && [ "$elapsed" -gt 0 ] && [ "$fps" != "0.00" ]; then
            local remaining_frames=$((TOTAL_FRAMES - completed_frames))
            eta_seconds=$(echo "scale=0; $remaining_frames / $fps" | bc -l 2>/dev/null || echo "0")
            if [ "$eta_seconds" -gt 0 ] 2>/dev/null; then
                local eta_min=$((eta_seconds / 60))
                local eta_sec=$((eta_seconds % 60))
                eta_display=$(printf "%02d:%02d" $eta_min $eta_sec)
            fi
        fi
        
        # Format elapsed time
        local elapsed_min=$((elapsed / 60))
        local elapsed_sec=$((elapsed % 60))
        local elapsed_display=$(printf "%02d:%02d" $elapsed_min $elapsed_sec)
        
        # Create progress bar
        local bar_width=20
        local filled_width=$((progress * bar_width / 100))
        local bar=""
        
        # Build the progress bar
        for ((i=0; i<filled_width; i++)); do
            bar+="█"
        done
        
        # Add partial block if needed
        local remainder=$((progress * bar_width % 100))
        if [ "$filled_width" -lt "$bar_width" ] && [ "$remainder" -gt 0 ]; then
            if [ "$remainder" -gt 75 ]; then
                bar+="▉"
            elif [ "$remainder" -gt 50 ]; then
                bar+="▌"
            elif [ "$remainder" -gt 25 ]; then
                bar+="▎"
            else
                bar+="▏"
            fi
            filled_width=$((filled_width + 1))
        fi
        
        # Fill the rest with spaces
        for ((i=filled_width; i<bar_width; i++)); do
            bar+=" "
        done
        
        # Print progress line (overwrite previous line)
        printf "\rProcessing: %3d%%|%s| %d/%d [%s<%s, %s frame/s]" \
            "$progress" "$bar" "$completed_frames" "$TOTAL_FRAMES" \
            "$elapsed_display" "$eta_display" "$fps"
        
        sleep 0.5
    done
    
    # Final progress line
    local final_elapsed=$(($(date +%s) - start_time))
    local final_elapsed_min=$((final_elapsed / 60))
    local final_elapsed_sec=$((final_elapsed % 60))
    local final_elapsed_display=$(printf "%02d:%02d" $final_elapsed_min $final_elapsed_sec)
    local final_fps="0.00"
    if [ "$final_elapsed" -gt 0 ]; then
        final_fps=$(echo "scale=2; $TOTAL_FRAMES / $final_elapsed" | bc -l 2>/dev/null || echo "0.00")
    fi
    
    printf "\rProcessing: 100%%|████████████████████| %d/%d [%s<00:00, %s frame/s]\n" \
        "$TOTAL_FRAMES" "$TOTAL_FRAMES" "$final_elapsed_display" "$final_fps"
    
    # Mark monitor as inactive
    echo "0" > "$PROGRESS_DIR/monitor_active"
}

# Export the function
export -f process_chunk

# Start progress monitor in background (only if not already active)
if [ "$(cat "$PROGRESS_DIR/monitor_active" 2>/dev/null || echo "0")" -eq 0 ]; then
    monitor_progress &
    MONITOR_PID=$!
fi

# Run parallel processing (suppress individual output to avoid multiple progress bars)
echo "Processing chunks in parallel..."
SUCCESSFUL_CHUNKS=$(seq 0 $((NUM_CHUNKS-1)) | parallel --jobs $CORES --no-notice process_chunk)

# Stop progress monitor
if [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null
    wait $MONITOR_PID 2>/dev/null
fi

if [ -z "$SUCCESSFUL_CHUNKS" ]; then
    echo "Error: No chunks were processed successfully"
    echo "Checking for chunk files..."
    for ((i=0; i<NUM_CHUNKS; i++)); do
        if [ -f "$TMP_DIR/chunk_$i.mp4" ]; then
            echo "Found: $TMP_DIR/chunk_$i.mp4"
            SUCCESSFUL_CHUNKS="$SUCCESSFUL_CHUNKS $i"
        fi
    done
    
    if [ -z "$SUCCESSFUL_CHUNKS" ]; then
        echo "No valid chunk files found. Check your generative_tracker.py script."
        exit 1
    else
        echo "Found some chunk files, continuing with merge..."
    fi
fi

echo "All chunks processed."

# --- Merge and Add Audio ---
MERGE_LIST="$TMP_DIR/merge_list.txt"
> "$MERGE_LIST"

# Sort successful chunks numerically and include only valid ones
echo "Checking chunks for merging..."
VALID_CHUNKS=0
for chunk_id in $(echo "$SUCCESSFUL_CHUNKS" | tr ' ' '\n' | sort -n); do
    CHUNK_FILE="$TMP_DIR/chunk_${chunk_id}.mp4"
    if [ -f "$CHUNK_FILE" ]; then
        # Verify chunk has actual video content
        CHUNK_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$CHUNK_FILE" 2>/dev/null || echo "0")
        if [ "$CHUNK_FRAMES" -gt 0 ]; then
            echo "file '$PWD/$CHUNK_FILE'" >> "$MERGE_LIST"
            echo "  Chunk $chunk_id: $CHUNK_FRAMES frames"
            VALID_CHUNKS=$((VALID_CHUNKS + 1))
        else
            echo "  Chunk $chunk_id: EMPTY (skipping)"
        fi
    else
        echo "  Chunk $chunk_id: MISSING"
    fi
done

echo "Found $VALID_CHUNKS valid chunks for merging"

# Check if we have any chunks to merge
if [ ! -s "$MERGE_LIST" ] || [ "$VALID_CHUNKS" -eq 0 ]; then
    echo "Error: No valid chunks found for merging"
    echo "Debug: Listing all files in chunks directory:"
    ls -la "$TMP_DIR"/
    exit 1
fi

echo "Merging chunks..."
if ! ffmpeg -y -f concat -safe 0 -i "$MERGE_LIST" -c copy "$TMP_DIR/merged.mp4"; then
    echo "Error: Failed to merge chunks"
    exit 1
fi

# Get the duration of the merged video to match audio to it
MERGED_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$TMP_DIR/merged.mp4")

echo "Adding audio..."
# FIXED: Truncate audio to match video duration and ensure video timing is preserved
if ! ffmpeg -y -i "$TMP_DIR/merged.mp4" -i "$INPUT" \
    -c:v copy -c:a aac \
    -map 0:v:0 -map 1:a:0 \
    -t "$MERGED_DURATION" \
    -avoid_negative_ts make_zero \
    "$OUTPUT"; then
    echo "Error: Failed to add audio"
    exit 1
fi

echo "Done! Final video saved as $OUTPUT"

# Verify the final output
echo "Final video verification:"
FINAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$OUTPUT" 2>/dev/null || echo "0")
FINAL_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT" 2>/dev/null || echo "0")
FINAL_SIZE=$(du -h "$OUTPUT" 2>/dev/null | cut -f1)

echo "  Frames: $FINAL_FRAMES"
echo "  Duration: ${FINAL_DURATION}s"
echo "  File size: $FINAL_SIZE"

if [ "$FINAL_FRAMES" -eq 0 ]; then
    echo "WARNING: Final video appears to have no frames!"
    echo "Check if the processing step is working correctly."
else
    echo "SUCCESS: Video processing completed successfully!"
fi

# Uncomment the next line to clean up temporary files
rm -rf "$TMP_DIR"

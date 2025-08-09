#!/bin/bash
# GPU-Optimized Parallel Runner for Generative Tracker
# Automatically detects GPU capabilities and optimizes chunk processing

# --- Configuration ---
if [ -z "$1" ]; then
    echo "Usage: $0 input_video.mp4 [output_video.mp4] [--cores C] [--end-frame N] [--gpu-mode MODE]"
    echo "GPU modes: auto, single, multi, cpu-only"
    exit 1
fi

INPUT="$1"
OUTPUT=""
CORES=""
END_FRAME=""
GPU_MODE="auto"

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
        --gpu-mode)
            GPU_MODE="$2"
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

# --- GPU Detection and Configuration ---
detect_gpu_setup() {
    local gpu_count=0
    local gpu_memory=0
    local gpu_type="none"
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            gpu_type="nvidia"
            # Get memory of first GPU (in MB)
            gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
        fi
    fi
    
    # Check for AMD GPU (ROCm)
    if [ "$gpu_type" = "none" ] && command -v rocm-smi &> /dev/null; then
        gpu_count=$(rocm-smi --showid 2>/dev/null | grep -c "GPU\[")
        if [ "$gpu_count" -gt 0 ]; then
            gpu_type="amd"
            # ROCm memory detection (simplified)
            gpu_memory=8192  # Default assumption for modern AMD GPUs
        fi
    fi
    
    # Check OpenCL support
    local opencl_support=false
    if command -v clinfo &> /dev/null; then
        if clinfo 2>/dev/null | grep -q "Platform Name"; then
            opencl_support=true
        fi
    fi
    
    echo "$gpu_count,$gpu_memory,$gpu_type,$opencl_support"
}

# Detect GPU configuration
GPU_INFO=$(detect_gpu_setup)
IFS=',' read -r GPU_COUNT GPU_MEMORY GPU_TYPE OPENCL_SUPPORT <<< "$GPU_INFO"

echo "=== GPU Detection Results ==="
echo "GPU Type: $GPU_TYPE"
echo "GPU Count: $GPU_COUNT"
echo "GPU Memory: ${GPU_MEMORY}MB (per GPU)"
echo "OpenCL Support: $OPENCL_SUPPORT"
echo "============================="

# --- Optimize Configuration Based on GPU ---
optimize_for_gpu() {
    local optimal_cores=1
    local optimal_chunks=1
    local gpu_message=""
    
    case "$GPU_MODE" in
        "cpu-only")
            optimal_cores=$(nproc)
            optimal_chunks=$((optimal_cores * 3))
            gpu_message="CPU-only mode: Using all CPU cores"
            ;;
        "single"|"auto")
            if [ "$GPU_COUNT" -gt 0 ]; then
                # Single GPU optimization
                if [ "$GPU_MEMORY" -gt 16000 ]; then
                    # High-end GPU: Allow more parallel processes
                    optimal_cores=3
                    optimal_chunks=$((optimal_cores * 2))
                    gpu_message="Single GPU mode: High-end GPU detected"
                elif [ "$GPU_MEMORY" -gt 8000 ]; then
                    # Mid-range GPU: Moderate parallelism
                    optimal_cores=2  
                    optimal_chunks=$((optimal_cores * 2))
                    gpu_message="Single GPU mode: Mid-range GPU detected"
                else
                    # Lower-end GPU: Sequential processing for stability
                    optimal_cores=1
                    optimal_chunks=4
                    gpu_message="Single GPU mode: Lower-end GPU detected"
                fi
            else
                # No GPU detected, fall back to CPU
                optimal_cores=$(nproc)
                optimal_chunks=$((optimal_cores * 2))
                gpu_message="No GPU detected: Falling back to CPU processing"
            fi
            ;;
        "multi")
            if [ "$GPU_COUNT" -gt 1 ]; then
                # Multi-GPU: One process per GPU
                optimal_cores=$GPU_COUNT
                optimal_chunks=$((optimal_cores * 2))
                gpu_message="Multi-GPU mode: Using $GPU_COUNT GPUs"
                export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
            else
                gpu_message="Multi-GPU requested but only $GPU_COUNT GPU(s) available"
                optimal_cores=1
                optimal_chunks=4
            fi
            ;;
        *)
            gpu_message="Unknown GPU mode: $GPU_MODE"
            optimal_cores=2
            optimal_chunks=8
            ;;
    esac
    
    echo "$gpu_message"
    echo "$optimal_cores,$optimal_chunks"
}

# Get optimal configuration
OPTIMAL_OUTPUT=$(optimize_for_gpu)
# Split the output - first line is message, second line is values
GPU_MESSAGE=$(echo "$OPTIMAL_OUTPUT" | head -1)
OPTIMAL_CONFIG=$(echo "$OPTIMAL_OUTPUT" | tail -1)

echo "$GPU_MESSAGE"
IFS=',' read -r OPTIMAL_CORES OPTIMAL_CHUNKS <<< "$OPTIMAL_CONFIG"

# Use user-specified cores if provided, otherwise use optimal
if [ -n "$CORES" ]; then
    echo "Using user-specified cores: $CORES"
else
    CORES=$OPTIMAL_CORES
    echo "Using optimized core count: $CORES"
fi

NUM_CHUNKS=$OPTIMAL_CHUNKS

# --- System Checks ---
TOTAL_SYS_CORES=$(nproc)
if [ "$CORES" -gt "$TOTAL_SYS_CORES" ]; then
    CORES=$TOTAL_SYS_CORES
    echo "Reduced cores to system maximum: $CORES"
fi

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    if ! command -v parallel &> /dev/null; then
        missing_deps+=("parallel")
    fi
    
    if ! command -v ffprobe &> /dev/null || ! command -v ffmpeg &> /dev/null; then
        missing_deps+=("ffmpeg")
    fi
    
    if ! command -v bc &> /dev/null; then
        missing_deps+=("bc")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "Error: Missing dependencies: ${missing_deps[*]}"
        echo "Install with:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install ${missing_deps[*]}"
        echo ""
        echo "Note: For parallel, you might need 'parallel' or 'moreutils' package"
        exit 1
    fi
}

check_dependencies

# --- Setup Directories ---
TMP_DIR="chunks"
mkdir -p "$TMP_DIR"
find "$TMP_DIR" -name "*.mp4" -delete 2>/dev/null || true
find "$TMP_DIR" -name "*.txt" -delete 2>/dev/null || true

# --- Get Video Information ---
get_video_info() {
    echo "Analyzing input video..."
    
    # Get total frames
    TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames \
                   -show_entries stream=nb_read_frames \
                   -of default=nokey=1:noprint_wrappers=1 "$INPUT" 2>/dev/null)
    
    if [ -z "$TOTAL_FRAMES" ] || [ "$TOTAL_FRAMES" -eq 0 ]; then
        # Fallback method using duration and fps
        local duration=$(ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$INPUT" 2>/dev/null)
        local fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nokey=1:noprint_wrappers=1 "$INPUT" 2>/dev/null)
        
        if [ -n "$duration" ] && [ -n "$fps" ]; then
            # Convert fps fraction to decimal
            TOTAL_FRAMES=$(echo "scale=0; $duration * ($fps)" | bc -l 2>/dev/null | cut -d. -f1)
        fi
        
        if [ -z "$TOTAL_FRAMES" ] || [ "$TOTAL_FRAMES" -eq 0 ]; then
            echo "Error: Could not determine total frames from input video"
            exit 1
        fi
    fi
    
    # Get video properties for optimization
    local width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of default=nokey=1:noprint_wrappers=1 "$INPUT" 2>/dev/null)
    local height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nokey=1:noprint_wrappers=1 "$INPUT" 2>/dev/null)
    local fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nokey=1:noprint_wrappers=1 "$INPUT" 2>/dev/null)
    
    echo "Video: ${width}x${height}, ${fps} fps, ${TOTAL_FRAMES} frames"
    
    # Adjust chunk size based on video resolution for GPU memory management
    if [ "$GPU_COUNT" -gt 0 ] && [ -n "$width" ] && [ -n "$height" ]; then
        local pixels=$((width * height))
        if [ "$pixels" -gt 8000000 ]; then  # 4K+ content
            echo "4K+ content detected: Using smaller chunks for memory efficiency"
            NUM_CHUNKS=$((NUM_CHUNKS * 2))
        elif [ "$pixels" -gt 2000000 ]; then  # 1080p content
            echo "1080p content detected: Using standard chunk size"
        else
            echo "Lower resolution content: Can use larger chunks"
            NUM_CHUNKS=$((NUM_CHUNKS / 2))
            if [ "$NUM_CHUNKS" -lt 2 ]; then NUM_CHUNKS=2; fi
        fi
    fi
}

get_video_info

if [ -n "$END_FRAME" ] && [ "$END_FRAME" -lt "$TOTAL_FRAMES" ]; then
    TOTAL_FRAMES="$END_FRAME"
    echo "Limited processing to frame $END_FRAME"
fi

CHUNK_SIZE=$((TOTAL_FRAMES / NUM_CHUNKS))
if [ "$CHUNK_SIZE" -eq 0 ]; then CHUNK_SIZE=1; fi

echo "========================================="
echo "Processing Configuration:"
echo "  Total frames: $TOTAL_FRAMES"
echo "  Parallel processes: $CORES" 
echo "  Chunks: $NUM_CHUNKS (${CHUNK_SIZE} frames each)"
echo "  GPU Mode: $GPU_MODE"
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "  GPU Utilization: ${GPU_TYPE^^} (${GPU_COUNT}x ${GPU_MEMORY}MB)"
fi
echo "========================================="

# --- GPU Environment Setup ---
setup_gpu_environment() {
    # Set OpenCV GPU flags
    export OPENCV_DNN_OPENCL=1
    export OPENCV_OPENCL_RUNTIME=""
    
    if [ "$GPU_TYPE" = "nvidia" ]; then
        # NVIDIA-specific optimizations
        export CUDA_CACHE_DISABLE=0
        export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
        
        # Enable GPU memory management
        export OPENCV_GPU_DEVICE_ID=0
        
        # Multi-GPU setup
        if [ "$GPU_MODE" = "multi" ] && [ "$GPU_COUNT" -gt 1 ]; then
            echo "Setting up multi-GPU environment for $GPU_COUNT GPUs"
        fi
    elif [ "$GPU_TYPE" = "amd" ]; then
        # AMD ROCm optimizations
        export HSA_OVERRIDE_GFX_VERSION="10.3.0"  # Common compatibility setting
        export ROC_ENABLE_PRE_VEGA=1
    fi
    
    # Memory management for large videos
    if [ "$GPU_MEMORY" -lt 6000 ]; then
        echo "Limited GPU memory detected: Enabling conservative memory usage"
        export OPENCV_GPU_MEM_POOL_SIZE=$((GPU_MEMORY / 4))
    fi
}

if [ "$GPU_COUNT" -gt 0 ] && [ "$GPU_MODE" != "cpu-only" ]; then
    setup_gpu_environment
fi

# Export variables for parallel processes
export INPUT TOTAL_FRAMES NUM_CHUNKS CHUNK_SIZE TMP_DIR GPU_MODE GPU_COUNT

# Create progress tracking
PROGRESS_DIR="$TMP_DIR/progress"
mkdir -p "$PROGRESS_DIR"
echo "0" > "$PROGRESS_DIR/completed_frames"
echo "$(date +%s)" > "$PROGRESS_DIR/start_time"
echo "0" > "$PROGRESS_DIR/monitor_active"
> "$TMP_DIR/progress/completed_chunks.txt"

# --- GPU-Aware Chunk Processing ---
process_chunk() {
    local chunk_id=$1
    local START=$((chunk_id * CHUNK_SIZE))
    local END
    local FRAMES_IN_CHUNK
    
    # Multi-GPU: Assign GPU to process
    if [ "$GPU_MODE" = "multi" ] && [ "$GPU_COUNT" -gt 1 ]; then
        local gpu_id=$((chunk_id % GPU_COUNT))
        export CUDA_VISIBLE_DEVICES=$gpu_id
    fi
    
    # Calculate END frame and chunk size
    if [ "$chunk_id" -eq $((NUM_CHUNKS - 1)) ]; then
        END=$TOTAL_FRAMES
        FRAMES_IN_CHUNK=$((TOTAL_FRAMES - START))
    else
        END=$(((chunk_id + 1) * CHUNK_SIZE))
        FRAMES_IN_CHUNK=$CHUNK_SIZE
    fi
    
    # Add retry mechanism for GPU memory issues
    local max_retries=3
    local retry_count=0
    local success=false
    
    while [ $retry_count -lt $max_retries ] && [ "$success" = false ]; do
        # Run with GPU memory monitoring
        if python3 generative_tracker.py \
            --input "$INPUT" \
            --output "$TMP_DIR/chunk_${chunk_id}.mp4" \
            --start-frame $START \
            --end-frame $END > /dev/null 2>&1; then
            success=true
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "Chunk $chunk_id failed, retrying ($retry_count/$max_retries)..." >&2
                sleep 1
            fi
        fi
    done
    
    if [ "$success" = true ]; then
        echo "$chunk_id $FRAMES_IN_CHUNK" >> "$TMP_DIR/progress/completed_chunks.txt"
        echo "$chunk_id"
    else
        echo "Error processing chunk $chunk_id after $max_retries attempts (frames $START-$END)" >&2
        return 1
    fi
}

# Progress monitoring function (enhanced with GPU monitoring)
monitor_progress() {
    local start_time=$(cat "$PROGRESS_DIR/start_time")
    echo "1" > "$PROGRESS_DIR/monitor_active"
    
    while true; do
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
        
        local progress=0
        if [ "$TOTAL_FRAMES" -gt 0 ]; then
            progress=$((completed_frames * 100 / TOTAL_FRAMES))
        fi
        
        local fps="0.00"
        if [ "$elapsed" -gt 0 ] && [ "$completed_frames" -gt 0 ]; then
            fps=$(echo "scale=2; $completed_frames / $elapsed" | bc -l 2>/dev/null || echo "0.00")
        fi
        
        # GPU utilization info (if available)
        local gpu_info=""
        if [ "$GPU_COUNT" -gt 0 ] && [ "$GPU_MODE" != "cpu-only" ]; then
            if command -v nvidia-smi &> /dev/null; then
                local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
                if [ -n "$gpu_util" ]; then
                    gpu_info=" GPU:${gpu_util}%"
                fi
            fi
        fi
        
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
        
        local elapsed_min=$((elapsed / 60))
        local elapsed_sec=$((elapsed % 60))
        local elapsed_display=$(printf "%02d:%02d" $elapsed_min $elapsed_sec)
        
        # Enhanced progress bar
        local bar_width=25
        local filled_width=$((progress * bar_width / 100))
        local bar=""
        
        for ((i=0; i<filled_width; i++)); do
            bar+="█"
        done
        
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
        
        for ((i=filled_width; i<bar_width; i++)); do
            bar+=" "
        done
        
        printf "\r%s Processing: %3d%%|%s| %d/%d [%s<%s, %s fps%s]" \
            "🚀" "$progress" "$bar" "$completed_frames" "$TOTAL_FRAMES" \
            "$elapsed_display" "$eta_display" "$fps" "$gpu_info"
        
        sleep 0.3  # Faster updates for better feedback
    done
    
    local final_elapsed=$(($(date +%s) - start_time))
    local final_elapsed_min=$((final_elapsed / 60))
    local final_elapsed_sec=$((final_elapsed % 60))
    local final_elapsed_display=$(printf "%02d:%02d" $final_elapsed_min $final_elapsed_sec)
    local final_fps="0.00"
    if [ "$final_elapsed" -gt 0 ]; then
        final_fps=$(echo "scale=2; $TOTAL_FRAMES / $final_elapsed" | bc -l 2>/dev/null || echo "0.00")
    fi
    
    printf "\r✅ Processing: 100%%|%s| %d/%d [%s<00:00, %s fps]\n" \
        "█████████████████████████" "$TOTAL_FRAMES" "$TOTAL_FRAMES" "$final_elapsed_display" "$final_fps"
    
    echo "0" > "$PROGRESS_DIR/monitor_active"
}

export -f process_chunk

# Start optimized processing
echo "🚀 Starting GPU-optimized processing..."
if [ "$(cat "$PROGRESS_DIR/monitor_active" 2>/dev/null || echo "0")" -eq 0 ]; then
    monitor_progress &
    MONITOR_PID=$!
fi

# Check if we have the right parallel command
if command -v parallel &> /dev/null; then
    # Test if it's GNU parallel
    if parallel --version 2>/dev/null | grep -q "GNU parallel"; then
        echo "Using GNU parallel for processing..."
        SUCCESSFUL_CHUNKS=$(seq 0 $((NUM_CHUNKS-1)) | parallel --jobs $CORES --load 80% --no-notice process_chunk)
    else
        echo "Using moreutils parallel for processing..."
        # moreutils parallel has different syntax
        SUCCESSFUL_CHUNKS=$(seq 0 $((NUM_CHUNKS-1)) | parallel -j $CORES process_chunk {})
    fi
else
    echo "Parallel not available, using sequential processing..."
    SUCCESSFUL_CHUNKS=""
    for chunk_id in $(seq 0 $((NUM_CHUNKS-1))); do
        if process_chunk $chunk_id; then
            SUCCESSFUL_CHUNKS="$SUCCESSFUL_CHUNKS $chunk_id"
        fi
    done
fi

if [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null
    wait $MONITOR_PID 2>/dev/null
fi

# Validation and merging
if [ -z "$SUCCESSFUL_CHUNKS" ]; then
    echo "❌ Error: No chunks were processed successfully"
    echo "Check if generative_tracker.py exists and is working correctly."
    exit 1
fi

echo "🔗 Merging chunks..."
MERGE_LIST="$TMP_DIR/merge_list.txt"
> "$MERGE_LIST"

VALID_CHUNKS=0
for chunk_id in $(echo "$SUCCESSFUL_CHUNKS" | tr ' ' '\n' | sort -n); do
    CHUNK_FILE="$TMP_DIR/chunk_${chunk_id}.mp4"
    if [ -f "$CHUNK_FILE" ]; then
        CHUNK_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$CHUNK_FILE" 2>/dev/null || echo "0")
        if [ "$CHUNK_FRAMES" -gt 0 ]; then
            echo "file '$PWD/$CHUNK_FILE'" >> "$MERGE_LIST"
            VALID_CHUNKS=$((VALID_CHUNKS + 1))
        fi
    fi
done

if [ "$VALID_CHUNKS" -eq 0 ]; then
    echo "❌ Error: No valid chunks found for merging"
    echo "Debugging info:"
    echo "Available chunk files:"
    ls -la "$TMP_DIR"/chunk_*.mp4 2>/dev/null || echo "No chunk files found"
    exit 1
fi

# GPU-accelerated merging if available
FFMPEG_GPU_FLAGS=""
if [ "$GPU_TYPE" = "nvidia" ] && [ "$GPU_MODE" != "cpu-only" ]; then
    # Test NVENC availability
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "nvenc"; then
        FFMPEG_GPU_FLAGS="-c:v h264_nvenc"
        echo "🎮 Using NVIDIA GPU acceleration for final merge"
    fi
elif [ "$GPU_TYPE" = "amd" ] && [ "$GPU_MODE" != "cpu-only" ]; then
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "amf"; then
        FFMPEG_GPU_FLAGS="-c:v h264_amf"
        echo "🎮 Using AMD GPU acceleration for final merge"
    fi
fi

# Merge with GPU acceleration if available
if [ -n "$FFMPEG_GPU_FLAGS" ]; then
    if ! ffmpeg -y -f concat -safe 0 -i "$MERGE_LIST" $FFMPEG_GPU_FLAGS "$TMP_DIR/merged.mp4" 2>/dev/null; then
        echo "⚠️  GPU merge failed, falling back to CPU..."
        ffmpeg -y -f concat -safe 0 -i "$MERGE_LIST" -c copy "$TMP_DIR/merged.mp4"
    fi
else
    ffmpeg -y -f concat -safe 0 -i "$MERGE_LIST" -c copy "$TMP_DIR/merged.mp4"
fi

# Add audio with timing precision
echo "🎵 Adding audio..."
MERGED_DURATION=$(ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$TMP_DIR/merged.mp4")

ffmpeg -y -i "$TMP_DIR/merged.mp4" -i "$INPUT" \
    -c:v copy -c:a aac -b:a 192k \
    -map 0:v:0 -map 1:a:0 \
    -t "$MERGED_DURATION" \
    -avoid_negative_ts make_zero \
    -fflags +genpts \
    "$OUTPUT"

echo "✅ Processing completed successfully!"

# Final verification
FINAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$OUTPUT" 2>/dev/null || echo "0")
FINAL_SIZE=$(du -h "$OUTPUT" 2>/dev/null | cut -f1)

echo "📊 Final Results:"
echo "   Output: $OUTPUT"
echo "   Frames: $FINAL_FRAMES"
echo "   Size: $FINAL_SIZE"
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "   GPU: ${GPU_TYPE^^} acceleration used"
fi

# Cleanup
rm -rf "$TMP_DIR"

echo "🎉 All done!"

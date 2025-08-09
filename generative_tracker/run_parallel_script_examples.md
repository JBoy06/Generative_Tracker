# Parallel Video Processor - Usage Examples

## Basic Usage

### 1. Minimal Usage (Using Defaults)
```bash
./run_parallel.sh input_video.mp4
```
- Uses default output filename: `final_output.mp4`
- Uses default cores: 3
- Processes entire video

**Expected Output:**
```
----------------------------------------
Total frames: 2500, Target Cores: 3
Dynamic Queue: Creating 12 small chunks...
----------------------------------------
Processing chunks in parallel...
Processing:  72%|██████████████▎     | 1800/2500 [03:25<01:18,  8.76 frame/s]
```

### 2. Specify Output File
```bash
./run_parallel.sh input_video.mp4 my_processed_video.mp4
```
- Custom output filename
- Uses default cores: 3
- Processes entire video

## Advanced Usage Examples

### 3. Custom Core Count
```bash
./run_parallel.sh test.mp4 output.mp4 --cores 8
```
- Uses 8 CPU cores for parallel processing
- Processes entire video
- Good for high-end systems

### 4. Process Only First N Frames
```bash
./run_parallel.sh test.mp4 output.mp4 --cores 4 --end-frame 500
```
- Uses 4 CPU cores
- Processes only the first 500 frames
- Useful for testing or previews

### 5. Conservative Processing (Low Resource Usage)
```bash
./run_parallel.sh large_video.mp4 processed_video.mp4 --cores 2 --end-frame 1000
```
- Uses only 2 cores (good for systems under load)
- Processes first 1000 frames
- Leaves resources for other tasks

## Real-World Scenarios

### 6. Quick Test Run
```bash
./run_parallel.sh sample_footage.mp4 test_output.mp4 --cores 2 --end-frame 100
```
**Use Case**: Testing your generative_tracker.py script on a small sample before processing the full video

### 7. Full Production Run
```bash
./run_parallel.sh wedding_video_4k.mp4 final_wedding_edit.mp4 --cores 12
```
**Use Case**: Full processing of a large 4K video using maximum available cores

### 8. Batch Processing Setup
```bash
./run_parallel.sh raw_footage_01.mp4 processed_01.mp4 --cores 6 &
./run_parallel.sh raw_footage_02.mp4 processed_02.mp4 --cores 6 &
wait
```
**Use Case**: Processing multiple videos simultaneously (adjust cores to avoid oversubscription)

### 9. Memory-Constrained Environment
```bash
./run_parallel.sh high_res_video.mp4 output.mp4 --cores 2 --end-frame 2000
```
**Use Case**: Processing on a system with limited RAM where too many parallel processes would cause swapping

### 10. Development/Debugging
```bash
./run_parallel.sh debug_video.mp4 debug_output.mp4 --cores 1 --end-frame 50
```
**Use Case**: Single-threaded processing of a very small segment for debugging your Python script

## Progress Bar Features

The script now provides detailed real-time progress with:

- **Percentage completion**: Shows exact percentage (0-100%)
- **Visual progress bar**: Uses Unicode block characters for smooth progress indication
- **Frame counters**: Shows current/total frames processed
- **Timing information**: 
  - Elapsed time in MM:SS format
  - Estimated time remaining (ETA)
  - Processing speed in frames per second

**Example Progress Output:**
```
Processing:  72%|██████████████▎     | 1824/2500 [03:25<01:18,  8.91 frame/s]
Processing:  89%|█████████████████▋  | 2234/2500 [04:12<00:18, 10.45 frame/s]
Processing: 100%|████████████████████| 2500/2500 [04:30<00:00, 9.26 frame/s]
```

## Performance Optimization Examples

### 11. High-End Workstation (16+ cores)
```bash
./run_parallel.sh 8k_footage.mp4 processed_8k.mp4 --cores 16
```

### 12. Mid-Range System (8 cores)
```bash
./run_parallel.sh video.mp4 output.mp4 --cores 6
```
**Note**: Leaving 2 cores free for system tasks

### 13. Low-End System (4 cores)
```bash
./run_parallel.sh video.mp4 output.mp4 --cores 3
```
**Note**: Conservative approach leaving 1 core for OS

## Argument Order Flexibility

All of these are equivalent:
```bash
./run_parallel.sh input.mp4 output.mp4 --cores 4 --end-frame 300
./run_parallel.sh input.mp4 --cores 4 output.mp4 --end-frame 300
./run_parallel.sh input.mp4 --cores 4 --end-frame 300 output.mp4
./run_parallel.sh input.mp4 --end-frame 300 --cores 4 output.mp4
```

## Error Handling Examples

### 14. What happens with too many cores?
```bash
./run_parallel.sh video.mp4 output.mp4 --cores 50
```
**Result**: Automatically caps at your system's actual core count (e.g., 8 cores)

### 15. What happens with invalid end-frame?
```bash
./run_parallel.sh video.mp4 output.mp4 --end-frame 999999
```
**Result**: Processes entire video if end-frame exceeds total frames

## Monitoring Progress

The script provides real-time feedback:
- Total frames detected
- Number of chunks created
- Progress bar during parallel processing
- Merge and audio addition status

## Tips for Optimal Performance

1. **Core Count**: Start with `cores = physical_cores - 2` for best balance
2. **Testing**: Always test with `--end-frame 100` first on new videos
3. **Storage**: Ensure adequate disk space (chunks can temporarily use 2-3x input size)
4. **Memory**: Monitor RAM usage; reduce cores if system starts swapping
5. **I/O**: SSDs dramatically improve performance for temporary chunk files
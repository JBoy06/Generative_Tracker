# Generative_Tracker

A parallel video processing toolkit using shell scripts and Python, optimized for batch processing, scalability, and resource management.

## Features

- **Parallel Processing:** Splits videos into chunks and processes them using multiple CPU cores.
- **Progress Bar:** Real-time terminal progress with Unicode blocks, frame counters, ETA, and FPS.
- **Flexible Usage:** Supports various input/output options, core counts, and frame ranges.
- **Diagnostics:** Includes quick video checking (`check_video.sh`).
- **Robust Error Handling:** Detects missing dependencies, invalid arguments, and failed processing steps.
- **Optimized for Resource Usage:** Suitable for high-end workstations, mid-range systems, low-memory environments, and batch operations.

## Requirements

- Python 3
- [Numpy](https://numpy.org/) (Python package)
- [FFmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` command-line tools)
- [GNU Parallel](https://www.gnu.org/software/parallel/)
- `bc` (calculator utility)
- Bash shell (Linux/macOS recommended)

### Installing dependencies (Ubuntu example)

```bash
sudo apt-get update
sudo apt-get install parallel ffmpeg bc python3 python3-pip
pip3 install -r requirements.txt
```

## Quick Start

1. **Check your video file:**
   ```bash
   ./generative_tracker/check_video.sh input_video.mp4
   ```

2. **Basic video processing (using defaults):**
   ```bash
   ./generative_tracker/run_parallel.sh input_video.mp4
   ```
   - Uses default output filename: `final_output.mp4`
   - Default: 3 CPU cores

3. **Specify output file and core count:**
   ```bash
   ./generative_tracker/run_parallel.sh input_video.mp4 my_processed_video.mp4 --cores 8
   ```

4. **Process only first N frames (for testing):**
   ```bash
   ./generative_tracker/run_parallel.sh input_video.mp4 output.mp4 --end-frame 500
   ```

## Monitoring Progress

- The terminal shows:
  - Total frames, chunk count, % completion
  - Visual progress bar and frame counters
  - Elapsed time, ETA, and FPS

## Performance Tips

- **Core Count:** Use `physical_cores - 2` for best balance.
- **Testing:** Run with `--end-frame 100` first on new videos.
- **Storage:** Make sure you have enough disk space (chunks can temporarily use 2-3x input size).
- **Memory:** Monitor RAM and reduce cores if swapping occurs.
- **I/O:** SSDs are recommended for chunk file performance.

## Advanced Usage

See [`generative_tracker/run_parallel_script_examples.md`](generative_tracker/run_parallel_script_examples.md) for real-world scenarios, error handling, argument flexibility, and optimization guidelines.

## Troubleshooting

- **Too many cores:** The script automatically caps at your system's physical core count.
- **Invalid end-frame:** Processes entire video if end-frame exceeds total frames.
- **Missing dependencies:** The script checks and reports missing tools.
- **Corrupt video:** Use `check_video.sh` for diagnostics.

---

**Contributions welcome!**  
Open issues or pull requests for improvements, bugfixes, or new features.

## Processing Modes

### GPU Processing (Recommended)
Use `run_parallel_GPU.sh` to leverage automatic or manual GPU selection:

#### Automatic GPU detection (recommended)
```bash
./run_parallel_GPU.sh input.mp4 output.mp4
```

#### Force single GPU optimization
```bash
./run_parallel_GPU.sh input.mp4 output.mp4 --gpu-mode single
```

#### Use multiple GPUs if available
```bash
./run_parallel_GPU.sh input.mp4 output.mp4 --gpu-mode multi
```

#### Custom core count with GPU
```bash
./run_parallel_GPU.sh input.mp4 output.mp4 --cores 4 --gpu-mode auto
```

### CPU-only Processing
Use `run_parallel_CPU.sh` for CPU-only processing:

#### Default CPU mode
```bash
./run_parallel_CPU.sh input.mp4 output.mp4
```

#### Custom core count (CPU-only)
```bash
./run_parallel_CPU.sh input.mp4 output.mp4 --cores 4
```

### High-End Workstation (16+ cores)
```bash
./run_parallel_CPU.sh 8k_footage.mp4 processed_8k.mp4 --cores 16
```

### Mid-Range System (8 cores)
```bash
./run_parallel_CPU.sh video.mp4 output.mp4 --cores 6
```
**Note**: Leaving 2 cores free for system tasks

### Low-End System (4 cores)
```bash
./run_parallel_CPU.sh video.mp4 output.mp4 --cores 3
```
**Note**: Conservative approach leaving 1 core for OS

## Argument Order Flexibility

All of these are equivalent:
```bash
./run_parallel_CPU.sh input.mp4 output.mp4 --cores 4 --end-frame 300
./run_parallel_CPU.sh input.mp4 --cores 4 output.mp4 --end-frame 300
./run_parallel_CPU.sh input.mp4 --cores 4 --end-frame 300 output.mp4
./run_parallel_CPU.sh input.mp4 --end-frame 300 --cores 4 output.mp4
```

## Error Handling Examples

### What happens with too many cores?
```bash
./run_parallel_CPU.sh video.mp4 output.mp4 --cores 50
```
**Result**: Automatically caps at your system's actual core count (e.g., 8 cores)

### What happens with invalid end-frame?
```bash
./run_parallel_CPU.sh video.mp4 output.mp4 --end-frame 999999
```
**Result**: Processes entire video if end-frame exceeds total frames


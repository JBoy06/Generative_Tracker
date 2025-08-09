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

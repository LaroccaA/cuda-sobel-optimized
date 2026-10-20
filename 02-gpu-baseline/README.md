# 02-gpu-baseline — GPU Baseline (Global Memory)

## What this is
First CUDA port of the Sobel filter. The kernel reads the image and filter coefficients from **global memory**. No shared memory, no constant memory, no loop unrolling.

## Changes from CPU reference
| Aspect | CPU | GPU Baseline |
|--------|-----|--------------|
| Execution | 4 nested Python loops | 2D grid of threads (16×16 blocks) |
| Filter storage | Python list / NumPy array | `float*` in global memory (`d_horizontal`, `d_vertical`) |
| Border check | `if` inside innermost loop | `if` inside filter loop |
| Magnitude | `np.sqrt` + `np.clip` | `sqrtf` + device `clamp` |
| Timing | `time.time()` whole pipeline | `time.time()` around kernel + `cuda.Context.synchronize()` |

## Kernel highlights
```cuda
__global__ void sobelFilterKernel(
    unsigned char* d_input,
    unsigned char* d_output,
    float* d_horizontal,   // global memory
    float* d_vertical,     // global memory
    int width, int height, int filterSize
)
```
- Each thread computes one output pixel
- 3×3 filter loop unrolled by compiler (not manual)
- Filters passed as device pointers, copied once per run via `cudaMemcpyHtoD`

## Block configuration
- Fixed: **16×16** threads per block (256 threads)
- Grid: `(ceil(W/16), ceil(H/16))`

## Output naming
`<stem>_sobel_baseline_16x16.png`

## Purpose in the study
- "Hello world" CUDA port — minimal changes from CPU algorithm
- Establishes baseline GPU performance
- Shows overhead of global memory accesses for both image and filter data

## How to run
```bash
mkdir -p Dataset
cp /path/to/image.jpg Dataset/
python3 gpu-2D2D.py
ls Output/
```

## Build standalone executable
```bash
pyinstaller --clean gpu-2D2D.spec
```
> Requires target machine to have NVIDIA driver + CUDA Toolkit (`nvcc`) because PyCUDA compiles kernels at runtime.
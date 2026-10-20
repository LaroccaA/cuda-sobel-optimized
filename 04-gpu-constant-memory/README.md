# 04-gpu-constant-memory — Constant Memory for Filters

## What this is
Moves the Sobel filter coefficients from global memory into **constant memory** (`__constant__`). Constant memory is cached, broadcast-efficient, and ideal for small read-only data accessed uniformly by all threads (like 3×3 filter weights).

## Changes from baseline
| Aspect | Baseline (02) | Constant Memory (04) |
|--------|---------------|---------------------|
| Filter storage | `float*` in global memory | `__constant__ float d_horizontal[9], d_vertical[9]` |
| Filter access | `d_horizontal[fy*3+fx]` (global load) | `d_horizontal[fy*3+fx]` (constant cache) |
| Host→device copy | `cudaMemcpyHtoD` to `CUdeviceptr` | `cudaMemcpyHtoD` to `__constant__` symbol |
| Image access | Global memory (unchanged) | Global memory (unchanged) |
| Shared memory | None | None |

## Kernel highlights
```cuda
__constant__ float d_horizontal[9];
__constant__ float d_vertical[9];

__global__ void sobelFilterKernel(
    unsigned char* d_input,
    unsigned char* d_output,
    int width, int height, int filterSize  // filters not passed as args
) {
    // Access via constant cache — single broadcast per warp
    horizontalResult += pixelValue * d_horizontal[fy * filterSize + fx];
    verticalResult   += pixelValue * d_vertical[fy * filterSize + fx];
}
```

## Host-side constant memory setup
```python
mod = SourceModule(cuda_code)
sobel_filter_kernel = mod.get_function("sobelFilterKernel")
const_horizontal = mod.get_global("d_horizontal")[0]
const_vertical   = mod.get_global("d_vertical")[0]

cuda.memcpy_htod(const_horizontal, sobel_horizontal)
cuda.memcpy_htod(const_vertical,   sobel_vertical)
```

## Block configuration
- Fixed: **16×16** (same as baseline for direct comparison)

## Output naming
`<stem>_sobel_cm_16x16.png`

## Purpose in the study
- Isolates the effect of **constant memory** for filter coefficients
- Constant cache: 8 KB on modern GPUs, broadcasts to all threads in warp in 1 cycle when all threads read same address
- Sobel filters (9 floats × 2 = 72 bytes) fit easily — zero bank conflicts, zero global traffic for weights
- Expected win: modest (filters are small), but demonstrates the pattern for larger read-only lookup tables

## How to run
```bash
mkdir -p Dataset
cp /path/to/image.jpg Dataset/
python3 gpu-2D2D-CM.py
ls Output/
```

## Build
```bash
pyinstaller --clean gpu-2D2D-CM.spec
```
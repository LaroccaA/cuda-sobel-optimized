# 07-gpu-full-optimization — Constant + Shared + Unrolled + Block Sweep

## What this is
The most optimized variant: combines **constant memory** (filters), **shared memory** (image tiles), **manual loop unrolling** (3×3 Sobel), and a **comprehensive block-size sweep** (10 configurations).

## Changes from previous variants
| Aspect | 05 (CM+SM) | 06 (SM+UNR) | 07 (Full) |
|--------|------------|-------------|-----------|
| Filter storage | `__constant__ [3][3]` | Hardcoded literals | `__constant__ [3][3]` |
| Image tiles | Shared memory | Shared memory | Shared memory |
| Filter loop | Nested `for` | Fully unrolled | Fully unrolled |
| Halo loading | All edge threads | Boundary threads only | Boundary threads only |
| Block sizes | 16×16 only | 16×16 only | **10 shapes** |

## Kernel highlights
```cuda
__constant__ float horizontal_filter[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
__constant__ float vertical_filter[3][3]   = { {-1,-2,-1}, {0, 0, 0}, {1, 2, 1} };

extern __shared__ unsigned char shared_mem[];

int local_x = threadIdx.x + 1;
int local_y = threadIdx.y + 1;
int shared_stride = blockDim.x + 2;

// Halo load by boundary threads only
if (threadIdx.x == 0 && x > 0) ...
if (threadIdx.x == blockDim.x-1 && x < width-1) ...
if (threadIdx.y == 0 && y > 0) ...
if (threadIdx.y == blockDim.y-1 && y < height-1) ...

__syncthreads();

// Unrolled computation using constant memory
horizontal_result = shared_mem[...] * horizontal_filter[0][0] + ...;
vertical_result   = shared_mem[...] * vertical_filter[0][0]   + ...;
```

## Block configurations tested
| Block (x, y, z) | Threads | Shape | Rationale |
|-----------------|---------|-------|-----------|
| (32, 16, 1) | 512 | 2:1 rect | Wider tiles for coalescing |
| (32, 32, 1) | 1024 | Square | Max threads/block |
| (64, 16, 1) | 1024 | 4:1 rect | Tests warp alignment |
| (128, 8, 1) | 1024 | 16:1 | 1D-ish, many warps |
| (128, 4, 1) | 512 | 32:1 | Fewer warps |
| (256, 4, 1) | 1024 | 64:1 | Wide, 4 rows |
| (256, 2, 1) | 512 | 128:1 | Very wide |
| (512, 2, 1) | 1024 | 256:1 | Near 1D |
| (512, 1, 1) | 512 | 512:1 | Single row |
| (1024, 1, 1) | 1024 | 1024:1 | Max 1D block |

All configurations respect 1024 threads/block limit.

## Output naming
`<stem>_sobel_cm-sm-unr_<W>x<H>.png` (one per block size)

## Purpose in the study
- **Full factorial**: all optimizations enabled simultaneously
- **Block geometry sweep**: finds optimal block shape for this kernel on the target GPU
- Tests interaction: does unrolling help more with certain block shapes?
- Reveals occupancy vs. shared memory tradeoffs (larger blocks = more shared mem/block = fewer resident blocks)

## Known issues
- **Halo corners uninitialized** — border pixels non-deterministic
- 10× more output files per image
- Longer total runtime due to sweep

## How to run
```bash
mkdir -p Dataset
cp /path/to/image.jpg Dataset/
python3 gpu-2D2D-CM-SM-UNR-DimDiverse.py
ls Output/
```

## Build
```bash
pyinstaller --clean gpu-2D2D-CM-SM-UNR-DimDiverse.spec
```
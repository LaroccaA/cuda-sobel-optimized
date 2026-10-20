# 06-gpu-shared-unrolled — Shared Memory + Manual Loop Unrolling

## What this is
Takes the shared-memory kernel (03) and **manually unrolls** the 3×3 filter loops into explicit arithmetic. The `CM` in the filename is historical — this variant does **not** use constant memory; coefficients are hardcoded as literals.

## Changes from shared memory (03)
| Aspect | Shared Mem (03) | Shared + Unrolled (06) |
|--------|-----------------|------------------------|
| Filter loop | `for fy in 0..2: for fx in 0..2` | Fully unrolled — 6 multiplies each for H/V |
| Filter storage | `float[3][3]` in kernel | Hardcoded literals (`* -1`, `* 2`, etc.) |
| Constant memory | No | No (despite `CM` in name) |
| Shared memory | Yes | Yes |
| Halo loading | Cooperative, all edges | Only edge **threads** load halo (simpler) |
| Index calculation | `radius = filterSize/2` | Hardcoded `+1` offset |

## Kernel highlights
```cuda
// Hardcoded offsets — radius = 1 for 3×3 Sobel
int local_x = threadIdx.x + 1;
int local_y = threadIdx.y + 1;
int shared_stride = blockDim.x + 2;

// Simpler halo load: only boundary threads load their one neighbor
if (threadIdx.x == 0 && x > 0)
    shared_mem[local_y * stride + local_x - 1] = d_input[y * width + x - 1];
// ... similarly for right, top, bottom edges ...

__syncthreads();

// Fully unrolled 3×3 Sobel — no loops, no index arithmetic
horizontal_result =
    shared_mem[(local_y-1)*stride + local_x-1] * -1 +
    shared_mem[(local_y-1)*stride + local_x+1] *  1 +
    shared_mem[ local_y   *stride + local_x-1] * -2 +
    shared_mem[ local_y   *stride + local_x+1] *  2 +
    shared_mem[(local_y+1)*stride + local_x-1] * -1 +
    shared_mem[(local_y+1)*stride + local_x+1] *  1;

vertical_result =
    shared_mem[(local_y-1)*stride + local_x-1] * -1 +
    shared_mem[(local_y-1)*stride + local_x  ] * -2 +
    shared_mem[(local_y-1)*stride + local_x+1] * -1 +
    shared_mem[(local_y+1)*stride + local_x-1] *  1 +
    shared_mem[(local_y+1)*stride + local_x  ] *  2 +
    shared_mem[(local_y+1)*stride + local_x+1] *  1;
```

## Block configuration
- Fixed: **16×16**

## Output naming
`<stem>_sobel_sm-unr_16x16.png`

## Purpose in the study
- **Loop unrolling** eliminates:
  - Loop overhead (increment, compare, branch)
  - Index arithmetic inside loop
  - Register pressure from loop counters
- Lets compiler optimize instruction scheduling and use SIMD/ILP
- Hardcoded coefficients enable constant propagation
- Tests whether compute-bound optimizations matter when kernel is memory-bound

## Known issues
- **Name mismatch**: `CM` in filename but no constant memory used
- **Halo corners still uninitialized** — same border issue as 03/05
- Only tests one block size (16×16)

## How to run
```bash
mkdir -p Dataset
cp /path/to/image.jpg Dataset/
python3 gpu-2D2D-CM-SM-UNR.py
ls Output/
```

## Build
```bash
pyinstaller --clean gpu-2D2D-CM-SM-UNR.spec
```
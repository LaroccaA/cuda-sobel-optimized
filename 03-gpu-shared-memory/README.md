# 03-gpu-shared-memory — Shared Memory Tiling

## What this is
Optimizes the baseline by loading a tile of the input image into **shared memory** (on-chip SRAM). Each thread block cooperatively loads its tile plus a 1-pixel halo, then computes Sobel from shared memory instead of global memory.

## Changes from baseline
| Aspect | Baseline (02) | Shared Memory (03) |
|--------|---------------|-------------------|
| Image reads | Global memory (coalesced) | Shared memory (bank-conflict-free for 3×3) |
| Filter storage | Global memory (`float*`) | Hardcoded in kernel as `float[3][3]` |
| Halo loading | Implicit via boundary `if` | Explicit cooperative load by edge threads |
| Synchronization | None | `__syncthreads()` after tile load |
| Block sizes tested | 16×16 only | **8×8, 16×16, 32×32, 1024×1** |

## Kernel highlights
```cuda
extern __shared__ unsigned char shared_mem[];  // (blockDim.x+2) × (blockDim.y+2)
int radius = 1;  // Sobel 3×3

// Cooperative load: each thread loads its pixel + edge threads load halo
if (x < width && y < height) {
    shared_mem[local_y * stride + local_x] = d_input[y * width + x];
    if (threadIdx.x < radius) { /* load left/right halo */ }
    if (threadIdx.y < radius) { /* load top/bottom halo */ }
}
__syncthreads();

// Filter applied from shared memory
for (fy=0; fy<3; fy++) for (fx=0; fx<3; fx++) {
    pixel = shared_mem[(local_y+fy-1)*stride + (local_x+fx-1)];
    ...
}
```

## Block configurations tested
| Block | Threads | Shared mem/block | Notes |
|-------|---------|------------------|-------|
| 8×8   | 64      | 100 bytes        | High occupancy, small tiles |
| 16×16 | 256     | 324 bytes        | Baseline size |
| 32×32 | 1024    | 1156 bytes       | Max threads/block on most GPUs |
| 1024×1| 1024    | 1026 bytes       | 1D block, tests warp-level behavior |

## Output naming
`<stem>_sobel_sm_<W>x<H>.png` (one per block size)

## Purpose in the study
- Demonstrates **data reuse**: each input pixel read once from global memory, used up to 9 times from shared memory
- Tests sensitivity to block geometry (square vs 1D)
- Reveals shared memory capacity limits and bank conflicts

## Known issue
- **Halo corners not initialized** — the four corner elements of the shared memory tile (e.g., `local_x±1, local_y±1` simultaneously) are never loaded. The 1-pixel image border may contain garbage. Compare interior pixels only, or use baseline/constant-memory variants for exact CPU match.

## How to run
```bash
mkdir -p Dataset
cp /path/to/image.jpg Dataset/
python3 gpu-2D2D-SM.py
ls Output/
```

## Build
```bash
pyinstaller --clean gpu-2D2D-SM.spec
```
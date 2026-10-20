# 05-gpu-constant-shared — Constant Memory + Shared Memory

## What this is
Combines the two previous optimizations: **constant memory** for filter coefficients + **shared memory** for image tiles. This is the first variant that uses both on-chip memories together.

## Changes from previous variants
| Aspect | Shared Mem (03) | Constant Mem (04) | Combined (05) |
|--------|-----------------|-------------------|---------------|
| Filter storage | Hardcoded in kernel | `__constant__` | `__constant__ float horizontal_filter[3][3]`, `vertical_filter[3][3]` |
| Image tiles | Shared memory | Global memory | Shared memory |
| Halo loading | Cooperative, edge threads | N/A | Cooperative, edge threads |
| Constant memory | No | Yes (1D arrays) | Yes (2D arrays `[3][3]`) |

## Kernel highlights
```cuda
__constant__ float horizontal_filter[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
__constant__ float vertical_filter[3][3]   = { {-1,-2,-1}, {0, 0, 0}, {1, 2, 1} };

extern __shared__ unsigned char shared_mem[];  // (blockDim.x+2) × (blockDim.y+2)

// ... cooperative tile + halo load ...

__syncthreads();

for (fy=0; fy<3; fy++) for (fx=0; fx<3; fx++) {
    pixel = shared_mem[...];
    h += pixel * horizontal_filter[fy][fx];  // constant cache
    v += pixel * vertical_filter[fy][fx];    // constant cache
}
```

## Block configuration
- Fixed: **16×16** (to isolate memory effects from block geometry)

## Output naming
`<stem>_sobel_cm-sm_16x16.png`

## Purpose in the study
- Tests **composition**: does constant + shared = sum of individual gains?
- Constant memory removes global filter traffic; shared memory removes redundant image loads
- Both use on-chip memory but different types: constant cache (read-only, broadcast) vs shared memory (read-write, programmer-managed)
- Reveals whether kernel is **memory-bound** (likely) or compute-bound

## Known issue
- Same **uninitialized halo corners** as 03-gpu-shared-memory. Border pixels may differ from CPU.

## How to run
```bash
mkdir -p Dataset
cp /path/to/image.jpg Dataset/
python3 gpu-2D2D-CM-SM.py
ls Output/
```

## Build
```bash
pyinstaller --clean gpu-2D2D-CM-SM.spec
```
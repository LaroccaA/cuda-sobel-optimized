# CUDA Sobel Optimized — Progressive GPU Optimization Study

A didactic repository demonstrating how to accelerate a 2D convolution (Sobel edge detection) on NVIDIA GPUs using PyCUDA. Seven standalone experiments progress from a pure-Python CPU reference through six GPU variants, each adding one optimization technique.

## Repository Structure

```
cuda-sobel-optimized/
├── 01-cpu-reference/          # Pure Python baseline
│   ├── cpu_sobel.py
│   └── README.md
├── 02-gpu-baseline/           # Global memory only
│   ├── gpu-2D2D.py
│   ├── gpu-2D2D.spec
│   └── README.md
├── 03-gpu-shared-memory/      # Shared memory tiling
│   ├── gpu-2D2D-SM.py
│   └── README.md
├── 04-gpu-constant-memory/    # Constant memory for filters
│   ├── gpu-2D2D-CM.py
│   ├── gpu-2D2D-CM.spec
│   └── README.md
├── 05-gpu-constant-shared/    # Constant + shared memory
│   ├── gpu-2D2D-CM-SM.py
│   ├── gpu-2D2D-CM-SM.spec
│   └── README.md
├── 06-gpu-shared-unrolled/    # Shared memory + loop unrolling
│   ├── gpu-2D2D-CM-SM-UNR.py
│   ├── gpu-2D2D-CM-SM-UNR.spec
│   └── README.md
├── 07-gpu-full-optimization/  # All optimizations + block sweep
│   ├── gpu-2D2D-CM-SM-UNR-DimDiverse.py
│   ├── gpu-2D2D-CM-SM-UNR-DimDiverse.spec
│   └── README.md
├── Dataset/                   # Input images (git-ignored)
├── Output/                    # Generated outputs (git-ignored)
├── requirements.txt
├── smoke_test.sh
└── README.md                  # This file
```

Each subfolder is a **self-contained experiment** with its own script, PyInstaller spec, and README explaining the optimization, what changed from the previous step, and how to run it.

## The Study: Progressive CUDA Optimization

The goal is to understand how each CUDA feature affects performance on a memory-bound 2D stencil:

| Step | Folder | Optimization Added | Research Question |
|------|--------|-------------------|-------------------|
| 0 | 01-cpu-reference | — | How slow is pure Python? |
| 1 | 02-gpu-baseline | Port to CUDA, global memory | What's the raw GPU speedup? |
| 2 | 03-gpu-shared-memory | Shared memory tile + halo | Does on-chip image reuse help? |
| 3 | 04-gpu-constant-memory | `__constant__` for filters | Does constant cache help for 3×3 weights? |
| 4 | 05-gpu-constant-shared | Constant + shared together | Do gains compose? |
| 5 | 06-gpu-shared-unrolled | Manual 3×3 unrolling | Does ILP matter when memory-bound? |
| 6 | 07-gpu-full-optimization | All above + block sweep | What's the best block shape? |

### Key concepts illustrated

- **Memory hierarchy**: global → shared → constant → registers
- **Data reuse**: each input pixel used 9 times → load once to shared memory
- **Coalescing**: 2D blocks with contiguous x-dimension
- **Occupancy vs. shared memory**: larger blocks → more shared mem/block → fewer resident blocks
- **Constant cache**: broadcast to warp, ideal for uniform read-only data
- **Loop unrolling**: eliminates index arithmetic, enables instruction-level parallelism
- **Block geometry**: square vs. rectangular vs. 1D — affects coalescing, occupancy, halo overhead

### What this is NOT

- A production image processing library (use OpenCV, cuDNN, or custom kernels with tensor cores)
- A rigorous benchmark suite (no statistical runs, no multiple GPUs, no NCU logs)
- A tutorial on PyCUDA best practices (kernels are strings, duplicated across files)

## Quick Start

```bash
# 1. Create virtualenv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -c "import pycuda.autoinit; print('PyCUDA OK')"

# 2. Add at least one test image
mkdir -p Dataset
cp /path/to/photo.jpg Dataset/

# 3. Run any experiment from repo root (scripts expect ../Dataset and ../Output)
cd 02-gpu-baseline
python3 gpu-2D2D.py
ls ../Output/
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU + driver (`nvidia-smi`)
- CUDA Toolkit with `nvcc` on `PATH` (`nvcc --version`)
- C++ compiler (for PyCUDA build from source)
- `numpy`, `Pillow`, `pycuda` (see `requirements.txt`)

## Output Naming Convention

All scripts write to `../Output/` (created automatically):

| Experiment | Token | Example (`HD.jpg`) |
|------------|-------|-------------------|
| CPU reference | `cpu` | `HD_sobel_cpu.png` |
| GPU baseline | `baseline` | `HD_sobel_baseline_16x16.png` |
| Shared memory | `sm` | `HD_sobel_sm_16x16.png` |
| Constant memory | `cm` | `HD_sobel_cm_16x16.png` |
| Constant + shared | `cm-sm` | `HD_sobel_cm-sm_16x16.png` |
| Shared + unrolled | `sm-unr` | `HD_sobel_sm-unr_16x16.png` |
| Full optimization | `cm-sm-unr` | `HD_sobel_cm-sm-unr_32x16.png` |

Format: `<stem>_sobel_<token>[_<W>x<H>].png` (CPU omits block size).

Running the same variant + block size overwrites the file — intentional. For independent runs, use separate output directories.

## Smoke Test

```bash
./smoke_test.sh
# Or test a specific variant:
SCRIPT=../03-gpu-shared-memory/gpu-2D2D-SM.py EXPECT_GLOB="*_sobel_sm_16x16.png" ./smoke_test.sh
```

Copies one image from `Dataset/` to a temp directory, runs the script, verifies a non-zero output PNG appears.

## PyInstaller Builds

Each GPU experiment has a matching `.spec`:

```bash
cd 02-gpu-baseline
pyinstaller --clean gpu-2D2D.spec
```

The executable is **not fully standalone** — PyCUDA compiles kernels at runtime, so the target machine needs NVIDIA driver + CUDA Toolkit.

## Known Limitations

- **CPU reference**: 4 nested loops, crashes on `height < 10` (progress print division)
- **Shared-memory variants (03, 05, 06, 07)**: halo corners uninitialized → 1-pixel border may differ from CPU. Compare interior pixels only.
- **Timers**: measure kernel launch + `cuda.Context.synchronize()` only — exclude allocation, H2D/D2H copies, disk I/O
- **Kernel duplication**: CUDA source is copy-pasted in all 6 GPU scripts (no shared module). `KERNEL SYNC` comments mark this.
- **No CI, tests, or lint** — verification is `smoke_test.sh` + visual comparison

## Historical Data Note

The `origin/Andrea` branch contains CSV results from a **rotation experiment** (8 angles per image), not Sobel. They are not comparable and are not included here. Run the scripts to generate your own Sobel timings.

## License

MIT — use freely for learning, teaching, or as a starting point for real projects.
# CUDA - Sobel Optimized

A comparative study of Sobel edge detection: a pure-Python CPU reference versus six progressively optimized PyCUDA GPU kernels (baseline, shared memory, constant memory, manual loop unrolling, and block-shape exploration).

## Overview

This project applies the Sobel operator to grayscale images and saves the gradient magnitude as PNG. The CPU reference is implemented in pure Python; GPU variants use PyCUDA with the CUDA kernel embedded as a string and compiled at runtime.

**Goal:** Educational/experimental comparison of correctness and timing across optimization strategies—not a production-ready library.

## File Structure

| File | Role |
|------|------|
| `cpu_sobel.py` | Pure-Python CPU reference |
| `gpu-2D2D.py` | Baseline GPU, filters in global memory |
| `gpu-2D2D-SM.py` | Shared-memory tiling |
| `gpu-2D2D-CM.py` | Constant-memory filters |
| `gpu-2D2D-CM-SM.py` | Constant memory + shared memory |
| `gpu-2D2D-CM-SM-UNR.py` | Shared memory + manually unrolled 3×3 kernel (CM in name only) |
| `gpu-2D2D-CM-SM-UNR-DimDiverse.py` | Constant + shared + unrolled, block-shape sweep |
| `*.spec` | PyInstaller build configs for GPU scripts |
| `Dataset/` | Local input images (not versioned) |
| `Output/` | Generated outputs (not versioned) |
| `smoke_test.sh` | Quick single-image sanity check |
| `requirements.txt` | Python dependencies |

## Prerequisites

- Python 3.10+ (tested on 3.12)
- NVIDIA GPU with drivers (`nvidia-smi`)
- CUDA Toolkit with `nvcc` on `PATH` (`nvcc --version`)
- C++ compiler for PyCUDA build from source
- Python packages: `numpy`, `Pillow`, `pycuda`

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -c "import pycuda.autoinit; print('PyCUDA OK')"
```

## Quick Start

```bash
mkdir -p Dataset
cp /path/to/image.jpg Dataset/

python3 gpu-2D2D.py
ls Output/
```

`Dataset/` must exist and contain at least one `.png`, `.jpg`, or `.jpeg`. `Output/` is created automatically. Scripts have no CLI arguments—paths and block sizes are set in each `__main__` block.

The `main` branch does not track `Dataset/`; sample images are on branch `origin/Andrea`. Do not commit local images or generated outputs.

## Variants

| Script | Optimization | Block size(s) | Example output (`HD.jpg`) |
|--------|--------------|---------------|---------------------------|
| `cpu_sobel.py` | Pure Python reference | n/a | `HD_sobel_cpu.png` |
| `gpu-2D2D.py` | Baseline, filters in global memory | 16×16 | `HD_sobel_baseline_16x16.png` |
| `gpu-2D2D-SM.py` | Shared memory | 8×8, 16×16, 32×32, 1024×1 | `HD_sobel_sm_16x16.png` |
| `gpu-2D2D-CM.py` | Constant memory | 16×16 | `HD_sobel_cm_16x16.png` |
| `gpu-2D2D-CM-SM.py` | Constant + shared memory | 16×16 | `HD_sobel_cm-sm_16x16.png` |
| `gpu-2D2D-CM-SM-UNR.py` | Shared + manually unrolled 3×3; **no** constant memory despite name | 16×16 | `HD_sobel_sm-unr_16x16.png` |
| `gpu-2D2D-CM-SM-UNR-DimDiverse.py` | Constant + shared + unrolled | 10 shapes (32×16 … 1024×1) | `HD_sobel_cm-sm-unr_32x16.png` |

> **Note:** `gpu-2D2D-CM-SM-UNR.py` retains `CM` in its name for historical reasons but does **not** use constant memory—the Sobel coefficients are hardcoded in the kernel.

## Output Naming

GPU variants:
```
<stem>_sobel_<variant>_<W>x<H>.png
```

CPU reference (no block size):
```
<stem>_sobel_cpu.png
```

Variant tokens:

| Variant | Token |
|---------|-------|
| `cpu_sobel.py` | `cpu` |
| `gpu-2D2D.py` | `baseline` |
| `gpu-2D2D-SM.py` | `sm` |
| `gpu-2D2D-CM.py` | `cm` |
| `gpu-2D2D-CM-SM.py` | `cm-sm` |
| `gpu-2D2D-CM-SM-UNR.py` | `sm-unr` |
| `gpu-2D2D-CM-SM-UNR-DimDiverse.py` | `cm-sm-unr` |

Running the same variant and block size overwrites the previous file—this is intentional. For independent experiment runs, use separate `Output/` directories.

## Regenerating Outputs

To start fresh, list and then delete only the PNGs in `Output/`:

```bash
find Output -maxdepth 1 -type f -name '*.png' -print
# If the list looks correct:
find Output -maxdepth 1 -type f -name '*.png' -delete
```

The naming scheme avoids collisions between baseline/CM and SM/CM-SM. The CPU reference is very slow on large images—use downscaled images for CPU vs GPU comparison.

## Results

No versioned Sobel benchmarks exist in this repository. The CSVs recovered from `origin/Andrea` (`report.csv` and `Report/RISULTATI -CPU(Foglio1).csv`) belong to a previous **rotation** experiment (column `Angle`) and are not comparable to Sobel variants.

Historical CPU rotation times (8 angles per image), included only as an archive:

| Image | Resolution | Mean (s) | Min (s) | Max (s) |
|-------|------------|---------:|--------:|--------:|
| `640x960.jpg` | 640×960 | 2.27 | 1.78 | 2.83 |
| `HD.jpg` | 1200×673 | 1.43 | 1.02 | 1.90 |
| `FullHD.jpg` | 1920×1080 | 7.09 | 2.84 | 9.72 |
| `4K.jpg` | 4368×2893 | 28.93 | 14.92 | 60.43 |
| `8K.jpg` | 4480×6720 | 107.63 | 82.85 | 136.37 |
| `Sopra 8k.jpg` | 6336×9520 | 191.89 | 71.89 | 272.38 |

To obtain Sobel timings, run the scripts and note the printed kernel times. Remember: timers measure **kernel launch + `cuda.Context.synchronize()` only**—they exclude allocation, host↔device copies, and disk I/O.

## Kernel Duplication

The CUDA source is duplicated across all six GPU scripts. There is no shared module; any change to Sobel coefficients, clamping, `sqrtf`, or halo handling must be replicated in every variant.

Each GPU script includes a `KERNEL SYNC` comment as a reminder. Extracting kernels into a future `kernels.py` is possible but requires updating imports, PyInstaller specs, and all variants—outside the minimal setup.

## PyInstaller Builds

Each GPU script has a matching `.spec`:

```bash
pyinstaller --clean gpu-2D2D.spec
```

If you rename a script, update both the path in `Analysis([...])` and the `name` field in the spec. `gpu.spec` is a legacy file pointing to the deleted `gpu.py`—do not use it until updated.

**Important:** The produced executable is **not fully standalone**. PyCUDA compiles kernels at runtime, so the target machine must have NVIDIA drivers and CUDA Toolkit/`nvcc`.

## Known Limitations

- CPU reference uses four nested Python loops—very slow on large images.
- `cpu_sobel.py` crashes on images with height < 10 px due to `height // 10` progress division.
- Shared-memory kernels leave halo cells outside the image (including block corners) uninitialized; the 1-pixel border may differ from CPU or be non-deterministic. Compare interior pixels, or use baseline/CM variants for direct CPU comparison.
- Timers measure kernel launch + `cuda.Context.synchronize()` only—no allocation, transfers, or I/O.
- Performance depends on GPU architecture and CUDA toolkit version; optimal block sizes are not universal.
- No automated tests, CI, or lint. Verification: `smoke_test.sh` + visual/instrumental output comparison.
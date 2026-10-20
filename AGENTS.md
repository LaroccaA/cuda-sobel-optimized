# AGENTS.md

## What this is

Project: **CUDA - Sobel Optimized** (GitHub repo `cuda-sobel-optimized`). Standalone Sobel edge-detection experiments comparing a pure-Python CPU reference with several PyCUDA GPU kernels. There is no package, test suite, linter, or CI. Each script is independent; the CUDA kernel is embedded as a Python string and compiled at import time.

## Setup and run

- Dependencies: `numpy`, `Pillow`, `pycuda`; minimum versions are in `requirements.txt`. PyCUDA also needs an NVIDIA GPU and the CUDA toolkit (`nvcc`) because kernels compile at startup.
- In this checkout the default `/usr/bin/python3` has none of these packages installed and `nvcc` is not on `PATH`; set up a virtualenv and CUDA toolkit before running.
- Run from the repo root, e.g. `python3 gpu-2D2D.py` or `python3 cpu_sobel.py`.
- Paths are hardcoded relative to the current directory: input `Dataset/` (must exist) and output `Output/` (auto-created). There are no CLI arguments; edit the `__main__` block to change paths or block sizes.
- `Dataset/` is not tracked on `main`. Sample images (`HD.jpg`, `FullHD.jpg`, `4K.jpg`, `8K.jpg`, etc.) are available on `origin/Andrea`.
- GPU scripts process every image in `Dataset/` and every configured block size. Use a one-image `Dataset/` for smoke tests; the CPU script is a quadruple Python loop and is very slow on large images.
- `./smoke_test.sh` runs the baseline on one image in a temporary directory (without touching the repo's `Dataset/` or `Output/`). It needs the Python dependencies and a working GPU.

## Variant map

| Script | Optimization | Block size(s) |
| --- | --- | --- |
| `gpu-2D2D.py` | Baseline: filters in global memory | 16x16 |
| `gpu-2D2D-SM.py` | Shared-memory tile | 8x8, 16x16, 32x32, 1024x1 |
| `gpu-2D2D-CM.py` | Filters in constant memory | 16x16 |
| `gpu-2D2D-CM-SM.py` | Constant memory + shared memory | 16x16 |
| `gpu-2D2D-CM-SM-UNR.py` | Shared memory + manually unrolled 3x3 | 16x16 |
| `gpu-2D2D-CM-SM-UNR-DimDiverse.py` | Constant + shared + unrolled | 10 shapes from 32x16 to 1024x1 |
| `cpu_sobel.py` | Pure-Python reference implementation | n/a |

`gpu-2D2D-CM-SM-UNR.py` has `CM` in its name but does **not** use constant memory; the filters are hardcoded in the kernel.

## Output naming

GPU outputs use `<stem>_sobel_<variant>_<W>x<H>.png`; the CPU uses `<stem>_sobel_cpu.png`. Variant tokens are `baseline`, `sm`, `cm`, `cm-sm`, `sm-unr`, and `cm-sm-unr`. The old cross-variant collisions (baseline/CM and SM/CM-SM) are resolved by these tokens. Running the same variant and block again overwrites that file intentionally; use separate `Output/` directories for independent runs.

## Gotchas

- The CUDA kernel is duplicated in every GPU script. There is no shared module, so kernel changes must be replicated across all variants. Each GPU script has a `KERNEL SYNC` header comment.
- Timers measure only kernel launch plus `cuda.Context.synchronize()`. They exclude allocation, host-to-device/device-to-host copies, and disk I/O, so they are not end-to-end timings.
- Shared-memory kernels leave halo cells outside the image (including block corners) uninitialized, so the 1-pixel border can contain garbage. Compare interior pixels, or use the non-shared baseline/CM variants for direct CPU comparisons.
- `cpu_sobel.py` crashes on images with height < 10 because its progress print divides by `height // 10`.
- PyInstaller specs exist for each GPU script; build with `pyinstaller --clean gpu-2D2D.spec`. `gpu.spec` is legacy/stale and references the deleted `gpu.py`: leave it untouched and do not use it until repointed. The other six specs match their scripts; keep a spec's source path and output name in sync if renaming a script.
- There is no test or lint command. Verification means `bash -n smoke_test.sh`, running the smoke test, and inspecting the generated images in `Output/`.
- Keep `README.md` in sync when changing commands, output names, or prerequisites.
- The "Results" table in `README.md` is historical rotation data recovered from `origin/Andrea`, not Sobel benchmark data. Do not mix those numbers with Sobel results.

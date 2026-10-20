# 01-cpu-reference — Pure-Python CPU Reference

## What this is
Sequential Sobel edge detection implemented entirely in Python with NumPy. Serves as the correctness baseline and performance floor for all GPU variants.

## Algorithm
- Converts input images to grayscale (8-bit)
- Applies two 3×3 Sobel kernels (horizontal `[-1,0,1; -2,0,2; -1,0,1]` and vertical `[-1,-2,-1; 0,0,0; 1,2,1]`) via nested loops
- Computes gradient magnitude: `sqrt(H² + V²)` with clamping to [0, 255]
- Saves output as `<stem>_sobel_cpu.png`

## Implementation details
- **Four nested loops** (y, x, ky, kx) — O(H·W·K²) with K=3
- Progress printed every 10% of rows (`height // 10`)
- No SIMD, no multiprocessing, no JIT — pure interpreted Python
- Crashes on images with `height < 10` due to division by zero in progress print

## Purpose in the study
- Ground truth for numerical correctness (compare interior pixels only)
- Baseline to quantify GPU speedup
- Demonstrates why data-parallel image filters are a textbook GPU use case

## How to run
```bash
mkdir -p Dataset
cp /path/to/image.jpg Dataset/
python3 cpu_sobel.py
ls Output/
```

## Expected output
```
HD_sobel_cpu.png
```

## Known limitations
- Extremely slow on large images (minutes for 4K+)
- Border handling via conditional inside innermost loop
- No parallelism — single CPU core only
# Optical Flow Estimation in Python

Python reimplementation of the MATLAB codebase (https://cs.brown.edu/people/mjblack/code.html) from:

> **"Secrets of Optical Flow Estimation and Their Principles"**
> Deqing Sun, Stefan Roth, and Michael J. Black
> *IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2010*

This repository contains two packages:

- **`optical_flow`** -- Python port of the original MATLAB code. 
- **`flow_fast`** -- High-performance drop-in replacement using Numba JIT, OpenCV, and an optimized PCG solver.


## Features

- **Four optical flow methods:**
  - **Horn-Schunck (HS)** -- Laplacian spatial regularization
  - **Black-Anandan (BA)** -- Robust penalties with GNC optimization
  - **Classic+NL** -- Non-local term with color-guided weighted median filtering
  - **Alternative BA (Alt-BA)** -- Auxiliary flow field with Li-Osher denoising

- **10 robust penalty functions:** quadratic, lorentzian, charbonnier, generalized charbonnier, Geman-McClure, Huber, Tukey biweight, Gaussian, Student-t, and unnormalized Student-t

- **Complete pipeline:** Gaussian image pyramids, ROF structure-texture decomposition, Hermite bicubic interpolation, IRLS optimization, sparse linear system solvers (PCG, direct, SOR), occlusion detection, weighted median filtering

- **Middlebury .flo I/O:** read and write standard .flo flow files

- **Visualization:** Middlebury color coding, quiver plots, magnitude maps, HSV encoding

- **Evaluation metrics:** Average Angular Error (AAE) and Average Endpoint Error (AEPE)

- **`flow_fast` acceleration backends:**
  - Numba `@njit(parallel=True)` for weighted median, ROF denoising, penalty functions, bicubic interpolation
  - OpenCV for image warping (`cv2.remap`), filtering (`cv2.filter2D`), pyramid construction (`cv2.resize`)
  - PCG solver with Jacobi preconditioner (replaces SuperLU `spsolve`)
  - Optional CHOLMOD direct solver via scikit-sparse

## Installation

```bash
cd flow_code_python
pip install -e ".[fast]"
```

This installs both `optical_flow` and `flow_fast` with all dependencies. To install only the base package:

```bash
pip install -e .
```

For development (tests + notebooks):

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_derivatives.py
```

## Quick Start

```python
import numpy as np
from PIL import Image

# --- Use the fast version (recommended) ---
from flow_fast import estimate_flow, flow_to_color, plot_flow

# Load two consecutive frames
im1 = np.array(Image.open('frame1.png')).astype(float)
im2 = np.array(Image.open('frame2.png')).astype(float)

# Estimate optical flow (Classic+NL-fast is the recommended default)
uv = estimate_flow(im1, im2, method='classic+nl-fast')

# uv is (H, W, 2): uv[:,:,0] = horizontal, uv[:,:,1] = vertical

# Visualize as Middlebury color image
color_img = flow_to_color(uv)

# Or use matplotlib
ax = plot_flow(uv, style='color')
```

The original `optical_flow` package has the same API:

```python
from optical_flow import estimate_flow  # identical interface, just slower
```

## Available Methods

| Method Name | Description |
|---|---|
| `'classic+nl-fast'` | Classic+NL with reduced iterations (recommended) |
| `'classic+nl'` | Classic+NL with texture decomposition and weighted median |
| `'classic+nl-full'` | Classic+NL with full weighted median version |
| `'hs'` | Horn-Schunck with ROF texture constancy |
| `'hs-brightness'` | Horn-Schunck with brightness constancy |
| `'ba'` / `'classic-l'` | Black-Anandan with lorentzian, texture |
| `'ba-brightness'` | Black-Anandan with brightness constancy |
| `'classic-c'` | Classic with charbonnier penalties, texture |
| `'classic-c-brightness'` | Classic with charbonnier, brightness |
| `'classic++'` | Classic++ with generalized charbonnier, bi-cubic interpolation |
| `'classic-c-a'` | Alt-BA with charbonnier penalties |

## Using Pre-configured Methods

```python
from optical_flow import estimate_flow

# Use a named method with default parameters
uv = estimate_flow(im1, im2, method='classic+nl-fast')

# Override specific parameters
uv = estimate_flow(im1, im2, method='hs', params={'lambda': 50, 'max_iters': 15})
```

## Using Method Classes Directly

```python
from optical_flow.methods import HSOpticalFlow, BAOpticalFlow, ClassicNLOpticalFlow
from optical_flow.robust.robust_function import RobustFunction
import numpy as np

# Horn-Schunck
hs = HSOpticalFlow()
hs.lambda_ = 40
hs.texture = True
hs.images = np.stack([gray1, gray2], axis=2)
uv = hs.compute_flow(np.zeros((H, W, 2)))

# Black-Anandan with custom penalties
ba = BAOpticalFlow()
ba.rho_spatial_u = [RobustFunction('lorentzian', 0.03),
                    RobustFunction('lorentzian', 0.03)]
ba.rho_spatial_v = [RobustFunction('lorentzian', 0.03),
                    RobustFunction('lorentzian', 0.03)]
ba.rho_data = RobustFunction('lorentzian', 1.5)
ba.images = np.stack([gray1, gray2], axis=2)
uv = ba.compute_flow(np.zeros((H, W, 2)))
```

## Loading and Evaluating Middlebury Sequences

```python
from optical_flow import estimate_flow, flow_angular_error
from optical_flow.io.flo_io import read_flow_file, read_flo, write_flo

# Load RubberWhale test sequence (images + ground truth)
im1, im2, tu, tv = read_flow_file('RubberWhale', 10)

# Estimate flow
uv = estimate_flow(im1, im2, method='classic+nl-fast')

# Compare against ground truth
aae, std_ae, aepe = flow_angular_error(tu, tv, uv[:,:,0], uv[:,:,1])
print(f'Average Angular Error: {aae:.2f} degrees')
print(f'Average Endpoint Error: {aepe:.3f} pixels')

# Read/write .flo files directly
flow = read_flo('ground_truth.flo')
write_flo(uv, 'output.flo')
```

## Visualization

```python
from optical_flow import plot_flow, flow_to_color
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Middlebury color coding
plot_flow(uv, style='color', ax=axes[0, 0])

# Quiver plot
plot_flow(uv, style='quiver', ax=axes[0, 1], step=8)

# Flow magnitude
plot_flow(uv, style='magnitude', ax=axes[1, 0])

# HSV encoding (hue=direction, value=magnitude)
plot_flow(uv, style='hsv', ax=axes[1, 1])

plt.tight_layout()
plt.savefig('flow_visualization.png')
```

## Notebooks

Jupyter notebooks are provided in `notebooks/`. Each `optical_flow` notebook has a corresponding `flow_fast` version with identical analysis but using the accelerated package:

| `optical_flow` (original) | `flow_fast` (accelerated) | Description |
|---|---|---|
| `optical_flow_demo.ipynb` | `flow_fast_demo.ipynb` | Classic+NL-Fast on RubberWhale with visualization and GT evaluation |
| `optical_flow_demo_additional.ipynb` | `flow_fast_demo_additional.ipynb` | Multi-method comparison, penalty functions, pyramids, parameter sensitivity |
| `middlebury_benchmark.ipynb` | `flow_fast_benchmark.ipynb` | Full benchmark on 8 Middlebury sequences with error maps and bar charts |


## Package Structure

```
flow_code_python/
├── setup.py
├── requirements.txt
├── optical_flow/               # Original Python port (scipy-based)
│   ├── __init__.py
│   ├── interface.py            # estimate_flow() high-level API
│   ├── methods/                # HS, BA, Classic+NL, Alt-BA
│   ├── robust/                 # Penalty functions + RobustFunction
│   ├── utils/                  # Derivatives, pyramid, warping, weighted median
│   ├── io/                     # .flo file I/O
│   ├── viz/                    # Flow visualization
│   └── evaluation/             # AAE, EPE metrics
├── flow_fast/                  # Accelerated version (same API)
│   ├── __init__.py
│   ├── interface.py            # Same estimate_flow() API
│   ├── methods/                # Same methods, wired to fast backends
│   ├── _accel/                 # Numba JIT kernels (weighted median, ROF, etc.)
│   ├── solvers/                # PCG + CHOLMOD solver dispatch
│   ├── robust/                 # Numba-accelerated penalty functions
│   ├── utils/                  # OpenCV-based derivatives, pyramid, warping
│   ├── io/                     # .flo file I/O (unchanged)
│   ├── viz/                    # Flow visualization (unchanged)
│   └── evaluation/             # AAE, EPE metrics (unchanged)
├── data/                       # Middlebury sequences
│   ├── other-data/             # Image pairs (frame10.png, frame11.png)
│   └── other-gt-flow/          # Ground truth flow (.flo files)
├── notebooks/                  # Demo notebooks (optical_flow + flow_fast versions)
└── tests/                      # Unit and integration tests
```

## Tests

Run the test suite with:

```bash
cd flow_code_python
pytest tests/ -v
```

82 tests cover robust functions, .flo I/O, sparse operators, image derivatives, pyramid construction, evaluation metrics, and integration tests for each method (HS, BA, Classic+NL).

## References

- D. Sun, S. Roth, and M. J. Black. "Secrets of Optical Flow Estimation and Their Principles." *CVPR*, 2010.
- B. Horn and B. Schunck. "Determining Optical Flow." *Artificial Intelligence*, 1981.
- M. J. Black and P. Anandan. "The Robust Estimation of Multiple Motions." *CVIU*, 1996.
- S. Baker et al. "A Database and Evaluation Methodology for Optical Flow." *IJCV*, 2011.

## License

This code is provided for **research purposes only**, consistent with the original MATLAB release from Brown University. **Commercial use is strictly prohibited.**

See [LICENSE](LICENSE) file for full terms.

**Important:** If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{sun2010secrets,
  title={Secrets of optical flow estimation and their principles},
  author={Sun, Deqing and Roth, Stefan and Black, Michael J},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2010}
}
```

# Optical Flow Estimation in Python

Python reimplementation of the MATLAB codebase (https://cs.brown.edu/people/mjblack/code.html) from:

> **"Secrets of Optical Flow Estimation and Their Principles"**
> Deqing Sun, Stefan Roth, and Michael J. Black
> *IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2010*

This package provides classical variational optical flow methods with robust penalty functions, coarse-to-fine estimation, and graduated non-convexity (GNC) optimization. 

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

## Installation

```bash
cd flow_code_python
pip install -e .
```

Or install dependencies manually:

```bash
pip install numpy scipy matplotlib Pillow scikit-image
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details on testing and development.

## Quick Start

```python
import numpy as np
from PIL import Image
from optical_flow import estimate_flow, flow_to_color, plot_flow

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

## Available Methods

| Method Name | Description | Speed |
|---|---|---|
| `'classic+nl-fast'` | Classic+NL with reduced iterations (recommended) | Fast |
| `'classic+nl'` | Classic+NL with texture decomposition and weighted median | Medium |
| `'classic+nl-full'` | Classic+NL with full weighted median version | Slow |
| `'hs'` | Horn-Schunck with ROF texture constancy | Fast |
| `'hs-brightness'` | Horn-Schunck with brightness constancy | Fast |
| `'ba'` / `'classic-l'` | Black-Anandan with lorentzian, texture | Medium |
| `'ba-brightness'` | Black-Anandan with brightness constancy | Medium |
| `'classic-c'` | Classic with charbonnier penalties, texture | Medium |
| `'classic-c-brightness'` | Classic with charbonnier, brightness | Medium |
| `'classic++'` | Classic++ with generalized charbonnier, bi-cubic interpolation | Medium |
| `'classic-c-a'` | Alt-BA with charbonnier penalties | Slow |

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

Two Jupyter notebooks are provided in `notebooks/`:

- **`optical_flow_demo.ipynb`** -- Runs Classic+NL-Fast on the RubberWhale sequence, displays color-coded and vector flow fields, and evaluates against ground truth. Matches the original MATLAB `estimate_flow_demo.m`.

- **`middlebury_benchmark.ipynb`** -- Benchmarks all methods on the 8 Middlebury training sequences (Dimetrodon, Grove2, Grove3, Hydrangea, RubberWhale, Urban2, Urban3, Venus). Datasets are downloaded automatically on first run. Includes results tables, flow visualizations, error maps, and a multi-method comparison.

## Robust Penalty Functions

Each penalty function supports three evaluation modes: value (`d_type=0`), first derivative (`d_type=1`), and IRLS weight `rho'(x)/x` (`d_type=2`).

| Function | Formula | Properties |
|---|---|---|
| `quadratic` | `x^2 / sigma^2` | Convex, non-robust |
| `lorentzian` | `log(1 + x^2/(2*sigma^2))` | Robust, smooth |
| `charbonnier` | `sigma^2 * sqrt(1 + (x/sigma^2)^2)` | Robust, smooth, L1-like |
| `generalized_charbonnier` | `(sigma^2 + x^2)^a` | Tunable robustness via exponent `a` |
| `geman_mcclure` | `x^2 / (sigma^2 + x^2)` | Robust, redescending |
| `huber` | piecewise quadratic/linear | Robust, piecewise smooth |
| `tukey` | biweight function | Robust, compact support |
| `gaussian` | `-log N(x; 0, sigma)` | Quadratic (non-robust) |
| `tdist` | Student-t negative log-likelihood | Robust, heavy tails |
| `tdist_unnorm` | unnormalized Student-t | Robust, heavy tails |

## Key Parameters

| Parameter | Description | Typical Values |
|---|---|---|
| `lambda_` | Regularization weight | 3--80 (depends on method) |
| `pyramid_spacing` | Downsampling ratio between levels | 2.0 |
| `gnc_iters` | GNC stages (1 = no GNC) | 1--3 |
| `max_iters` | Warping iterations per pyramid level | 3--10 |
| `texture` | Use ROF structure-texture decomposition | `True`/`False` |
| `median_filter_size` | Weighted median filter size | `[5, 5]` |
| `interpolation_method` | Image interpolation | `'cubic'`, `'bi-linear'` |

## Package Structure

```
flow_code_python/
├── setup.py
├── requirements.txt
├── optical_flow/
│   ├── __init__.py              # Public API exports
│   ├── interface.py             # estimate_flow() high-level API
│   ├── methods/
│   │   ├── base.py              # BaseOpticalFlow abstract class
│   │   ├── hs.py                # Horn-Schunck
│   │   ├── ba.py                # Black-Anandan
│   │   ├── classic_nl.py        # Classic+NL
│   │   ├── alt_ba.py            # Alternative BA
│   │   └── config.py            # load_of_method() factory
│   ├── robust/
│   │   ├── penalties.py         # Penalty function implementations
│   │   └── robust_function.py   # RobustFunction wrapper class
│   ├── utils/
│   │   ├── derivatives.py       # Spatiotemporal derivatives with warping
│   │   ├── pyramid.py           # Gaussian image pyramid
│   │   ├── sparse_ops.py        # Sparse convolution matrices
│   │   ├── warping.py           # Flow field resampling
│   │   ├── image_processing.py  # ROF decomposition, Gaussian kernels
│   │   ├── occlusion.py         # Occlusion detection
│   │   ├── weighted_median.py   # Color-guided weighted median filtering
│   │   └── denoising.py         # Li-Osher median denoising
│   ├── io/
│   │   └── flo_io.py            # .flo file read/write, sequence loader
│   ├── viz/
│   │   ├── flow_color.py        # Middlebury color coding
│   │   └── plot_flow.py         # Flow visualization
│   └── evaluation/
│       └── metrics.py           # AAE, EPE metrics
├── data/                        # Middlebury sequences
│   ├── other-data/              # Image pairs (frame10.png, frame11.png)
│   └── other-gt-flow/           # Ground truth flow (.flo files)
├── notebooks/
│   ├── optical_flow_demo.ipynb        # Single-sequence demo
│   └── middlebury_benchmark.ipynb     # Full benchmark with comparisons
└── tests/                       # Unit and integration tests (82 tests)
```

## Tests

Run the test suite with:

```bash
cd flow_code_python
pytest tests/ -v
```

82 tests cover robust functions, .flo I/O, sparse operators, image derivatives, pyramid construction, evaluation metrics, and integration tests for each method (HS, BA, Classic+NL).

## MATLAB Compatibility

This implementation closely follows the original MATLAB code, including:

- Column-major (Fortran) ordering for array operations
- MATLAB-style rounding (half away from zero) for `rgb2gray` and pyramid construction
- MATLAB `imresize` coordinate convention for bilinear interpolation
- Exact ROF structure-texture decomposition with [-1, 1] normalization
- BT.709 RGB-to-Lab color space conversion with D65 white point
- Hermite bicubic interpolation with analytical spatial derivatives

On the RubberWhale sequence, Classic+NL-Fast produces AAE 2.463 (MATLAB reference: 2.401).

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

For commercial uses, contact the Technology Venture Office of Brown University.

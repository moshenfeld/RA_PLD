# Module Documentation

This file maps current modules to responsibilities and key entry points.

## Core Package: `PLD_accounting`

### `types.py`
Defines shared enums and dataclasses:
- `BoundType`, `SpacingType`, `ConvolutionMethod`, `Direction`
- `PrivacyParams`, `AllocationSchemeConfig`

### `discrete_dist.py`
Defines `DiscreteDist`:
- Validates monotone grid, nonnegative PMF, and shape consistency
- Supports mass checks and domination-aware edge truncation

### `core_utils.py`
Numeric helpers and shared tolerances:
- Spacing checks (`compute_bin_width`, `compute_bin_ratio`, ...)
- Stable equality checks
- Infinite-mass convolution helpers
- `enforce_mass_conservation`

### `distribution_discretization.py`
Continuous-to-discrete conversion and grid transforms:
- `discretize_continuous_distribution`
- `discretize_continuous_to_pmf`
- `discretize_aligned_range`
- `change_spacing_type`

### `utils.py`
Shared composition/bounds helpers:
- `binary_self_convolve`
- `combine_distributions`
- Grid-alignment helpers for CCDF bound tightening

### `FFT_convolution.py`
Linear-grid FFT convolution:
- `FFT_convolve`
- `FFT_self_convolve`
- direct FFT power path with truncation-aware windowing

### `geometric_convolution.py`
Geometric-grid convolution:
- `geometric_convolve`
- `geometric_self_convolve`

### `convolution_API.py`
Dispatcher API:
- `convolve_discrete_distributions`
- `self_convolve_discrete_distributions`

### `random_allocation_accounting.py`
Main numerical allocation accounting APIs:
- `allocation_PLD`
- `numerical_allocation_epsilon`
- `numerical_allocation_delta`

Internal flow:
1. Build per-round ADD/REMOVE bounds
2. Compose per-round distributions
3. Self-compose across `num_epochs * num_selected`
4. Convert to `dp_accounting` PLD for epsilon/delta queries

### `subsample_PLD.py`
Subsampling amplification utilities:
- `subsample_PLD`
- `subsample_PMF`

### `dp_accounting_support.py`
Bridges internal structures and `dp_accounting` objects:
- `discrete_dist_to_dp_accounting_pmf`
- `dp_accounting_pmf_to_discrete_dist`

## Experiments: `comparisons/experiments`

Scripts for runtime and accuracy comparisons. Some scripts require optional external baselines (`random_allocation`).

Representative files:
- `epsilon_from_sigma.py`
- `delta_comparison.py`
- `runtime.py`
- `grid_size.py`
- `utility_comparison.py`
- `PREAMBLE.py`

## Dependency Sketch

```text
types/core_utils/discrete_dist
         ↑
discretization + convolution backends + utils
         ↑
convolution_API + subsample_PLD + dp_accounting_support
         ↑
random_allocation_accounting
         ↑
comparisons and tests
```

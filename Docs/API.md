# API Reference

This document reflects the current implementation in `PLD_accounting/`.

## Types (`PLD_accounting/types.py`)

### `BoundType`
- `DOMINATES`
- `IS_DOMINATED`
- `BOTH`

### `SpacingType`
- `LINEAR`
- `GEOMETRIC`

### `ConvolutionMethod`
- `GEOM`
- `FFT`
- `COMBINED`
- `BEST_OF_TWO`

### `Direction`
- `ADD`
- `REMOVE`
- `BOTH`

### `PrivacyParams`
```python
@dataclass
class PrivacyParams:
    sigma: float
    num_steps: int
    num_selected: int = 1
    num_epochs: int = 1
    epsilon: float = None
    delta: float = None
```

### `AllocationSchemeConfig`
```python
@dataclass
class AllocationSchemeConfig:
    loss_discretization: float = 1e-2
    tail_truncation: float = 1e-12
    max_grid_FFT: int = 1_000_000
    max_grid_mult: int = -1
    convolution_method: ConvolutionMethod = ConvolutionMethod.GEOM
```

## Distribution Containers

### `DiscreteDist` (`PLD_accounting/discrete_dist.py`)
```python
DiscreteDist(
    x_array: np.ndarray,
    PMF_array: np.ndarray,
    p_neg_inf: float = 0.0,
    p_pos_inf: float = 0.0,
)
```

Methods:
- `validate_mass_conservation(bound_type)`
- `truncate_edges(tail_truncation, bound_type)`
- `copy()`

## Discretization API (`PLD_accounting/distribution_discretization.py`)

### `discretize_continuous_distribution`
```python
def discretize_continuous_distribution(
    dist,
    tail_truncation,
    bound_type,
    spacing_type,
    n_grid,
    align_to_multiples,
) -> DiscreteDist
```

### `discretize_continuous_to_pmf`
```python
def discretize_continuous_to_pmf(
    dist,
    x_array,
    bound_type,
    PMF_min_increment,
) -> DiscreteDist
```

### `discretize_aligned_range`
```python
def discretize_aligned_range(
    x_min,
    x_max,
    spacing_type,
    align_to_multiples,
    discretization=None,
    n_grid=None,
) -> np.ndarray
```

### `change_spacing_type`
```python
def change_spacing_type(
    dist,
    tail_truncation,
    loss_discretization,
    spacing_type,
    bound_type,
) -> DiscreteDist
```

## Convolution API (`PLD_accounting/convolution_API.py`)

### `convolve_discrete_distributions`
```python
def convolve_discrete_distributions(
    dist_1,
    dist_2,
    tail_truncation,
    bound_type,
    convolution_method,
) -> DiscreteDist
```

### `self_convolve_discrete_distributions`
```python
def self_convolve_discrete_distributions(
    dist,
    T,
    tail_truncation,
    bound_type,
    convolution_method,
) -> DiscreteDist
```

Backend functions:
- `PLD_accounting/FFT_convolution.py`: `FFT_convolve`, `FFT_self_convolve`
- `PLD_accounting/geometric_convolution.py`: `geometric_convolve`, `geometric_self_convolve`

## Allocation Accounting API (`PLD_accounting/random_allocation_accounting.py`)

### `allocation_PLD`
```python
def allocation_PLD(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    direction: Direction = Direction.BOTH,
    bound_type: BoundType = BoundType.DOMINATES,
) -> PrivacyLossDistribution
```

Notes:
- Per-round steps use `floor(num_steps / num_selected)`.
- Final self-composition count is `num_epochs * num_selected`.

### `numerical_allocation_epsilon`
```python
def numerical_allocation_epsilon(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    direction: Direction = Direction.BOTH,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float
```
Uses `params.delta`.

### `numerical_allocation_delta`
```python
def numerical_allocation_delta(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    direction: Direction = Direction.BOTH,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float
```
Uses `params.epsilon`.

## Subsampling API (`PLD_accounting/subsample_PLD.py`)

Public functions:
- `subsample_PLD(pld, sampling_probability, bound_type)`
- `subsample_PMF(base_pld, sampling_prob, direction, bound_type)`
- `calc_subsampled_grid(lower_loss, discretization, num_buckets, grid_size, direction)`

## dp-accounting Bridge (`PLD_accounting/dp_accounting_support.py`)

Public functions:
- `discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)`
- `dp_accounting_pmf_to_discrete_dist(pmf)`

## Minimal Example

```python
from PLD_accounting.types import (
    PrivacyParams, AllocationSchemeConfig, Direction, ConvolutionMethod, BoundType
)
from PLD_accounting.random_allocation_accounting import numerical_allocation_epsilon

params = PrivacyParams(sigma=1.0, num_steps=100, num_selected=10, num_epochs=5, delta=1e-5)
config = AllocationSchemeConfig(
    loss_discretization=0.1,
    tail_truncation=1e-3,
    convolution_method=ConvolutionMethod.FFT,
)

epsilon = numerical_allocation_epsilon(
    params=params,
    config=config,
    direction=Direction.REMOVE,
    bound_type=BoundType.DOMINATES,
)
print(epsilon)
```

# User Guide

Practical usage guide for the current implementation.

## Installation

```bash
git clone <repository-url>
cd random_allocation_PLD
pip install numpy scipy numba dp-accounting matplotlib pytest
```

## Core Workflow

1. Create `PrivacyParams`
2. Create `AllocationSchemeConfig`
3. Call one of:
- `numerical_allocation_epsilon`
- `numerical_allocation_delta`
- `allocation_PLD`

## Example: Epsilon for fixed Delta

```python
from PLD_accounting.types import (
    PrivacyParams,
    AllocationSchemeConfig,
    Direction,
    ConvolutionMethod,
    BoundType,
)
from PLD_accounting.random_allocation_accounting import numerical_allocation_epsilon

params = PrivacyParams(
    sigma=1.5,
    num_steps=2000,
    num_selected=250,
    num_epochs=10,
    delta=1e-5,
)
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

## Example: Delta for fixed Epsilon

```python
from PLD_accounting.random_allocation_accounting import numerical_allocation_delta

params.epsilon = 3.0

delta = numerical_allocation_delta(
    params=params,
    config=config,
    direction=Direction.REMOVE,
    bound_type=BoundType.DOMINATES,
)
print(delta)
```

## Example: Get PLD object

```python
from PLD_accounting.random_allocation_accounting import allocation_PLD

pld = allocation_PLD(
    params=params,
    config=config,
    direction=Direction.REMOVE,
    bound_type=BoundType.DOMINATES,
)

print(pld.get_epsilon_for_delta(1e-6))
print(pld.get_delta_for_epsilon(3.0))
```

## Direction and Bound Semantics

- `Direction.REMOVE` and `Direction.ADD` select neighboring relation.
- `Direction.BOTH` computes both (and the wrapper follows `dp_accounting` behavior when queried).
- `BoundType.DOMINATES` gives pessimistic upper bounds.
- `BoundType.IS_DOMINATED` gives optimistic lower bounds.

## Convolution Method Selection

- `ConvolutionMethod.FFT`: linear grid backend, typically fastest for larger compositions.
- `ConvolutionMethod.GEOM`: geometric grid backend.
- `ConvolutionMethod.COMBINED`: REMOVE via FFT, ADD via GEOM.
- `ConvolutionMethod.BEST_OF_TWO`: computes both and combines tighter bounds.

## Parameter Notes

- Internally, per-round steps are `floor(num_steps / num_selected)`.
- Final composition count is `num_epochs * num_selected`.
- Smaller `loss_discretization` and `tail_truncation` usually improve accuracy and increase runtime.

## Subsampling Utilities

For direct subsampling transforms:
- `subsample_PLD`
- `subsample_PMF`

See `PLD_accounting/subsample_PLD.py`.

## Experiments

Run example experiment scripts from repo root:

```bash
python comparisons/experiments/epsilon_from_sigma.py
python comparisons/experiments/runtime.py
```

Some scripts depend on optional external baselines.

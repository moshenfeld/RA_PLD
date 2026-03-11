# PLD_accounting

Numerical privacy accounting for random allocation and subsampling using Privacy Loss Distributions (PLDs).

## What This Repository Provides

- End-to-end numerical accounting for random allocation privacy guarantees.
- Direction-aware accounting (`REMOVE`, `ADD`, or `BOTH`) with explicit upper/lower bound semantics.
- `FFT` convolution for linearly spaced grids.
- `GEOM` convolution for geometrically spaced positive grids.
- Subsampling amplification directly on PLD representations.
- Interop helpers for `dp_accounting` PMF objects.

## Repository Structure

- `PLD_accounting/`: core library code (types, discretization, convolution, accounting, subsampling).
- `tests/`: unit and integration tests.
- `usage_example.py`: executable usage examples for common workflows.

## Installation

### From PyPI

```bash
pip install PLD_accounting
```

### From Source

Clone the repository and install:

```bash
git clone https://github.com/moshenfeld/PLD_accounting.git
cd PLD_accounting
pip install .
```

### Development Installation

For local development with editable installation:

```bash
pip install -e .
```

Install with test extras:

```bash
pip install -e ".[test]"
```

Install with development dependencies:

```bash
pip install ".[dev]"
```

## Requirements

- Python >= 3.10
- numpy >= 1.23
- scipy >= 1.10
- numba >= 0.58
- dp-accounting >= 0.4.3

All dependencies are automatically installed with the package.

## Quick Start

```python
from PLD_accounting import (
    PrivacyParams,
    AllocationSchemeConfig,
    Direction,
    BoundType,
    numerical_allocation_epsilon,
)

params = PrivacyParams(
    sigma=1.0,
    num_steps=1000,
    num_selected=10,
    num_epochs=1,
    delta=1e-6,
)

config = AllocationSchemeConfig(
    loss_discretization=0.02,
    tail_truncation=1e-8,
    max_grid_FFT=1_000_000,
)

eps = numerical_allocation_epsilon(
    params=params,
    config=config,
    direction=Direction.BOTH,
    bound_type=BoundType.DOMINATES,
)
print(eps)
```

For more comprehensive examples including PLD construction, adaptive queries, and subsampling workflows, see [usage_example.py](usage_example.py).

## Main API

- `allocation_PLD(params, config, direction, bound_type)`: returns a `dp_accounting` `PrivacyLossDistribution`.
- `numerical_allocation_epsilon(params, config, direction, bound_type)`: computes `epsilon` for `params.delta`.
- `numerical_allocation_delta(params, config, direction, bound_type)`: computes `delta` for `params.epsilon`.
- `numerical_allocation_epsilon_range(sigma, num_steps, delta, ...)`: adaptively refines resolution and returns `(upper_bound, lower_bound)` for epsilon.
- `numerical_allocation_delta_range(sigma, num_steps, epsilon, ...)`: adaptively refines resolution and returns `(upper_bound, lower_bound)` for delta.
- `subsample_PLD(pld, sampling_probability, bound_type)`: applies subsampling amplification to an existing PLD.

## Important Parameter Notes

- `sigma` must be positive and finite.
- `tail_truncation` must be in `(0, 1)`.
- `loss_discretization` controls the accuracy/runtime tradeoff; smaller values are tighter but slower and larger-memory.
- `max_grid_FFT` and `max_grid_mult` cap grid sizes; too-small budgets can invalidate a run.
- `bound_type=DOMINATES` is pessimistic (upper bound), `IS_DOMINATED` is optimistic (lower bound).
- Adaptive queries treat any negative `target_accuracy` as a convenience default:
  - epsilon query: `target_accuracy = 0.10 *` Poisson-estimated epsilon
  - delta query: `target_accuracy = 0.10 *` Poisson-estimated delta
- Adaptive initialization now starts from a Poisson-subsampled Gaussian guess with:
  - sampling probability `num_selected / num_steps`
  - composition count `num_selected * num_epochs`

## Running Tests

From repository root:

```bash
pytest -q
```

With coverage:

```bash
./tests/run_tests.sh --coverage
```

Build a wheel/sdist:

```bash
python -m build
```

## Examples

See [usage_example.py](usage_example.py) for comprehensive examples including:
- Direct epsilon/delta queries
- PLD construction and repeated lookups
- Adaptive random-allocation queries
- Subsampling and composition workflows

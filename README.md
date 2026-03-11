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

### From TestPyPI (Current)

The package is currently available on TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ PLD_accounting
```

**Note**: The `--extra-index-url` ensures dependencies are installed from the main PyPI.

### From PyPI (After Official Release)

After the official release, you can install directly:

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

Direct import from the dedicated random-allocation API module:

```python
from PLD_accounting.random_allocation_api import (
    allocation_PLD,
    numerical_allocation_epsilon_range,
    numerical_allocation_epsilon,
)
from PLD_accounting.types import AllocationSchemeConfig, BoundType, Direction, PrivacyParams

params = PrivacyParams(
    sigma=3.0,
    num_steps=200,
    num_selected=10,
    delta=1e-6,
)
config = AllocationSchemeConfig(
    loss_discretization=0.05,
    tail_truncation=1e-7,
)

eps = numerical_allocation_epsilon(
    params=params,
    config=config,
    direction=Direction.BOTH,
    bound_type=BoundType.DOMINATES,
)

pld = allocation_PLD(
    params=params,
    config=config,
    direction=Direction.BOTH,
    bound_type=BoundType.DOMINATES,
)

epsilon_upper, epsilon_lower = numerical_allocation_epsilon_range(
    sigma=3.0,
    num_steps=200,
    delta=1e-6,
    num_selected=10,
    epsilon_accuracy=-1,  # any negative value resolves to 10% of the Poisson guess
)
print(epsilon_upper, epsilon_lower)
```

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

- `usage_example.py` includes direct epsilon queries.
- `usage_example.py` includes PLD construction and repeated epsilon lookups.
- `usage_example.py` includes adaptive random-allocation queries through `PLD_accounting.random_allocation_api`.
- `usage_example.py` includes a subsampling + composition workflow.

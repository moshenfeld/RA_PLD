# Implementation Overview

This document explains the conceptual architecture and design approach of PLD_accounting. For installation, API reference, and code examples, see [README.md](README.md).

## Repository Structure

The codebase is organized into focused modules:

### Core Library (`PLD_accounting/`)

- **`types.py`**: Core type definitions (`BoundType`, `Direction`, `PrivacyParams`, etc.)
- **`discrete_dist.py`**: Internal distribution representation with explicit mass tracking
- **`distribution_discretization.py`**: Continuous-to-discrete conversion with domination-aware rounding
- **`convolution_API.py`**: Dispatcher for composition operations
- **`FFT_convolution.py`**: Linear-grid convolution via FFT
- **`geometric_convolution.py`**: Multiplicative-grid convolution for positive supports
- **`random_allocation_accounting.py`**: Core accounting logic for random allocation mechanisms
- **`random_allocation_api.py`**: High-level user-facing API (adaptive queries, epsilon/delta computation)
- **`adaptive_random_allocation.py`**: Adaptive resolution refinement for tight bounds
- **`subsample_PLD.py`**: Subsampling amplification in PLD space
- **`dp_accounting_support.py`**: Interop layer with Google's `dp_accounting` library
- **`core_utils.py`**: Numerical utilities (mass conservation, compensated summation)
- **`utils.py`**: General utilities (binary exponentiation, etc.)

### Testing (`tests/`)

- **`unit/`**: Unit tests for individual functions and modules
- **`integration/`**: End-to-end workflow tests
- **`regression/`**: Backward compatibility tests

### Examples

- **`usage_example.py`**: Executable examples demonstrating common workflows

## Conceptual Architecture

PLD_accounting computes privacy guarantees by representing privacy loss as discrete probability distributions and maintaining rigorous upper/lower bounds throughout all numerical operations. The key insight is that by tracking the full distribution rather than using analytical bounds, we achieve significantly tighter privacy guarantees.

## How It Works

### Privacy Bounds

The library maintains two types of bounds throughout all computations:

- **Upper Bounds (DOMINATES)**: Conservative/pessimistic bounds on privacy loss
- **Lower Bounds (IS_DOMINATED)**: Optimistic bounds on privacy loss

These bounds ensure that the computed privacy guarantees are rigorous and verifiable.

### Privacy Loss Distributions

Privacy loss is represented as discrete probability distributions over a grid of possible loss values. The library:

1. Constructs discrete approximations from continuous random variables
2. Applies privacy-relevant transformations
3. Composes distributions across multiple operations
4. Converts results to standard epsilon-delta privacy parameters

### Numerical Guarantees

The implementation maintains several critical properties:

- **Mass Conservation**: Probability mass is preserved across all operations
- **Tail Semantics**: Explicit tracking of probability mass at infinite endpoints
- **Stable Computation**: Uses compensated summation and log-space arithmetic to maintain numerical stability
- **Bound Consistency**: Ensures upper/lower bound semantics are preserved through all transformations

### Convolution Methods

The library offers multiple convolution strategies with different performance characteristics:

- **GEOM**: Optimized for multiplicative grids and positive supports
- **FFT**: Efficient for linear-grid compositions with many rounds
- **COMBINED/BEST_OF_TWO**: Hybrid approaches that combine methods to achieve tighter bounds

## Privacy Guarantees

The library provides rigorous differential privacy guarantees by:

1. Maintaining explicit bound directions (upper/lower) at every stage
2. Using domination-aware rounding and truncation
3. Preserving tail behavior through explicit infinity-mass tracking
4. Ensuring all numerical operations maintain bound consistency

Each computation stage produces a controlled approximation that remains consistent with the intended privacy guarantee semantics.

## Subsampling

Subsampling is handled directly in privacy loss distribution space using:

- Stable loss transforms for subsample probabilities
- Dual-based construction for proper mixture handling
- Grid remapping with bound-aware rounding

This ensures subsampling integrates seamlessly with the composition machinery.

## API Integration

The library provides a thin compatibility layer with Google's `dp_accounting` library, converting internal `DiscreteDist` representations to standard `PrivacyLossDistribution` objects. This allows users to leverage existing tooling while benefiting from the tighter bounds computed by this library.

For usage examples and API details, see [README.md](README.md).

# Archived Sparsity Implementation Inventory

## Snapshot Date

2026-03-09

## Runtime Modules

- `source_snapshot/PLD_accounting/discrete_dist.py`
  - Structured distribution hierarchy, including the sparse-related types and conversion helpers that existed at the revision point.
- `source_snapshot/PLD_accounting/geometric_convolution.py`
  - Geometric convolution implementation, including the sparse-capable logic under evaluation.
- `source_snapshot/PLD_accounting/random_allocation_accounting.py`
  - Allocation path wiring that exposes the sparse toggle and backend split behavior.
- `source_snapshot/PLD_accounting/distribution_discretization.py`
  - Discretization logic that feeds the structured linear/geometric flow used by the sparse path.
- `source_snapshot/PLD_accounting/dp_accounting_support.py`
  - Adapter layer for the distribution types used in dp-accounting interoperability.
- `source_snapshot/PLD_accounting/types.py`
  - Config and enum definitions, including `use_sparse`.

## Tests

- `source_snapshot/tests/unit/core/test_discrete_types.py`
  - Type-level coverage for structured distributions, conversions, and sparse-related behavior.
- `source_snapshot/tests/unit/core/test_dp_accounting_support.py`
  - Adapter coverage for the redesigned distribution model.
- `source_snapshot/tests/integration/convolution/test_convolution_consistency.py`
  - Sparse-versus-default convolution consistency checks.
- `source_snapshot/tests/integration/convolution/test_random_allocation_backend_split.py`
  - Integration coverage for backend-specific allocation helpers and sparse toggling.
- `source_snapshot/tests/integration/convolution/test_convolution_methods.py`
  - Integration tests covering sparse-specific convolution behavior and FFT rejection of sparse inputs.

## Performance Benchmarks

- `source_snapshot/performance/`
  - Complete performance testing directory with sparse vs dense comparisons
  - `performance/README.md` - Documentation for running performance benchmarks
  - `performance/conftest.py` - Pytest configuration for performance tests
  - `performance/pytest.ini` - Pytest settings for performance test suite
  - `performance/convolution/test_sparse_performance.py` - Benchmark suite measuring sparse vs dense geometric convolution across different sparsity levels, grid sizes, and step counts

## Why This Snapshot Exists

- Future Phase 2 and Phase 3 work will likely reshape the runtime modules.
- Git history alone is a weak working baseline for a deferred feature with multiple interacting files.
- This archive keeps the original sparse implementation, its tests, and its performance harness together in one place for later review.

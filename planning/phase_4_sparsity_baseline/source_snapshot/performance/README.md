# Performance Benchmarks

This directory contains runtime-focused comparisons and benchmarks.

Scope:
- Timing comparisons between alternative implementations
- Large-scale runtime experiments
- Non-functional performance checks

Out of scope:
- Correctness/unit/integration tests required for CI quality gates

Current benchmarks:
- `performance/convolution/test_sparse_performance.py`

Run examples:
- `pytest -s performance/convolution/test_sparse_performance.py`
- `pytest -q performance/convolution/test_sparse_performance.py -k "allocation_api_sparse_vs_dense_10k_grid_1000_steps"`

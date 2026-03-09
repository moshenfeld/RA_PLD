# Phase 1: Linear/Geometric Redesign - Implementation Record

**Status:** Complete - Pure Phase 1 State Achieved
**Dependencies:** None
**Completion Date:** 2026-03-09

## Purpose

This file records the Phase 1 implementation as a pure linear/geometric redesign baseline,
with all Phase 4 (sparsity) code removed from the active runtime and properly archived.

## Implementation Summary

### Core Runtime Areas Updated

- `PLD_accounting/discrete_dist.py`
  - Structured distribution hierarchy with linear/geometric flow
  - Simple concrete classes: `LinearDiscreteDist`, `GeometricDiscreteDist`
  - No "Dense" prefix, no intermediate base classes
  - All Sparse* classes and abstractions removed
- `PLD_accounting/discrete_dist_utils.py`
  - Transform utilities for linear/geometric conversions
  - Sparse-specific functions removed
- `PLD_accounting/distribution_discretization.py`
  - Explicit structured return paths for discretization
- `PLD_accounting/geometric_convolution.py`
  - Dense geometric convolution only
  - All sparse kernels removed
- `PLD_accounting/random_allocation_accounting.py`
  - Structured transform/convolution usage
  - All use_sparse parameters removed
- `PLD_accounting/dp_accounting_support.py`
  - Adapter for dp_accounting compatibility
  - Sparse support removed, SparsePLDPmf inputs are densified
- `PLD_accounting/types.py`
  - Configuration types updated
  - use_sparse parameter removed from AllocationSchemeConfig

## Key Decisions Captured By The Implementation

- Explicit structured types take precedence over backward-compatibility helpers
- Linear and geometric are the active conceptual representations
- Simple class names without "Dense" prefix: `LinearDiscreteDist`, `GeometricDiscreteDist`
- No intermediate abstract base classes (LinearDiscreteDistBase, GeometricDiscreteDistBase)
- Minimal class hierarchy: DiscreteDistBase → concrete implementations
- Sparse-related code completely removed from active runtime
- All sparse functionality archived for potential Phase 4 reintroduction

## Verification

- **Test Suite:** 205 tests passing (2026-03-09)
- **Sparse Removal:** Complete
  - Sparse classes removed from discrete_dist.py
  - Sparse kernels removed from geometric_convolution.py
  - use_sparse parameter removed throughout
  - Sparse-specific tests archived or removed
  - Performance benchmarks (sparse vs dense) archived

## Archive Pointer

Sparse-related implementation material archived in:

- `planning/phase_4_sparsity_baseline/source_snapshot/`
  - Complete source files with sparse implementations
  - Sparse-specific tests
  - Performance benchmarks
  - Design documentation

This archive serves as the baseline for any future Phase 4 sparsity work.

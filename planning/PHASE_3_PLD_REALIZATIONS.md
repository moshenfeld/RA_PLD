# Phase 3: Random Allocation With PLD Realizations

**Dependencies:** Phase 2 complete
**Status:** Not Started

## Overview

Phase 3 adds realization-based allocation support so random allocation can be computed from explicit PLD realizations instead of only from direct Gaussian parameters.

## Objectives

- Implement Theorem 4.4 in a way that reuses the structured distribution work from Phase 1.
- Accept explicit remove/add realizations as API inputs.
- Validate realization invariants early and fail clearly on invalid inputs.

## Scope

### In Scope

- Loss-space to exp-space transforms
- Exp-space regridding and validation
- Direction-specific realization-based allocation helpers
- Public API for remove-only and remove/add realization inputs

### Out Of Scope

- Adaptive-resolution logic beyond consuming the interfaces produced in Phase 2
- Reintroducing sparsity as an active optimization

## Proposed Stages

### Stage 3.1: Design

- Lock the realization input contract and public API shape.
- Map Theorem 4.4 to the existing codebase structure.

### Stage 3.2: Transform And Validation Support

- Implement the realization transforms and regridding helpers.
- Validate invariants such as mass conservation and `E[e^{-L}] <= 1`.

### Stage 3.3: Core Allocation Implementation

- Implement direction-specific REMOVE and ADD realization paths.
- Reuse current composition code where it still fits cleanly.

### Stage 3.4: Integration And Comparison Tests

- Compare against the existing Gaussian-based path.
- Add tests for custom/non-Gaussian realizations.

## Success Criteria

- [ ] Realization-based allocation matches Gaussian-reference behavior within agreed tolerances
- [ ] Public API supports explicit remove/add realization inputs
- [ ] Invalid realizations are rejected clearly
- [ ] New implementation integrates without disturbing the Phase 2 adaptive interface

## Related Files

- `PHASE_3_IMPLEMENTATION.md`
- `DEVELOPMENT_PLAN.md`

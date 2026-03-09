# Phase 1: Linear/Geometric Redesign

**Dependencies:** None
**Status:** Complete

## Overview

Phase 1 is the structured-distribution redesign that separates the linear and geometric representation paths.

The implementation uses simple, direct class names without unnecessary abstraction:
- `LinearDiscreteDist` for evenly-spaced linear grids
- `GeometricDiscreteDist` for ratio-spaced geometric grids
- No "Dense" prefix (sparse support removed)
- No intermediate abstract base classes

## Delivered Scope

- Split the structured distribution flow into linear and geometric representations
- Simple concrete classes: LinearDiscreteDist and GeometricDiscreteDist
- Updated discretization to return explicitly structured outputs
- Added typed transform support needed for later allocation work
- Updated convolution and dp-accounting integration points to use the structured flow
- Clean, minimal class hierarchy without unnecessary abstraction layers

## Out Of Scope For Phase 1

- Sparse convolution performance is not a Phase 1 success criterion.
- Sparse geometric validation/conversion is not part of the active Phase 1 contract.
- Any future sparse reintroduction is handled in Phase 4.

## Completion Criteria

- [x] Structured linear/geometric redesign implemented
- [x] Integration updated across the current runtime path
- [x] Test suite and type checks validated on the redesigned baseline
- [x] Sparse work moved out of active roadmap scope and archived

## Archived Follow-On Material

- Future sparse work starts from `planning/phase_4_sparsity_baseline/`.
- The archived baseline preserves the current sparse design document, source snapshots, tests, and performance benchmark files.

## Phase 1 Work Breakdown

### Stage 1.1: Design

- Define the structured split between linear and geometric distributions.
- Decide that explicit types matter more than backward-compatibility shims.

### Stage 1.2: Core Implementation

- Implement the structured distribution hierarchy and conversions.
- Update discretization and transform helpers to operate on the structured flow.

### Stage 1.3: Integration and Verification

- Update convolution and adapter call sites.
- Re-run tests and type checks on the redesigned baseline.

## Related Files

- `PHASE_1_IMPLEMENTATION.md`
- `DEVELOPMENT_PLAN.md`
- `phase_4_sparsity_baseline/README.md`

# Phase 2: Adaptive Resolution Epsilon/Delta Functions

**Dependencies:** Phase 1 complete
**Status:** Not Started

## Overview

Phase 2 adds adaptive resolution APIs that refine discretization automatically until the upper/lower bound gap is below a target absolute accuracy threshold.

This phase is now the next active implementation target.

## Objectives

- Eliminate most manual tuning of `loss_discretization` and `tail_truncation`.
- Return convergence metadata alongside the computed epsilon or delta value.
- Reuse the existing allocation path instead of introducing a second privacy-accounting stack.

## Scope

### In Scope

- Adaptive epsilon query API
- Adaptive delta query API
- Iterative refinement of discretization and tail truncation
- Best-bound tracking across iterations
- Graceful handling of non-convergence and grid-limit failures

### Out Of Scope

- PLD realization APIs
- Sparse-specific optimization work
- New convolution backends

## Proposed Stages

### Stage 2.1: Design

- Finalize the adaptive loop and stopping rules.
- Define the result object returned to callers.
- Define warnings and failure behavior for non-convergence.

### Stage 2.2: Epsilon Implementation

- Implement adaptive epsilon on top of existing bound computations.
- Validate monotonicity assumptions and best-bound tracking.

### Stage 2.3: Delta Implementation

- Implement adaptive delta with the same convergence machinery.
- Reuse the shared loop and reporting structures where possible.

### Stage 2.4: Integration And Tests

- Add regression coverage.
- Compare adaptive outputs to fixed-resolution reference calculations.
- Document expected default target accuracies.

## Success Criteria

- [ ] Adaptive epsilon converges reliably or returns useful non-converged metadata
- [ ] Adaptive delta converges reliably or returns useful non-converged metadata
- [ ] Results remain consistent with fixed-resolution calculations
- [ ] Public API removes manual resolution tuning for standard use cases

## Related Files

- `PHASE_2_IMPLEMENTATION.md`
- `DEVELOPMENT_PLAN.md`

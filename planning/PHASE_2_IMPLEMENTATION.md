# Phase 2: Adaptive Resolution - Detailed Implementation

**Status:** Ready for implementation
**Dependencies:** Phase 1 complete

## Implementation Shape

### New Runtime Module

- `PLD_accounting/adaptive_accounting.py`

### Main Public APIs

- `adaptive_allocation_epsilon(...)`
- `adaptive_allocation_delta(...)`

### Shared Result Type

- `AdaptiveResult`
  - `value`
  - `upper_bound`
  - `lower_bound`
  - `absolute_gap`
  - `converged`
  - `iterations`
  - `initial_discretization`
  - `final_discretization`
  - `final_tail_truncation`
  - `target_accuracy`

## Core Algorithm

```text
1. Start from an initial discretization and tail truncation.
2. Compute upper and lower bounds with the current resolution.
3. Update best-known upper and lower bounds.
4. Check convergence using absolute gap:
   best_upper - best_lower < target_accuracy
5. If not converged, divide both discretization and tail truncation by 2.
6. Stop on convergence, max iterations, or a runtime/grid failure.
```

## Reused Components

- `allocation_PLD()`
- Existing `BoundType.DOMINATES` and `BoundType.IS_DOMINATED` calculations
- `AllocationSchemeConfig`
- Existing epsilon/delta query methods on the resulting PLD objects

## Design Constraints

- The adaptive layer should not reimplement privacy accounting internals.
- Warnings are preferred over exceptions for non-convergence after partial progress.
- Best-bound tracking should hide minor non-monotonicity caused by numerics.

## Testing Plan

- Unit tests for the result object and convergence logic
- Regression tests against fixed-resolution reference values
- Non-convergence tests for max-iteration and grid-limit exit paths
- Integration coverage for both epsilon and delta entry points

## Exit Criteria

- Adaptive epsilon and delta APIs are both implemented
- Result metadata is stable and documented
- New tests pass with the existing suite

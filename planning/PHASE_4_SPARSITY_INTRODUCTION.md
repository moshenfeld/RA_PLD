# Phase 4: Sparsity Introduction

**Dependencies:** Phase 3 complete
**Status:** Deferred

## Overview

Phase 4 is where sparse distributions and sparse convolution may be reintroduced, but only if the post-Phase-3 codebase still shows a clear performance case that justifies the extra complexity.

This phase does not start from scratch. It starts from the archived material in `planning/phase_4_sparsity_baseline/`.

## Entry Conditions

- Phase 3 is complete and stable.
- The active runtime design around linear/geometric distributions is still coherent.
- Fresh benchmarks show a meaningful opportunity that cannot be reached cheaply by simpler optimizations.

## Baseline Materials

- Archived design document from the original sparse effort
- Source snapshot of the current sparse-related implementation
- Test snapshot covering sparse and dense/sparse equivalence behavior
- Performance benchmark snapshot used during the original evaluation

## Proposed Stages

### Stage 4.1: Baseline Review

- Compare the archived sparse implementation to the then-current runtime.
- Identify what still fits, what must be dropped, and what naming/API changes are required.

### Stage 4.2: Reintroduction Design

- Decide how sparse support layers onto the default linear/geometric model.
- Keep the default user-facing terminology centered on linear and geometric, not on dense naming.

### Stage 4.3: Implementation And Integration

- Reintroduce only the sparse pieces that survive the design review.
- Keep dense/default behavior as the simple fallback path.

### Stage 4.4: Benchmark Gate

- Compare against the default baseline on the target workloads.
- Proceed only if the measured win is large enough to justify the extra code and test surface.

## Success Criteria

- [ ] Sparse support integrates cleanly with the evolved Phase 1-3 architecture
- [ ] Benchmarks show a meaningful net win on target workloads
- [ ] Complexity is justified by measured benefit
- [ ] Default non-sparse behavior remains the simple and well-tested path

## Related Files

- `phase_4_sparsity_baseline/README.md`
- `phase_4_sparsity_baseline/IMPLEMENTATION_INVENTORY.md`

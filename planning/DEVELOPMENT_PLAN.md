# RA_PLD Development Plan

**Branch:** `feature/sparse-convolution-and-adaptive-resolution`
**Status:** Phase 1 complete
**Plan Revision:** 2026-03-09

## Active Roadmap

**Phase 1:** Linear/geometric redesign of the discrete-distribution layer. Complete.
**Phase 2:** Adaptive resolution implementation.
**Phase 3:** PLD realization-based allocation.
**Phase 4:** Sparsity introduction, using the archived baseline under `planning/phase_4_sparsity_baseline/`.

## Revision Decisions

- Sparse support is removed from the active near-term roadmap. The current performance gain does not justify the added implementation and maintenance complexity.
- The linear/geometric distinction remains part of the core design.
- Active planning language treats the non-sparse representations as the default representations. The word `dense` is retained only when referencing the already-implemented baseline code and archived material.
- The current sparse-related implementation state is preserved under `planning/phase_4_sparsity_baseline/` so later phases can evolve the runtime code without losing the Phase 4 starting point.
- The active runtime target is a pure Phase 1 linear/geometric baseline. Archived sparse work belongs under planning artifacts, not in the active allocation/convolution path.
- `SparsePLDPmf` handling in `dp_accounting` interop remains acceptable as an input-boundary densification step; it is not treated as active sparse-distribution support for this roadmap.

## Phase Summary

### Phase 1: Linear/Geometric Redesign

**Objective:** Split the structured distribution flow into linear and geometric representations and make that distinction explicit across discretization, transforms, and convolution entry points.

**Delivered Scope:**
- Structured discrete-distribution hierarchy and explicit linear/geometric flow.
- Typed discretization and transform support needed by later phases.
- Integration updates in convolution and dp-accounting adapters.
- Active runtime aligned to the non-sparse linear/geometric path, with any sparse baseline preserved under `planning/`.

**Deferred From Active Scope:**
- Sparse representations and sparse convolution are no longer Phase 1 acceptance criteria.
- Sparse work completed so far is archived as future Phase 4 baseline material.

**Planning Files:**
- `PHASE_1_LINEAR_GEOMETRIC_REDESIGN.md`
- `PHASE_1_IMPLEMENTATION.md`

### Phase 2: Adaptive Resolution

**Objective:** Add epsilon/delta APIs that refine discretization automatically until upper/lower bounds converge to a target absolute gap.

**Key Components:**
- Iterative refinement loop.
- Best-bound tracking across iterations.
- Convergence metadata and reporting.
- Integration with existing allocation code paths.

**Planning Files:**
- `PHASE_2_ADAPTIVE_RESOLUTION.md`
- `PHASE_2_IMPLEMENTATION.md`

### Phase 3: PLD Realizations

**Objective:** Implement allocation directly from PLD realizations, reusing the structured transform/discretization work from Phase 1 and the adaptive-resolution-aware runtime shape from Phase 2.

**Key Components:**
- Loss-space to exp-space transforms.
- Realization validation and regridding.
- Theorem 4.4 direction-specific composition.
- Public APIs for realization-based allocation.

**Planning Files:**
- `PHASE_3_PLD_REALIZATIONS.md`
- `PHASE_3_IMPLEMENTATION.md`

### Phase 4: Sparsity Introduction

**Objective:** Reintroduce sparsity only if the post-Phase-3 system still shows a compelling performance case.

**Key Components:**
- Review the archived sparse baseline against the then-current runtime.
- Reintroduce sparse linear/geometric representations as optional extensions to the default linear/geometric design.
- Benchmark against the archived dense/default baseline and require a clear net win.
- Keep external-format densification boundaries separate from true runtime sparse support.

**Planning Files:**
- `PHASE_4_SPARSITY_INTRODUCTION.md`
- `phase_4_sparsity_baseline/README.md`

## Execution Order

- Phases execute sequentially: **1 -> 2 -> 3 -> 4**.
- Each phase follows: design/update plan -> implementation -> testing -> review.
- Phase 4 starts only after a fresh performance justification review.

## Testing and Review Strategy

- Keep full regression coverage passing after each phase.
- Treat Phase 2 and Phase 3 as correctness-driven work.
- Treat Phase 4 as benchmark-gated work: it must improve relevant workloads enough to repay complexity.
- Preserve archived Phase 4 baseline snapshots for comparison even as active runtime code changes.

## Success Criteria

### Phase 1
- [x] Linear and geometric representations are split in the active design.
- [x] Existing code paths use the structured representation flow.
- [x] Tests and type checks pass on the redesigned baseline.
- [x] Sparse work removed from active acceptance criteria and archived for later use.

### Phase 2
- [ ] Adaptive epsilon/delta APIs converge reliably or fail gracefully with usable metadata.
- [ ] Results remain consistent with fixed-resolution calculations.
- [ ] Public API is simple enough to remove manual resolution tuning for common use.

### Phase 3
- [ ] Realization-based allocation matches existing Gaussian-based results within agreed tolerances.
- [ ] Remove/add directions are both supported with explicit inputs.
- [ ] Validation catches invalid realizations early.

### Phase 4
- [ ] Archived sparse baseline is reconciled with the current runtime architecture.
- [ ] Sparse mode demonstrates a clear benchmark win on target workloads.
- [ ] Added complexity is justified by measured benefit, not just theoretical savings.

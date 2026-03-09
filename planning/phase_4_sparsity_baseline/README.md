# Phase 4 Sparsity Baseline

## Purpose

This directory preserves the sparse-related implementation state that existed when the roadmap was revised on 2026-03-09.

Sparse support was removed from the active Phase 1-3 plan, but the existing work is kept here so a future Phase 4 can start from a concrete baseline instead of reconstructing it from git history.

## Contents

- `sparse_geometric_design_2026-03-08.md`
  - Original sparse design/planning document from the first Phase 1 plan.
- `source_snapshot/PLD_accounting/`
  - Snapshot of the runtime modules most directly involved in the sparse implementation.
- `source_snapshot/tests/`
  - Snapshot of the tests that exercise the structured distribution layer and sparse-specific integration behavior.
- `source_snapshot/performance/`
  - Snapshot of the sparse-performance benchmark file used during evaluation.
- `IMPLEMENTATION_INVENTORY.md`
  - File-by-file explanation of what was archived and why.

## How To Use This Baseline In Phase 4

1. Start from `IMPLEMENTATION_INVENTORY.md`.
2. Compare the archived source snapshot to the then-current runtime files.
3. Keep only the sparse pieces that still make architectural and benchmark sense.
4. Re-benchmark before reintroducing any sparse path into the active runtime.

## Important Note On Naming

The archived source snapshot preserves the code exactly as it exists at the revision point. That means it still contains implementation names such as `Dense*` and `Sparse*`.

The active roadmap, however, treats linear and geometric as the default representations and reserves sparse naming for any later optional extension work.

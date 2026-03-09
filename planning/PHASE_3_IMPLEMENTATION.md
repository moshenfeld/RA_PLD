# Phase 3: PLD Realization-Based Allocation - Detailed Implementation

**Status:** Ready for implementation
**Dependencies:** Phase 2 complete

## Implementation Shape

### New Runtime Module

- `PLD_accounting/pld_transforms.py`

### Main Internal APIs

- `loss_to_exp_space(...)`
- `exp_to_loss_space(...)`
- `regrid_to_geometric_exp_space(...)`
- `validate_pld_realization(...)`
- `allocation_bound_from_realization(...)`

### Main Public API

- `allocation_PLD_from_realizations(...)`

## Core Theorem 4.4 Flow

```text
REMOVE
1. Validate remove realization
2. Compute dual realization
3. Transform to exp-space
4. Regrid to the canonical geometric grid
5. Run the required convolutions
6. Map back to loss-space and restore infinity/zero atoms

ADD
1. Validate add realization
2. Transform to exp(-L) space
3. Regrid to the canonical geometric grid
4. Run the required convolutions
5. Map back to loss-space and restore infinity/zero atoms
```

## Reused Components

- Structured linear/geometric distribution support from Phase 1
- Existing dual-PLD logic
- Existing convolution/discretization infrastructure where it stays coherent

## Design Constraints

- Realization validation should reject invalid inputs rather than silently repair them.
- Public APIs should require explicit add-direction realizations when add-direction output is requested.
- The implementation should stay independent of sparse-specific assumptions.

## Testing Plan

- Round-trip transform tests
- Invariant validation tests
- Gaussian-reference comparison tests
- Non-Gaussian realization examples

## Exit Criteria

- Realization-based public API implemented
- Validation logic covered by tests
- Numerical comparisons against the existing Gaussian path documented

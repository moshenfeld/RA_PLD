# Sparse Geometric Convolution - Design Document

**Date:** 2026-03-08
**Phase:** 1.1 - Analysis and Design
**Status:** Complete

---

## 1. Current Dense Implementation Analysis

### 1.1 DiscreteDist Structure

**Current Implementation** ([discrete_dist.py](../PLD_accounting/discrete_dist.py)):
- Stores explicit `x_array` and `PMF_array` for all support points
- No distinction between structured grids (linear, geometric) and general grids
- Key methods: `validate_mass_conservation()`, `truncate_edges()`, `copy()`
- Infinite mass handling: `p_neg_inf`, `p_pos_inf`

**Memory Usage:**
- Linear/Geometric grids: ~2× overhead (stores both x_array and PMF_array)
- Sparse geometric: Significant waste storing zeros in PMF_array

### 1.2 Dense Geometric Convolution

**Current Implementation** ([geometric_convolution.py](../PLD_accounting/geometric_convolution.py)):

**Key Assumptions About Density:**
1. **Full Grid Materialization:** x_array contains ALL points of form `x_min * ratio^i` for consecutive i
2. **Array Indexing:** Direct mapping between array indices and geometric indices
3. **Padding Strategy:** Right-padding to equalize array lengths assumes consecutive indices
4. **Kernel Mapping:** `delta_lohi` and `delta_hilo` assume dense output grid
5. **Numba Kernel:** Fixed-size output array assumes dense convolution result

**Algorithm Flow:**
```
1. Validate both inputs have same geometric ratio
2. Swap/pad to equalize lengths (assumes dense grids)
3. Compute shift parameters (delta_lohi, delta_hilo) for rounding
4. Execute Numba kernel with fixed-size output
5. Construct output x_array = x_base * (scale + 1.0)
```

**Performance Characteristics:**
- Time: O(n²) worst case, optimized with Numba
- Space: O(n) for output (assumes dense)
- Sparsity handling: None (zeros explicitly stored)

### 1.3 Grid Utilities

**Current Functions** ([core_utils.py](../PLD_accounting/core_utils.py)):
- `compute_bin_ratio()`: Validates geometric spacing, requires all consecutive points
- `compute_bin_ratio_two_arrays()`: Checks ratio compatibility
- No support for sparse geometric validation or index recovery

---

## 2. Sparse Representation Design

### 2.1 Class Hierarchy Architecture

**Motivation:**
- Current `DiscreteDist` is too general, requiring repeated grid validation
- Memory waste for structured grids (linear/geometric)
- Ambiguous sparse validation without explicit grid metadata

**Proposed Hierarchy:**

```
DiscreteDistBase (ABC)
├── GeneralDiscreteDist (arbitrary x_array)
├── LinearDiscreteDist (x_min, x_gap)
├── DenseGeometricDiscreteDist (x_min, ratio)
└── SparseGeometricDiscreteDist (x_min, ratio, indices)
```

**Design Principles:**
1. **Strict Invariants:** Each subclass enforces its grid structure
2. **Lazy Materialization:** x_array computed on-demand for structured types
3. **Type Safety:** Sparse kernels accept only typed distributions
4. **Backward Compatibility:** `DiscreteDist` aliased to `GeneralDiscreteDist`

### 2.2 SparseGeometricDiscreteDist Specification

**Data Structure:**
```python
class SparseGeometricDiscreteDist:
    x_min: float          # Base value (positive)
    ratio: float          # Geometric ratio (>1)
    indices: NDArray[int64]  # Strictly increasing integer indices
    PMF_array: NDArray[float64]  # Mass at each index
    p_neg_inf: float
    p_pos_inf: float
```

**Invariant:** `x[k] = x_min * ratio^(indices[k])`

**Key Insight:** Indices are **data**, not computed from x_array!

**Sparsity Metric:**
```python
sparsity = 1 - (num_points / (max_index - min_index + 1))
```

**Memory Savings:**
- Dense: stores 2n floats (x_array + PMF_array)
- Sparse (50% sparse): stores n integers + n floats + metadata
- Sparse (90% sparse): stores 0.2n integers + 0.2n floats + metadata

### 2.3 Sparse Geometric Validation

**Challenge:** Given arbitrary x_array, determine if it fits sparse geometric grid.

**Algorithm:**
```
1. Try to fit x_array to model: x[i] ≈ x_min * ratio^(k[i])
2. Search for integer indices k[i] that minimize max log-residual
3. If multiple ratio/offset combinations fit:
   - Deterministic tie-break:
     a. Minimize max log-space residual
     b. Then prefer smaller ratio
     c. Then prefer smaller index offset
4. Accept if max residual ≤ tolerance (1e-10)
```

**Implementation Location:** `core_utils.py`
- `validate_sparse_geometric()`: Returns (is_valid, metadata)
- `try_convert_to_sparse_geometric()`: Converts if valid, else returns unchanged

---

## 3. Sparse Convolution Algorithm

### 3.1 Mathematical Foundation

**Dense Convolution:**
```
For consecutive indices i, j:
Z[i+j] += X[i] * Y[j]
```

**Sparse Convolution:**
```
For sparse indices k1, k2:
Z[k1+k2] += X[k1] * Y[k2]
```

**Key Difference:** Output indices may be non-consecutive, requiring bucketing.

### 3.2 Sparse Kernel Design

**Algorithm:**
```
Input: (x_min_1, ratio, indices_1, PMF_1), (x_min_2, ratio, indices_2, PMF_2)

1. Compute index offset: offset = log(x_min_2/x_min_1) / log(ratio)
2. Align to common reference by shifting indices_2 by offset
3. For each pair (k1, k2):
   - Compute output index: k_out = k1 + k2
   - Accumulate mass: mass_dict[k_out] += PMF_1[k1] * PMF_2[k2]
4. Sort output indices and extract masses
5. Apply Kahan summation for numerical stability
```

**Rounding Strategy for Non-Consecutive Buckets:**
- `DOMINATES`: Round to right (upper bucket) for conservative bounds
- `IS_DOMINATED`: Round to left (lower bucket) for optimistic bounds

**Complexity:**
- Time: O(n1 * n2) for index pairs, O(m log m) for sorting (m ≤ n1 * n2)
- Space: O(m) for output (m typically << n1 * n2 for sparse inputs)

### 3.3 Runtime Contract

**Strict Input Typing:**
- Sparse kernel accepts **only** `SparseGeometricDiscreteDist`
- No ratio/index recovery inside runtime kernel
- API-level wrappers handle conversion/regridding

**Conversion Strategy:**
```
If use_sparse=True (top-level random allocation):
  - Try to convert inputs to SparseGeometricDiscreteDist
  - If successful for both, dispatch to sparse kernel
  - Else, fall back to dense geometric convolution
```

**Dense Conversion Trigger:**
- Auto-convert if sparsity < 20% (density_threshold = 0.8)
- Maintains performance by avoiding sparse overhead on dense data

---

## 4. Integration Design

### 4.1 API Changes

**New Parameter:** `use_sparse: bool`

**Affected Functions:**
1. `convolution_API.py::convolve_distributions()` - Add use_sparse parameter
2. `random_allocation_accounting.py::allocation_PLD()` - Add use_sparse parameter
3. `types.py::AllocationSchemeConfig` - Add use_sparse field (default: True)

**Dispatch Logic:**
```python
if use_sparse:
    dist_1 = try_convert_to_sparse_geometric(dist_1)
    dist_2 = try_convert_to_sparse_geometric(dist_2)
    if both are SparseGeometricDiscreteDist:
        return sparse_geometric_convolve(...)
# Fall back to dense
return geometric_convolve(...)
```

### 4.2 Backward Compatibility

**Alias Strategy:**
```python
# In discrete_dist.py
DiscreteDist = GeneralDiscreteDist
```

**Test Compatibility:**
- All existing tests use `DiscreteDist` constructor
- After alias, tests create `GeneralDiscreteDist` instances
- Behavior identical to current implementation
- Zero breaking changes

**Migration Path:**
- Existing code continues to work unchanged
- New code can opt into sparse via `use_sparse=True`
- Gradual adoption of typed distributions encouraged

---

## 5. Edge Cases and Handling

### 5.1 Sparse Validation Ambiguity

**Problem:** Multiple ratio/offset combinations may fit within tolerance.

**Solution:** Deterministic tie-breaking (see Section 2.3).

### 5.2 Sparse Becomes Dense

**Problem:** After convolution, sparse distribution may become dense.

**Solution:** Auto-convert to `DenseGeometricDiscreteDist` when sparsity < 20%.

### 5.3 Incompatible Grids

**Problem:** Inputs have different ratios.

**Solution:** Raise `ValueError` in sparse kernel (strict validation).

### 5.4 General to Sparse Conversion Failure

**Problem:** Input doesn't fit sparse geometric grid.

**Solution:** `try_convert_to_sparse_geometric()` returns unchanged, falls back to dense.

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Sparse Geometric Validation:**
- Test integer index recovery from x_array
- Test deterministic tie-breaking
- Test tolerance boundaries

**Sparse Convolution Kernel:**
- Mass conservation
- Index correctness (non-consecutive outputs)
- Symmetry properties
- Kahan summation accuracy

### 6.2 Integration Tests

**Numerical Equivalence:**
- Sparse vs dense: epsilon gap ≤ 5%
- CCDF consistency checks
- Round-trip conversion tests

**Performance Benchmarks:**
- >20% speedup when sparsity >50%
- <5% slowdown when dense
- Memory usage verification

### 6.3 Regression Tests

**Backward Compatibility:**
- All existing tests pass with `use_sparse=False`
- DiscreteDist alias works identically

---

## 7. Performance Expectations

### 7.1 Time Complexity

| Operation | Dense | Sparse (50% sparse) | Sparse (90% sparse) |
|-----------|-------|---------------------|---------------------|
| Convolution | O(n²) | O((n/2)²) = O(n²/4) | O((n/10)²) = O(n²/100) |
| Self-conv T times | T × O(n²) | T × O(n²/4) | T × O(n²/100) |

**Expected Speedup:**
- 50% sparse: ~4× faster
- 90% sparse: ~100× faster

### 7.2 Space Complexity

| Grid Type | x_array | PMF_array | indices | Total |
|-----------|---------|-----------|---------|-------|
| Dense | n floats | n floats | - | 2n floats |
| Sparse (50%) | - | n/2 floats | n/2 ints | 0.75n floats equiv |
| Sparse (90%) | - | n/10 floats | n/10 ints | 0.15n floats equiv |

**Memory Savings:**
- 50% sparse: ~60% reduction
- 90% sparse: ~90% reduction

---

## 8. Implementation Files

### New Files (3 test):
1. `tests/unit/convolution/test_sparse_geometric.py` (~200 lines)
2. `tests/integration/convolution/test_sparse_integration.py` (~150 lines)
3. `tests/integration/convolution/test_sparse_performance.py` (~100 lines)

### Modified Files (6):
1. `PLD_accounting/discrete_dist.py` - structured hierarchy, sparse/dense conversion, exp/log helpers
2. `PLD_accounting/core_utils.py` - typed ratio helpers and spacing utilities
3. `PLD_accounting/geometric_convolution.py` - unified dense+sparse geometric kernels
4. `PLD_accounting/convolution_API.py` - sparse dispatch + structured coercion
5. `PLD_accounting/random_allocation_accounting.py` - sparse preference and transform usage
6. `PLD_accounting/types.py` - `use_sparse` config field

---

## 9. Success Criteria (Phase 1)

### Part A: Class Hierarchy Refactoring
- [x] Design complete
- [ ] All existing tests pass with refactoring
- [ ] Memory savings ~50% for structured grids
- [ ] No performance regression
- [ ] Backward compatibility maintained

### Part B: Sparse Geometric Support
- [x] Design complete
- [ ] Sparse convolution correct (within 5% of dense)
- [ ] CCDF/stochastic-order consistency preserved
- [ ] Performance improvement when sparse (>20% speedup at >50% sparsity)
- [ ] Integrates with existing code
- [ ] All tests pass

---

## 10. Risk Assessment

### High Risk
1. **Sparse validation ambiguity** → Mitigated by deterministic tie-break
2. **Breaking existing tests** → Mitigated by backward compatibility alias

### Medium Risk
3. **Performance regression on dense** → Mitigated by auto-conversion
4. **Numba compatibility** → May need pure Python fallback initially

### Low Risk
5. **API confusion** → Mitigated by clear documentation and defaults

---

## 11. Next Steps

1. ✅ Design document complete
2. Implement/maintain hierarchy in `discrete_dist.py`
3. Add backward compatibility alias where needed
4. Keep sparse ownership utilities in `discrete_dist.py` and spacing helpers in `core_utils.py`
5. Run compatibility tests (all existing tests must pass)
6. Proceed to unified sparse+dense geometric convolution implementation (`geometric_convolution.py`)

---

**Design Approved:** Ready for implementation
**Estimated Implementation:** 10 tasks (see Phase 1 todo list)

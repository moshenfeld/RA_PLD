#!/usr/bin/env bash

set -euo pipefail

SUITE="${1:-short}"
PYTEST_FLAGS=("-q" "--tb=short")
SUITE_START=$(date +%s)

# Core integration block intentionally excludes dedicated category files
# (golden/property/matrix/differential) so blocks stay disjoint.
CORE_INTEGRATION_PATHS=(
  tests/integration/convolution
  tests/integration/subsampling
  tests/integration/test_adaptive_resolution.py
  tests/integration/test_distribution_utils.py
  tests/integration/test_edge_cases_and_consistency.py
  tests/integration/test_extreme_parameters.py
  tests/integration/test_general_allocation_composition_semantics.py
  tests/integration/test_old_resolution_gap.py
  tests/integration/test_pld_realizations.py
)

run_step() {
  local name="$1"
  shift

  local start end duration
  echo ""
  echo "==> ${name}"
  start=$(date +%s)
  "$@"
  end=$(date +%s)
  duration=$((end - start))
  echo "<== ${name} (${duration}s)"
}

run_unit() {
  run_step "unit" \
    pytest tests/unit/ "${PYTEST_FLAGS[@]}"
}

run_golden_full() {
  run_step "golden (full)" \
    pytest tests/integration/test_golden_oracles.py "${PYTEST_FLAGS[@]}"
}

run_matrix_fast() {
  run_step "equivalence matrix (fast subset)" \
    pytest tests/integration/test_cross_path_equivalence_matrix.py "${PYTEST_FLAGS[@]}" \
      -m "matrix and not slow and not nightly"
}

run_matrix_full() {
  run_step "equivalence matrix (full)" \
    pytest tests/integration/test_cross_path_equivalence_matrix.py "${PYTEST_FLAGS[@]}" \
      -m "matrix"
}

run_differential_fast() {
  run_step "differential (fast subset)" \
    pytest tests/integration/test_differential_backend_comparison.py "${PYTEST_FLAGS[@]}" \
      -m "differential and not slow and not nightly"
}

run_differential_full() {
  run_step "differential (full)" \
    pytest tests/integration/test_differential_backend_comparison.py "${PYTEST_FLAGS[@]}" \
      -m "differential"
}

run_property_full() {
  run_step "property (full)" \
    pytest tests/integration/test_property_based_invariants.py "${PYTEST_FLAGS[@]}" -m "property"
}

run_core_integration_full() {
  run_step "core integration (full)" \
    pytest "${CORE_INTEGRATION_PATHS[@]}" "${PYTEST_FLAGS[@]}"
}

run_short() {
  run_unit
  run_golden_full
  run_matrix_fast
  run_differential_fast
}

run_medium() {
  run_unit
  run_golden_full
  run_matrix_fast
  run_differential_fast
  run_core_integration_full
}

run_long() {
  run_unit
  run_golden_full
  run_core_integration_full
  run_property_full
  run_matrix_full
  run_differential_full
}

case "${SUITE}" in
  short)
    run_short
    ;;
  medium)
    run_medium
    ;;
  long)
    run_long
    ;;
  *)
    echo "Unknown suite: ${SUITE}"
    echo "Usage: $0 {short|medium|long}"
    exit 2
    ;;
esac

SUITE_END=$(date +%s)
echo ""
echo "Suite '${SUITE}' completed in $((SUITE_END - SUITE_START))s"

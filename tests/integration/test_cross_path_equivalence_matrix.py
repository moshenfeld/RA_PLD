"""
Cross-path equivalence matrix tests.

Tests equivalence between mathematically equivalent paths (Gaussian vs realization)
across comprehensive parameter combinations to prevent semantic drift.

Per Testing Guidelines Section 1: Cross-Path Equivalence Matrix (Mandatory)
"""

from __future__ import annotations

import pytest

# Mark all tests in this module
pytestmark = [pytest.mark.integration, pytest.mark.matrix]

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats

from PLD_accounting import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    PrivacyParams,
    gaussian_allocation_PLD,
    general_allocation_PLD,
)
from PLD_accounting.discrete_dist import LinearDiscreteDist, PLDRealization


@dataclass(frozen=True)
class EquivalenceTestCase:
    """Parameters for a single equivalence test case."""

    sigma: float
    num_steps: int
    num_selected: int
    num_epochs: int
    delta: float
    epsilon_tolerance: float
    description: str
    delta_rel_tolerance: float = 0.15
    nightly: bool = False


@lru_cache(maxsize=None)
def _gaussian_realization(
    sigma: float,
    num_points: int = 2200,
    num_std: float = 8.0,
) -> PLDRealization:
    """Build a reusable Gaussian PLD realization on a linear loss grid."""
    sigma_loss = 1.0 / sigma
    mean = 1.0 / (2.0 * sigma * sigma)
    loss_values = np.linspace(
        mean - num_std * sigma_loss,
        mean + num_std * sigma_loss,
        num_points,
    )
    loss_gap = loss_values[1] - loss_values[0]
    probabilities = stats.norm.pdf(loss_values, loc=mean, scale=sigma_loss) * loss_gap
    probabilities = probabilities / probabilities.sum()
    dist = LinearDiscreteDist.from_x_array(
        x_array=loss_values,
        PMF_array=probabilities,
    )
    return PLDRealization(
        x_min=dist.x_min,
        x_gap=dist.x_gap,
        PMF_array=dist.PMF_array,
        p_loss_inf=dist.p_pos_inf,
        p_loss_neg_inf=dist.p_neg_inf,
    )


def _test_config() -> AllocationSchemeConfig:
    """Standard config for equivalence testing."""
    return AllocationSchemeConfig(
        loss_discretization=5e-3,
        tail_truncation=1e-10,
        max_grid_FFT=200_000,
        max_grid_mult=30_000,
        convolution_method=ConvolutionMethod.GEOM,
    )


# ============================================================================
# Test Matrix: Cross-Path Equivalence
# ============================================================================
#
# Guidelines requirement:
# - num_selected: 1, 2, 5, 10
# - num_steps: divisible and non-divisible by num_selected
# - num_epochs: 1, >1
# - both directions: REMOVE and ADD (tested via same realization for both)
# - both bound types: DOMINATES and IS_DOMINATED
# ============================================================================

EQUIVALENCE_MATRIX = [
    # num_selected = 1
    EquivalenceTestCase(
        sigma=2.0, num_steps=10, num_selected=1, num_epochs=1,
        delta=1e-5, epsilon_tolerance=2e-2,
        description="nsel1-div-ep1"
    ),
    EquivalenceTestCase(
        sigma=2.2, num_steps=11, num_selected=1, num_epochs=1,
        delta=1e-5, epsilon_tolerance=2e-2,
        description="nsel1-nondiv-ep1"
    ),
    EquivalenceTestCase(
        sigma=2.5, num_steps=10, num_selected=1, num_epochs=3,
        delta=1e-5, epsilon_tolerance=6e-2,
        description="nsel1-div-ep3"
    ),
    EquivalenceTestCase(
        sigma=2.3, num_steps=11, num_selected=1, num_epochs=2,
        delta=1e-5, epsilon_tolerance=5e-2,
        description="nsel1-nondiv-ep2"
    ),

    # num_selected = 2
    EquivalenceTestCase(
        sigma=2.0, num_steps=10, num_selected=2, num_epochs=1,
        delta=1e-5, epsilon_tolerance=4e-2,
        delta_rel_tolerance=0.30,
        description="nsel2-div-ep1",
        nightly=True,
    ),
    EquivalenceTestCase(
        sigma=2.2, num_steps=11, num_selected=2, num_epochs=1,
        delta=1e-5, epsilon_tolerance=4e-2,
        delta_rel_tolerance=0.35,
        description="nsel2-nondiv-ep1",
        nightly=True,
    ),
    EquivalenceTestCase(
        sigma=2.0, num_steps=10, num_selected=2, num_epochs=2,
        delta=1e-5, epsilon_tolerance=6e-2,
        delta_rel_tolerance=0.35,
        description="nsel2-div-ep2",
        nightly=True,
    ),
    EquivalenceTestCase(
        sigma=2.2, num_steps=11, num_selected=2, num_epochs=3,
        delta=1e-5, epsilon_tolerance=9e-2,
        delta_rel_tolerance=0.35,
        description="nsel2-nondiv-ep3",
        nightly=True,
    ),

    # num_selected = 5
    EquivalenceTestCase(
        sigma=2.5, num_steps=15, num_selected=5, num_epochs=1,
        delta=1e-6, epsilon_tolerance=8e-2,
        delta_rel_tolerance=0.35,
        description="nsel5-div-ep1",
        nightly=True,
    ),
    EquivalenceTestCase(
        sigma=2.8, num_steps=17, num_selected=5, num_epochs=1,
        delta=1e-6, epsilon_tolerance=8e-2,
        delta_rel_tolerance=0.35,
        description="nsel5-nondiv-ep1",
        nightly=True,
    ),
    EquivalenceTestCase(
        sigma=2.5, num_steps=15, num_selected=5, num_epochs=2,
        delta=1e-6, epsilon_tolerance=1.2e-1,
        delta_rel_tolerance=0.35,
        description="nsel5-div-ep2",
        nightly=True,
    ),
    EquivalenceTestCase(
        sigma=2.8, num_steps=17, num_selected=5, num_epochs=3,
        delta=1e-6, epsilon_tolerance=2.0e-1,
        delta_rel_tolerance=0.35,
        description="nsel5-nondiv-ep3",
        nightly=True,
    ),

    # num_selected = 10
    EquivalenceTestCase(
        sigma=3.0, num_steps=20, num_selected=10, num_epochs=1,
        delta=1e-6, epsilon_tolerance=1.0e-1,
        delta_rel_tolerance=0.35,
        description="nsel10-div-ep1",
        nightly=True,
    ),
    EquivalenceTestCase(
        sigma=3.2, num_steps=23, num_selected=10, num_epochs=1,
        delta=1e-6, epsilon_tolerance=1.0e-1,
        delta_rel_tolerance=0.35,
        description="nsel10-nondiv-ep1",
        nightly=True,
    ),
    EquivalenceTestCase(
        sigma=3.0, num_steps=20, num_selected=10, num_epochs=2,
        delta=1e-6, epsilon_tolerance=2.0e-1,
        delta_rel_tolerance=0.35,
        description="nsel10-div-ep2",
        nightly=True,
    ),
    EquivalenceTestCase(
        sigma=3.2, num_steps=23, num_selected=10, num_epochs=3,
        delta=1e-6, epsilon_tolerance=3.0e-1,
        delta_rel_tolerance=0.35,
        description="nsel10-nondiv-ep3",
        nightly=True,
    ),
]


def _matrix_param(case: EquivalenceTestCase):
    marks = [pytest.mark.nightly, pytest.mark.slow] if case.nightly else []
    return pytest.param(case, marks=marks, id=case.description)


EQUIVALENCE_CASES = [_matrix_param(case) for case in EQUIVALENCE_MATRIX]


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
@pytest.mark.parametrize("test_case", EQUIVALENCE_CASES)
class TestCrossPathEquivalenceMatrix:
    """Comprehensive cross-path equivalence tests across parameter matrix.

    Tests that Gaussian path and realization path produce equivalent results
    across all combinations of:
    - num_selected: 1, 2, 5, 10
    - num_steps: divisible and non-divisible by num_selected
    - num_epochs: 1, >1
    - bound_type: DOMINATES, IS_DOMINATED
    """

    def test_gaussian_vs_realization_equivalence(
        self,
        test_case: EquivalenceTestCase,
        bound_type: BoundType,
    ):
        """Test Gaussian and realization paths produce equivalent epsilon."""
        config = _test_config()

        # Gaussian path
        params = PrivacyParams(
            sigma=test_case.sigma,
            num_steps=test_case.num_steps,
            num_selected=test_case.num_selected,
            num_epochs=test_case.num_epochs,
            delta=test_case.delta,
        )
        gaussian_pld = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=bound_type,
        )

        # Realization path
        realization = _gaussian_realization(test_case.sigma)
        realization_pld = general_allocation_PLD(
            num_steps=test_case.num_steps,
            num_selected=test_case.num_selected,
            num_epochs=test_case.num_epochs,
            remove_realization=realization,
            add_realization=realization,
            config=config,
            bound_type=bound_type,
        )

        # Compare epsilon values
        epsilon_gaussian = float(gaussian_pld.get_epsilon_for_delta(test_case.delta))
        epsilon_realization = float(realization_pld.get_epsilon_for_delta(test_case.delta))

        # Validate results are finite
        assert np.isfinite(epsilon_gaussian), \
            f"Gaussian path epsilon is not finite: {epsilon_gaussian}"
        assert np.isfinite(epsilon_realization), \
            f"Realization path epsilon is not finite: {epsilon_realization}"

        # Check equivalence within tolerance
        epsilon_diff = abs(epsilon_gaussian - epsilon_realization)
        assert epsilon_diff < test_case.epsilon_tolerance, (
            f"Paths diverged beyond tolerance:\n"
            f"  Gaussian epsilon:     {epsilon_gaussian:.6f}\n"
            f"  Realization epsilon:  {epsilon_realization:.6f}\n"
            f"  Difference:           {epsilon_diff:.6f}\n"
            f"  Tolerance:            {test_case.epsilon_tolerance:.6f}\n"
            f"  Parameters: sigma={test_case.sigma}, num_steps={test_case.num_steps}, "
            f"num_selected={test_case.num_selected}, num_epochs={test_case.num_epochs}, "
            f"bound_type={bound_type.name}"
        )

    def test_delta_equivalence(
        self,
        test_case: EquivalenceTestCase,
        bound_type: BoundType,
    ):
        """Test Gaussian and realization paths produce equivalent delta."""
        config = _test_config()
        epsilon_query = 1.0  # Fixed epsilon for delta query

        # Gaussian path
        params = PrivacyParams(
            sigma=test_case.sigma,
            num_steps=test_case.num_steps,
            num_selected=test_case.num_selected,
            num_epochs=test_case.num_epochs,
            epsilon=epsilon_query,
        )
        gaussian_pld = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=bound_type,
        )

        # Realization path
        realization = _gaussian_realization(test_case.sigma)
        realization_pld = general_allocation_PLD(
            num_steps=test_case.num_steps,
            num_selected=test_case.num_selected,
            num_epochs=test_case.num_epochs,
            remove_realization=realization,
            add_realization=realization,
            config=config,
            bound_type=bound_type,
        )

        # Compare delta values
        delta_gaussian = float(gaussian_pld.get_delta_for_epsilon(epsilon_query))
        delta_realization = float(realization_pld.get_delta_for_epsilon(epsilon_query))

        # Validate results are in valid range
        assert 0.0 <= delta_gaussian <= 1.0, \
            f"Gaussian path delta out of range: {delta_gaussian}"
        assert 0.0 <= delta_realization <= 1.0, \
            f"Realization path delta out of range: {delta_realization}"

        # Check equivalence (use relative tolerance for delta)
        # Delta can be very small, so use max of absolute and relative tolerance
        abs_tol = test_case.delta * 0.15
        rel_tol = test_case.delta_rel_tolerance

        if delta_gaussian > abs_tol and delta_realization > abs_tol:
            # Both non-trivial, use relative tolerance
            rel_diff = abs(delta_gaussian - delta_realization) / max(delta_gaussian, delta_realization)
            assert rel_diff < rel_tol, (
                f"Delta paths diverged beyond relative tolerance:\n"
                f"  Gaussian delta:      {delta_gaussian:.2e}\n"
                f"  Realization delta:   {delta_realization:.2e}\n"
                f"  Relative difference: {rel_diff:.2%}\n"
                f"  Tolerance:           {rel_tol:.2%}"
            )
        else:
            # At least one is very small, use absolute tolerance
            abs_diff = abs(delta_gaussian - delta_realization)
            assert abs_diff < abs_tol, (
                f"Delta paths diverged beyond absolute tolerance:\n"
                f"  Gaussian delta:      {delta_gaussian:.2e}\n"
                f"  Realization delta:   {delta_realization:.2e}\n"
                f"  Absolute difference: {abs_diff:.2e}\n"
                f"  Tolerance:           {abs_tol:.2e}"
            )


# ============================================================================
# Regression Tests: Specific known issues
# ============================================================================

class TestEquivalenceRegressions:
    """Regression tests for specific semantic drift bugs caught by matrix."""
    pytestmark = [pytest.mark.regression]

    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_remainder_handling_uses_all_steps(self, bound_type: BoundType):
        """Adaptive allocation uses all steps including remainder.

        When num_steps is not evenly divisible by num_selected, the adaptive
        allocation uses a mixture of floor and ceil distributions to ensure
        all steps are accounted for.

        num_steps=6, num_selected=2: divisible, uses only dist(3)
        num_steps=7, num_selected=2: non-divisible, uses mix of dist(3) and dist(4)

        These should give DIFFERENT results since num_steps=7 uses an extra step.
        """
        config = _test_config()
        realization = _gaussian_realization(2.0)
        delta = 1e-5

        # num_steps=6, num_selected=2 -> floor(6/2)=3, remainder=0 (divisible)
        pld_6 = general_allocation_PLD(
            num_steps=6,
            num_selected=2,
            num_epochs=2,
            remove_realization=realization,
            add_realization=realization,
            config=config,
            bound_type=bound_type,
        )

        # num_steps=7, num_selected=2 -> floor(7/2)=3, remainder=1 (non-divisible)
        pld_7 = general_allocation_PLD(
            num_steps=7,
            num_selected=2,
            num_epochs=2,
            remove_realization=realization,
            add_realization=realization,
            config=config,
            bound_type=bound_type,
        )

        epsilon_6 = float(pld_6.get_epsilon_for_delta(delta))
        epsilon_7 = float(pld_7.get_epsilon_for_delta(delta))

        # Should be DIFFERENT since num_steps=7 uses remainder distribution
        # (num_steps=6 is divisible, num_steps=7 uses mix of floor and ceil)
        assert abs(epsilon_6 - epsilon_7) > 1e-6, (
            f"Expected different epsilon values due to remainder handling:\n"
            f"  epsilon(num_steps=6): {epsilon_6}\n"
            f"  epsilon(num_steps=7): {epsilon_7}\n"
            f"  num_steps=7 should use extra step via remainder distribution"
        )

    def test_num_epochs_formula_stable(self):
        """Regression: new_num_epochs = num_selected * num_epochs must be stable."""
        config = _test_config()
        realization = _gaussian_realization(2.0)
        delta = 1e-5

        # Test that composition count formula is correct
        # num_selected=3, num_epochs=4 -> new_num_epochs=12
        pld = general_allocation_PLD(
            num_steps=15,  # floor(15/3)=5 inner compositions
            num_selected=3,
            num_epochs=4,
            remove_realization=realization,
            add_realization=realization,
            config=config,
            bound_type=BoundType.DOMINATES,
        )

        epsilon = float(pld.get_epsilon_for_delta(delta))
        assert np.isfinite(epsilon) and epsilon > 0.0, \
            "Expected finite positive epsilon from 12-round composition"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

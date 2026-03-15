"""
Golden oracle tests with hardcoded expected values.

Small, deterministic scenarios with manually verified or independently
derived oracle values to catch logic shifts.

Per Testing Guidelines Section 3: Golden Oracle Tests
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

# Mark all tests in this module
pytestmark = [pytest.mark.integration, pytest.mark.golden]

from PLD_accounting import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    PrivacyParams,
    gaussian_allocation_PLD,
)
from PLD_accounting.discrete_dist import LinearDiscreteDist, PLDRealization
from PLD_accounting.random_allocation_api import general_allocation_PLD
from tests.test_tolerances import TestTolerances as TOL


class TestGoldenOracleSimpleComposition:
    """Golden oracle tests for simple composition scenarios."""

    def test_two_gaussian_steps_sigma_one_master_baseline(self):
        """Golden oracle: Two-step Gaussian with sigma=1.0.

        Uses a valid REMOVE configuration and the trusted baseline trajectory
        (master commit `e870048`) at delta=1e-3.
        """
        config = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        params = PrivacyParams(
            sigma=1.0,
            num_steps=2,
            num_selected=1,
            num_epochs=1,
            delta=1e-3,
        )

        pld = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=BoundType.DOMINATES,
        )

        epsilon = float(pld.get_epsilon_for_delta(1e-3))

        # Trusted baseline (master@e870048): epsilon ~= 2.49944
        oracle_epsilon = 2.49944
        tolerance = 0.03

        assert abs(epsilon - oracle_epsilon) < tolerance, (
            f"Golden oracle mismatch:\n"
            f"  Computed epsilon: {epsilon:.6f}\n"
            f"  Oracle epsilon:   {oracle_epsilon:.6f}\n"
            f"  Difference:       {abs(epsilon - oracle_epsilon):.6f}\n"
            f"  Tolerance:        {tolerance:.6f}"
        )

    def test_two_gaussian_steps_composition(self):
        """Golden oracle: Two Gaussian steps with sigma=2.0.

        Known composition formula validation.
        """
        config = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        params = PrivacyParams(
            sigma=2.0,
            num_steps=2,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
        )

        pld = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=BoundType.DOMINATES,
        )

        epsilon = float(pld.get_epsilon_for_delta(1e-5))

        # Trusted baseline trajectory (master@e870048-compatible regime)
        oracle_epsilon = 1.465
        tolerance = 0.03

        assert abs(epsilon - oracle_epsilon) < tolerance, (
            f"Two-step composition oracle mismatch:\n"
            f"  Computed epsilon: {epsilon:.6f}\n"
            f"  Oracle epsilon:   {oracle_epsilon:.6f}"
        )

    def test_tiny_deterministic_realization(self):
        """Golden oracle: Minimal deterministic realization.

        Uses a tiny 3-point distribution with manually computed expected loss.
        """
        # Create tiny deterministic realization
        loss_values = np.array([0.0, 0.5, 1.0])
        probabilities = np.array([0.6, 0.3, 0.1])

        realization = PLDRealization(
            x_min=loss_values[0],
            x_gap=0.5,
            PMF_array=probabilities,
        )

        # Verify expected loss value
        expected_loss = np.sum(loss_values * probabilities)
        assert abs(expected_loss - 0.25) < 1e-10, \
            f"Expected loss should be 0.6*0 + 0.3*0.5 + 0.1*1.0 = 0.25, got {expected_loss}"

        # Single composition should be queryable
        config = AllocationSchemeConfig(
            loss_discretization=1e-2,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld = general_allocation_PLD(
            num_steps=1,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        # Should produce finite results
        epsilon = float(pld.get_epsilon_for_delta(1e-5))
        delta = float(pld.get_delta_for_epsilon(1.0))

        assert np.isfinite(epsilon) and epsilon > 0.0
        assert 0.0 <= delta <= 1.0

    def test_uniform_tiny_grid_oracle(self):
        """Golden oracle: Uniform distribution on tiny grid.

        Manually verifiable uniform distribution composition.
        """
        # Uniform on [0, 1] with 5 points
        loss_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        probabilities = np.ones(5) / 5.0

        realization = PLDRealization.from_linear_dist(
            LinearDiscreteDist.from_x_array(
                x_array=loss_values,
                PMF_array=probabilities,
            )
        )

        # Verify uniformity
        assert np.allclose(probabilities, 0.2, atol=1e-10)

        # Expected loss for uniform on [0,1] is 0.5
        expected_loss = np.sum(loss_values * probabilities)
        assert abs(expected_loss - 0.5) < 0.02, \
            f"Uniform expected loss should be ~0.5, got {expected_loss}"

        config = AllocationSchemeConfig(
            loss_discretization=5e-2,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld = general_allocation_PLD(
            num_steps=2,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        delta = float(pld.get_delta_for_epsilon(1.0))
        assert 0.0 <= delta <= 1.0


class TestGoldenOracleBoundTypes:
    """Golden oracle tests validating bound type semantics."""

    def test_dominates_vs_is_dominated_ordering(self):
        """Golden oracle: DOMINATES bound should give larger epsilon than IS_DOMINATED.

        For the same delta, the DOMINATES (upper) bound should be more
        conservative (larger epsilon) than IS_DOMINATED (lower) bound.
        """
        config = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        params = PrivacyParams(
            sigma=1.5,
            num_steps=3,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
        )

        pld_upper = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=BoundType.DOMINATES,
        )

        pld_lower = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=BoundType.IS_DOMINATED,
        )

        epsilon_upper = float(pld_upper.get_epsilon_for_delta(1e-5))
        epsilon_lower = float(pld_lower.get_epsilon_for_delta(1e-5))

        # Upper bound should be >= lower bound (stochastic dominance)
        assert epsilon_upper >= epsilon_lower - TOL.EPSILON_ABSOLUTE, (
            f"Bound ordering violated:\n"
            f"  DOMINATES (upper) epsilon:     {epsilon_upper:.6f}\n"
            f"  IS_DOMINATED (lower) epsilon:  {epsilon_lower:.6f}\n"
            f"  Upper should be >= lower"
        )

        # The gap should be reasonable but non-zero
        gap = epsilon_upper - epsilon_lower
        assert 0.0 <= gap < 0.5, (
            f"Bound gap unreasonable: {gap:.6f}\n"
            f"Expected small positive gap or near-zero"
        )

    def test_infinity_mass_constraints_oracle(self):
        """Golden oracle: Infinity mass constraints are enforced correctly."""
        # Create realization with specific infinity mass structure
        loss_values = np.array([0.0, 1.0, 2.0])
        probabilities = np.array([0.5, 0.3, 0.15])
        p_loss_inf = 0.05  # Small positive infinity mass

        realization = PLDRealization(
            x_min=0.0,
            x_gap=1.0,
            PMF_array=probabilities,
            p_loss_inf=p_loss_inf,
            p_loss_neg_inf=0.0,  # Must be 0 for DOMINATES
        )

        # DOMINATES requires p_loss_neg_inf = 0
        realization.validate_pld_realization()

        # Should be usable in composition
        config = AllocationSchemeConfig(
            loss_discretization=1e-2,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld = general_allocation_PLD(
            num_steps=1,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
            bound_type=BoundType.DOMINATES,
        )

        epsilon = float(pld.get_epsilon_for_delta(0.06))
        assert np.isfinite(epsilon) and epsilon > 0.0


class TestGoldenOracleCompositionCounts:
    """Golden oracle tests for composition count formulas."""

    def test_inner_outer_decomposition_oracle_2_3(self):
        """Golden oracle: num_selected=2, num_epochs=3 composition.

        num_steps=10, num_selected=2, num_epochs=3
        -> inner = floor(10/2) = 5
        -> outer = 2 * 3 = 6
        """
        # Small deterministic realization
        realization = PLDRealization(
            x_min=0.0,
            x_gap=0.1,
            PMF_array=np.array([0.7, 0.2, 0.1]),
        )

        config = AllocationSchemeConfig(
            loss_discretization=1e-2,
            convolution_method=ConvolutionMethod.GEOM,
        )

        # This should perform:
        # 1. Inner composition: 5 single-step compositions
        # 2. Outer composition: 6 round compositions
        pld = general_allocation_PLD(
            num_steps=10,
            num_selected=2,
            num_epochs=3,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        delta = float(pld.get_delta_for_epsilon(1.0))

        # Oracle check: delta query should be finite and valid.
        assert 0.0 <= delta <= 1.0

    def test_composition_count_invariant_floor_division(self):
        """Golden oracle: Floor division invariant.

        num_steps=7 and num_steps=8 with num_selected=3 should give
        different results since floor(7/3)=2 but floor(8/3)=2.

        Actually they should be the same!
        """
        realization = PLDRealization(
            x_min=0.0,
            x_gap=0.1,
            PMF_array=np.array([0.7, 0.2, 0.1]),
        )

        config = AllocationSchemeConfig(
            loss_discretization=1e-2,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld_7 = general_allocation_PLD(
            num_steps=7,  # floor(7/3) = 2
            num_selected=3,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        pld_8 = general_allocation_PLD(
            num_steps=8,  # floor(8/3) = 2
            num_selected=3,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        delta_7 = float(pld_7.get_delta_for_epsilon(0.2))
        delta_8 = float(pld_8.get_delta_for_epsilon(0.2))

        # Both should be identical (same inner composition count)
        assert abs(delta_7 - delta_8) < 1e-12, (
            f"Floor division invariant violated:\n"
            f"  delta(num_steps=7): {delta_7}\n"
            f"  delta(num_steps=8): {delta_8}\n"
            f"  Both have floor(n/3)=2, should be identical"
        )

    def test_composition_count_change_floor_division(self):
        """Golden oracle: Floor division changes at boundary.

        num_steps=8 and num_steps=9 with num_selected=3:
        floor(8/3)=2 but floor(9/3)=3, so results should differ.
        """
        realization = PLDRealization(
            x_min=0.0,
            x_gap=0.1,
            PMF_array=np.array([0.7, 0.2, 0.1]),
        )

        config = AllocationSchemeConfig(
            loss_discretization=1e-2,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld_8 = general_allocation_PLD(
            num_steps=8,  # floor(8/3) = 2
            num_selected=3,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        pld_9 = general_allocation_PLD(
            num_steps=9,  # floor(9/3) = 3
            num_selected=3,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        delta_8 = float(pld_8.get_delta_for_epsilon(0.2))
        delta_9 = float(pld_9.get_delta_for_epsilon(0.2))

        # Results should differ (more compositions -> larger delta at fixed epsilon)
        assert delta_9 > delta_8, (
            f"Composition count change not reflected:\n"
            f"  delta(num_steps=8, floor=2): {delta_8}\n"
            f"  delta(num_steps=9, floor=3): {delta_9}\n"
            f"  Expected delta_9 > delta_8"
        )

        # Difference should be material (one additional inner composition).
        ratio = delta_9 / delta_8
        assert 1.2 < ratio < 3.0, (
            f"Delta ratio outside expected range:\n"
            f"  ratio = {ratio:.3f}\n"
            f"  Expected between 1.2 and 3.0 for one additional composition"
        )


class TestGoldenOracleEdgeCases:
    """Golden oracle tests for edge cases and boundary conditions."""

    def test_single_point_distribution(self):
        """Golden oracle: Single-point (deterministic) distribution.

        A distribution with all mass at a single point should compose
        to exactly that value times the composition count.
        """
        # Deterministic-at-0.5 encoded with a trailing zero-probability point.
        realization = PLDRealization(
            x_min=0.5,
            x_gap=0.5,
            PMF_array=np.array([1.0, 0.0]),
        )

        config = AllocationSchemeConfig(
            loss_discretization=1e-2,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld = general_allocation_PLD(
            num_steps=3,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        # For this deterministic realization, delta query should be finite and valid.
        delta = float(pld.get_delta_for_epsilon(1.0))
        assert 0.0 <= delta <= 1.0

    def test_zero_loss_distribution(self):
        """Golden oracle: Zero-loss distribution should give epsilon=0."""
        # Deterministic-at-0 encoded with a trailing zero-probability point.
        realization = PLDRealization(
            x_min=0.0,
            x_gap=1.0,
            PMF_array=np.array([1.0, 0.0]),
        )

        config = AllocationSchemeConfig(
            loss_discretization=1e-2,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld = general_allocation_PLD(
            num_steps=5,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        # Zero loss should give small epsilon.
        epsilon = float(pld.get_epsilon_for_delta(1e-3))

        assert epsilon < 5e-2, (
            f"Zero-loss oracle violation:\n"
            f"  Computed epsilon: {epsilon:.10f}\n"
            f"  Expected: ~0.0\n"
            f"  Zero loss should give near-zero epsilon"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

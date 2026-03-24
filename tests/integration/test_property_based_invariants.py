"""
Property-based tests for structural invariants.

Tests invariants that must hold regardless of specific parameter values:
- Monotonicity in composition count
- Monotonicity under tighter bounds
- Stability under re-discretization

Per Testing Guidelines Section 4: Property-Based Tests
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

# Mark all tests in this module
pytestmark = [pytest.mark.integration, pytest.mark.property]

from PLD_accounting import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    PrivacyParams,
    gaussian_allocation_PLD,
    general_allocation_PLD,
)
from PLD_accounting.discrete_dist import LinearDiscreteDist, PLDRealization


def _gaussian_realization(sigma: float, num_points: int = 1200) -> PLDRealization:
    """Build a Gaussian PLD realization."""
    sigma_loss = 1.0 / sigma
    mean = 1.0 / (2.0 * sigma * sigma)
    loss_values = np.linspace(
        mean - 8.0 * sigma_loss,
        mean + 8.0 * sigma_loss,
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


def _default_config() -> AllocationSchemeConfig:
    """Default config for property tests."""
    return AllocationSchemeConfig(
        loss_discretization=1e-2,
        tail_truncation=1e-10,
        convolution_method=ConvolutionMethod.GEOM,
    )


class TestMonotonicityInCompositionCount:
    """Test monotonicity under random-allocation semantics.

    In this package, ``num_steps`` corresponds to total candidate steps ``t``.
    Increasing ``t`` (with fixed ``k``) amplifies privacy, so epsilon/delta
    decrease as ``num_steps`` grows.
    """

    @pytest.mark.parametrize("sigma", [2.0, 3.0])
    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_epsilon_monotonic_in_num_steps(self, sigma: float, bound_type: BoundType):
        """Property: epsilon decreases with num_steps."""
        config = _default_config()
        delta = 1e-5

        num_steps_values = [2, 4, 8, 16]
        epsilons = []

        for num_steps in num_steps_values:
            params = PrivacyParams(
                sigma=sigma,
                num_steps=num_steps,
                num_selected=1,
                num_epochs=1,
                delta=delta,
            )
            pld = gaussian_allocation_PLD(params=params, config=config, bound_type=bound_type)
            epsilon = float(pld.get_epsilon_for_delta(delta))
            epsilons.append(epsilon)

        # Check monotonicity
        for i in range(len(epsilons) - 1):
            assert epsilons[i] > epsilons[i + 1], (
                f"Epsilon monotonicity violated:\n"
                f"  num_steps={num_steps_values[i]}: epsilon={epsilons[i]:.6f}\n"
                f"  num_steps={num_steps_values[i+1]}: epsilon={epsilons[i+1]:.6f}\n"
                f"  Expected epsilon to decrease with num_steps"
            )

    @pytest.mark.parametrize("sigma", [2.0])
    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_epsilon_monotonic_in_num_epochs(self, sigma: float, bound_type: BoundType):
        """Property: epsilon increases with num_epochs."""
        config = _default_config()
        delta = 1e-5

        num_epochs_values = [1, 2, 3, 5]
        epsilons = []

        for num_epochs in num_epochs_values:
            params = PrivacyParams(
                sigma=sigma,
                num_steps=10,
                num_selected=2,
                num_epochs=num_epochs,
                delta=delta,
            )
            pld = gaussian_allocation_PLD(params=params, config=config, bound_type=bound_type)
            epsilon = float(pld.get_epsilon_for_delta(delta))
            epsilons.append(epsilon)

        # Check monotonicity
        for i in range(len(epsilons) - 1):
            assert epsilons[i] < epsilons[i + 1], (
                f"Epsilon monotonicity in epochs violated:\n"
                f"  num_epochs={num_epochs_values[i]}: epsilon={epsilons[i]:.6f}\n"
                f"  num_epochs={num_epochs_values[i+1]}: epsilon={epsilons[i+1]:.6f}\n"
                f"  Expected epsilon to increase with num_epochs"
            )

    @pytest.mark.parametrize("sigma", [2.0])
    def test_delta_monotonic_in_num_steps(self, sigma: float):
        """Property: delta decreases with num_steps (for fixed epsilon)."""
        config = _default_config()
        epsilon = 1.0

        num_steps_values = [2, 4, 8, 16]
        deltas = []

        for num_steps in num_steps_values:
            params = PrivacyParams(
                sigma=sigma,
                num_steps=num_steps,
                num_selected=1,
                num_epochs=1,
                epsilon=epsilon,
            )
            pld = gaussian_allocation_PLD(params=params, config=config)
            delta = float(pld.get_delta_for_epsilon(epsilon))
            deltas.append(delta)

        # Check monotonicity
        for i in range(len(deltas) - 1):
            assert deltas[i] > deltas[i + 1], (
                f"Delta monotonicity violated:\n"
                f"  num_steps={num_steps_values[i]}: delta={deltas[i]:.2e}\n"
                f"  num_steps={num_steps_values[i+1]}: delta={deltas[i+1]:.2e}\n"
                f"  Expected delta to decrease with num_steps"
            )

    @pytest.mark.parametrize("num_selected", [2])
    def test_epsilon_monotonic_in_num_selected(self, num_selected: int):
        """Property: epsilon decreases as ``num_steps`` grows for fixed ``num_selected``.

        This checks that the two-stage composition remains monotone in ``t``.
        """
        config = _default_config()
        sigma = 2.0
        delta = 1e-5

        # Fix total composition count by adjusting num_steps
        # E.g., for num_selected=2, num_epochs=3: inner=floor(num_steps/2), outer=6
        # We'll test increasing num_steps while holding num_selected and num_epochs fixed

        num_steps_values = [num_selected * 2, num_selected * 4, num_selected * 8]
        epsilons = []

        for num_steps in num_steps_values:
            params = PrivacyParams(
                sigma=sigma,
                num_steps=num_steps,
                num_selected=num_selected,
                num_epochs=1,
                delta=delta,
            )
            pld = gaussian_allocation_PLD(params=params, config=config)
            epsilon = float(pld.get_epsilon_for_delta(delta))
            epsilons.append(epsilon)

        # Check monotonicity
        for i in range(len(epsilons) - 1):
            assert epsilons[i] > epsilons[i + 1], (
                f"Epsilon monotonicity with num_selected={num_selected} violated:\n"
                f"  num_steps={num_steps_values[i]}: epsilon={epsilons[i]:.6f}\n"
                f"  num_steps={num_steps_values[i+1]}: epsilon={epsilons[i+1]:.6f}"
            )


class TestMonotonicityInPrivacyParameters:
    """Test monotonicity in privacy parameters (sigma, delta)."""

    @pytest.mark.parametrize("num_steps", [5, 10])
    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_epsilon_decreases_with_sigma(self, num_steps: int, bound_type: BoundType):
        """Property: epsilon decreases as sigma increases (stronger privacy)."""
        config = _default_config()
        delta = 1e-5

        sigma_values = [1.0, 1.5, 2.0, 3.0, 5.0]
        epsilons = []

        for sigma in sigma_values:
            params = PrivacyParams(
                sigma=sigma,
                num_steps=num_steps,
                num_selected=1,
                num_epochs=1,
                delta=delta,
            )
            pld = gaussian_allocation_PLD(params=params, config=config, bound_type=bound_type)
            epsilon = float(pld.get_epsilon_for_delta(delta))
            epsilons.append(epsilon)

        # Check monotonic decrease
        for i in range(len(epsilons) - 1):
            assert epsilons[i] > epsilons[i + 1], (
                f"Epsilon-sigma monotonicity violated:\n"
                f"  sigma={sigma_values[i]}: epsilon={epsilons[i]:.6f}\n"
                f"  sigma={sigma_values[i+1]}: epsilon={epsilons[i+1]:.6f}\n"
                f"  Expected epsilon to decrease as sigma increases"
            )

    @pytest.mark.parametrize("num_steps", [5, 10])
    def test_epsilon_decreases_with_delta(self, num_steps: int):
        """Property: epsilon decreases as delta increases."""
        config = _default_config()
        sigma = 2.0

        delta_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        epsilons = []

        for delta in delta_values:
            params = PrivacyParams(
                sigma=sigma,
                num_steps=num_steps,
                num_selected=1,
                num_epochs=1,
                delta=delta,
            )
            pld = gaussian_allocation_PLD(params=params, config=config)
            epsilon = float(pld.get_epsilon_for_delta(delta))
            epsilons.append(epsilon)

        # Check monotonic decrease
        for i in range(len(epsilons) - 1):
            assert epsilons[i] > epsilons[i + 1], (
                f"Epsilon-delta monotonicity violated:\n"
                f"  delta={delta_values[i]:.1e}: epsilon={epsilons[i]:.6f}\n"
                f"  delta={delta_values[i+1]:.1e}: epsilon={epsilons[i+1]:.6f}\n"
                f"  Expected epsilon to decrease with delta"
            )


class TestBoundTypeOrdering:
    """Test that bound types maintain proper ordering (DOMINATES >= IS_DOMINATED)."""

    @pytest.mark.parametrize("sigma", [2.0, 3.0])
    @pytest.mark.parametrize("num_steps", [4, 10])
    @pytest.mark.parametrize("num_selected", [1, 2])
    def test_dominates_bound_is_pessimistic(
        self,
        sigma: float,
        num_steps: int,
        num_selected: int,
    ):
        """Property: DOMINATES (upper) bound >= IS_DOMINATED (lower) bound.

        For the same parameters, the DOMINATES bound should give a larger
        (more conservative) epsilon than IS_DOMINATED.
        """
        # Gaussian REMOVE path requires at least two steps per round.
        if num_steps < 2 * num_selected:
            pytest.skip("requires floor(num_steps / num_selected) >= 2")

        config = _default_config()
        delta = 1e-5

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=1,
            delta=delta,
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

        epsilon_upper = float(pld_upper.get_epsilon_for_delta(delta))
        epsilon_lower = float(pld_lower.get_epsilon_for_delta(delta))

        # Upper bound should be >= lower bound (allow small numerical tolerance)
        tolerance = 1e-6
        assert epsilon_upper >= epsilon_lower - tolerance, (
            f"Bound ordering property violated:\n"
            f"  DOMINATES epsilon:     {epsilon_upper:.6f}\n"
            f"  IS_DOMINATED epsilon:  {epsilon_lower:.6f}\n"
            f"  Parameters: sigma={sigma}, num_steps={num_steps}, num_selected={num_selected}\n"
            f"  Expected DOMINATES >= IS_DOMINATED"
        )


class TestReDiscretizationStability:
    """Test that results are stable under re-discretization within tolerance."""
    pytestmark = [pytest.mark.nightly, pytest.mark.slow]

    @pytest.mark.parametrize("sigma", [2.0, 3.0])
    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_discretization_refinement_converges(self, sigma: float, bound_type: BoundType):
        """Property: Finer discretization should converge to stable value.

        Results should become more stable (less sensitive) as discretization
        becomes finer.
        """
        delta = 1e-5
        params = PrivacyParams(
            sigma=sigma,
            num_steps=5,
            num_selected=1,
            num_epochs=1,
            delta=delta,
        )

        discretization_values = [1e-2, 5e-3, 2e-3]
        epsilons = []

        for disc in discretization_values:
            config = AllocationSchemeConfig(
                loss_discretization=disc,
                tail_truncation=1e-10,
                convolution_method=ConvolutionMethod.GEOM,
            )
            pld = gaussian_allocation_PLD(params=params, config=config, bound_type=bound_type)
            epsilon = float(pld.get_epsilon_for_delta(delta))
            epsilons.append(epsilon)

        # Check that consecutive values get closer (convergence)
        differences = [abs(epsilons[i+1] - epsilons[i]) for i in range(len(epsilons) - 1)]

        # Differences should generally decrease or remain small
        # (allowing some variation due to discretization artifacts)
        max_diff = max(differences)
        assert max_diff < 0.05, (
            f"Discretization refinement shows instability:\n"
            f"  Discretization values: {discretization_values}\n"
            f"  Epsilon values: {epsilons}\n"
            f"  Differences: {differences}\n"
            f"  Max difference: {max_diff:.6f}\n"
            f"  Expected convergence with finer discretization"
        )

    def test_grid_coarsening_stability(self):
        """Property: Moderate coarsening shouldn't drastically change results."""
        sigma = 2.0
        delta = 1e-5

        realization = _gaussian_realization(sigma, num_points=1500)

        # Original fine grid
        config_fine = AllocationSchemeConfig(
            loss_discretization=1e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld_fine = general_allocation_PLD(
            num_steps=3,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config_fine,
        )

        # Moderately coarsened grid
        config_coarse = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld_coarse = general_allocation_PLD(
            num_steps=3,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config_coarse,
        )

        epsilon_fine = float(pld_fine.get_epsilon_for_delta(delta))
        epsilon_coarse = float(pld_coarse.get_epsilon_for_delta(delta))

        # Results should be close (within tolerance)
        rel_diff = abs(epsilon_fine - epsilon_coarse) / epsilon_fine
        assert rel_diff < 0.1, (
            f"Coarsening caused excessive drift:\n"
            f"  Fine discretization (1e-3): epsilon={epsilon_fine:.6f}\n"
            f"  Coarse discretization (5e-3): epsilon={epsilon_coarse:.6f}\n"
            f"  Relative difference: {rel_diff:.2%}\n"
            f"  Expected < 10% difference for moderate coarsening"
        )


class TestCompositionStructuralInvariants:
    """Test structural invariants of composition operations."""
    pytestmark = [pytest.mark.nightly, pytest.mark.slow]

    def test_composition_count_remainder_affects_result(self):
        """Property: non-divisible remainder contributes to composition outcome."""
        realization = PLDRealization(
            x_min=0.0,
            x_gap=0.05,
            PMF_array=np.array([0.55, 0.30, 0.15]),
        )

        config = AllocationSchemeConfig(
            loss_discretization=1e-2,
            convolution_method=ConvolutionMethod.GEOM,
        )

        num_selected = 4
        epsilon_query = 0.2

        # Test pairs with the same floor division but different remainders.
        test_pairs = [
            (4, 5),   # floor(4/4)=1, floor(5/4)=1
            (8, 9),   # floor(8/4)=2, floor(9/4)=2
            (12, 15), # floor(12/4)=3, floor(15/4)=3
        ]

        for num_steps_a, num_steps_b in test_pairs:
            pld_a = general_allocation_PLD(
                num_steps=num_steps_a,
                num_selected=num_selected,
                num_epochs=1,
                remove_realization=realization,
                add_realization=realization,
                config=config,
            )

            pld_b = general_allocation_PLD(
                num_steps=num_steps_b,
                num_selected=num_selected,
                num_epochs=1,
                remove_realization=realization,
                add_realization=realization,
                config=config,
            )

            delta_a = float(pld_a.get_delta_for_epsilon(epsilon_query))
            delta_b = float(pld_b.get_delta_for_epsilon(epsilon_query))

            assert abs(delta_a - delta_b) > 1e-6, (
                f"Expected remainder-sensitive difference:\n"
                f"  num_steps={num_steps_a}: delta={delta_a}\n"
                f"  num_steps={num_steps_b}: delta={delta_b}\n"
                f"  Both have floor(n/{num_selected})={num_steps_a // num_selected}, "
                f"but different remainders should change composition"
            )

    def test_epoch_multiplier_invariant(self):
        """Property: num_epochs multiplies outer composition count.

        Doubling num_epochs should increase privacy loss in a predictable way.
        """
        realization = PLDRealization(
            x_min=0.0,
            x_gap=0.05,
            PMF_array=np.array([0.55, 0.30, 0.15]),
        )

        config = AllocationSchemeConfig(
            loss_discretization=1e-2,
            convolution_method=ConvolutionMethod.GEOM,
        )

        epsilon_query = 0.2

        pld_1_epoch = general_allocation_PLD(
            num_steps=10,
            num_selected=2,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        pld_2_epochs = general_allocation_PLD(
            num_steps=10,
            num_selected=2,
            num_epochs=2,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        delta_1 = float(pld_1_epoch.get_delta_for_epsilon(epsilon_query))
        delta_2 = float(pld_2_epochs.get_delta_for_epsilon(epsilon_query))

        # 2 epochs should give higher delta than 1 epoch (fixed epsilon).
        assert delta_2 > delta_1, (
            f"Epoch multiplier monotonicity violated:\n"
            f"  1 epoch:  delta={delta_1:.6f}\n"
            f"  2 epochs: delta={delta_2:.6f}\n"
            f"  Expected delta_2 > delta_1"
        )

        # Ratio should be material but bounded.
        ratio = delta_2 / delta_1
        assert 1.2 < ratio < 3.0, (
            f"Epoch doubling delta ratio outside expected range:\n"
            f"  ratio = {ratio:.3f}\n"
            f"  Expected between 1.2 and 3.0 for doubling epochs"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

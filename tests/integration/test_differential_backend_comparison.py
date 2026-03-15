"""
Differential tests comparing alternate backends and implementations.

Tests equivalence between different computational paths:
- FFT vs Geometric convolution
- Direct Gaussian vs realization-built Gaussian
- Different discretization strategies

Per Testing Guidelines Section 5: Differential Test Strategy
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

# Mark all tests in this module
pytestmark = [pytest.mark.integration, pytest.mark.differential]

from PLD_accounting import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    PrivacyParams,
    gaussian_allocation_PLD,
    general_allocation_PLD,
)
from PLD_accounting.discrete_dist import LinearDiscreteDist, PLDRealization


def _gaussian_realization(sigma: float, num_points: int = 1500) -> PLDRealization:
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


# ============================================================================
# Differential Tests: FFT vs Geometric Convolution
# ============================================================================

class TestFFTvsGeometricBackend:
    """Test that FFT and Geometric convolution backends produce consistent results.

    Where both methods are applicable, they should give equivalent results
    within numerical tolerance.
    """

    @pytest.mark.parametrize("sigma", [2.0])
    @pytest.mark.parametrize("num_steps", [2, 5])
    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_fft_vs_geom_simple_gaussian(
        self,
        sigma: float,
        num_steps: int,
        bound_type: BoundType,
    ):
        """Test FFT vs GEOM for simple Gaussian allocation (num_selected=1).

        Both backends should produce very similar results for basic scenarios.
        """
        delta = 1e-5

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            delta=delta,
        )

        # FFT backend
        config_fft = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            max_grid_FFT=200_000,
            convolution_method=ConvolutionMethod.FFT,
        )

        pld_fft = gaussian_allocation_PLD(
            params=params,
            config=config_fft,
            bound_type=bound_type,
        )

        # Geometric backend
        config_geom = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            max_grid_mult=30_000,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld_geom = gaussian_allocation_PLD(
            params=params,
            config=config_geom,
            bound_type=bound_type,
        )

        # Compare epsilon values
        epsilon_fft = float(pld_fft.get_epsilon_for_delta(delta))
        epsilon_geom = float(pld_geom.get_epsilon_for_delta(delta))

        # Both should be finite
        assert np.isfinite(epsilon_fft) and np.isfinite(epsilon_geom)

        # Should be close (within tolerance)
        tolerance = 0.03  # 3% tolerance for backend differences
        abs_diff = abs(epsilon_fft - epsilon_geom)
        rel_diff = abs_diff / max(epsilon_fft, epsilon_geom)

        assert rel_diff < tolerance, (
            f"FFT vs GEOM backend mismatch:\n"
            f"  FFT epsilon:  {epsilon_fft:.6f}\n"
            f"  GEOM epsilon: {epsilon_geom:.6f}\n"
            f"  Relative difference: {rel_diff:.2%}\n"
            f"  Tolerance: {tolerance:.2%}\n"
            f"  Parameters: sigma={sigma}, num_steps={num_steps}, bound_type={bound_type.name}"
        )

    @pytest.mark.parametrize("num_steps", [4])
    def test_fft_vs_geom_delta_query(self, num_steps: int):
        """Test FFT vs GEOM for delta queries."""
        sigma = 2.0
        epsilon = 1.0

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            epsilon=epsilon,
        )

        # FFT backend
        config_fft = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.FFT,
        )

        pld_fft = gaussian_allocation_PLD(
            params=params,
            config=config_fft,
        )

        # Geometric backend
        config_geom = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld_geom = gaussian_allocation_PLD(
            params=params,
            config=config_geom,
        )

        # Compare delta values
        delta_fft = float(pld_fft.get_delta_for_epsilon(epsilon))
        delta_geom = float(pld_geom.get_delta_for_epsilon(epsilon))

        # Both should be in valid range
        assert 0.0 <= delta_fft <= 1.0
        assert 0.0 <= delta_geom <= 1.0

        # Should be close
        if delta_fft > 1e-8 and delta_geom > 1e-8:
            rel_diff = abs(delta_fft - delta_geom) / max(delta_fft, delta_geom)
            assert rel_diff < 0.15, (
                f"FFT vs GEOM delta mismatch:\n"
                f"  FFT delta:  {delta_fft:.2e}\n"
                f"  GEOM delta: {delta_geom:.2e}\n"
                f"  Relative difference: {rel_diff:.2%}"
            )
        else:
            # Very small deltas, use absolute comparison
            abs_diff = abs(delta_fft - delta_geom)
            assert abs_diff < 1e-7, (
                f"FFT vs GEOM small delta mismatch:\n"
                f"  FFT delta:  {delta_fft:.2e}\n"
                f"  GEOM delta: {delta_geom:.2e}\n"
                f"  Absolute difference: {abs_diff:.2e}"
            )

    @pytest.mark.parametrize("convolution_method", [ConvolutionMethod.BEST_OF_TWO, ConvolutionMethod.COMBINED])
    def test_hybrid_methods_consistency(self, convolution_method: ConvolutionMethod):
        """Test that hybrid methods (BEST_OF_TWO, COMBINED) produce valid results.

        These methods should automatically choose between FFT and GEOM,
        and results should be consistent with direct backend choices.
        """
        sigma = 2.0
        num_steps = 8
        delta = 1e-5

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            delta=delta,
        )

        config = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            max_grid_FFT=200_000,
            max_grid_mult=30_000,
            convolution_method=convolution_method,
        )

        pld = gaussian_allocation_PLD(
            params=params,
            config=config,
        )

        epsilon = float(pld.get_epsilon_for_delta(delta))

        # Result should be finite and reasonable
        assert np.isfinite(epsilon) and epsilon > 0.0
        assert epsilon < 5.0, f"Epsilon unexpectedly large: {epsilon}"

        # Should be in reasonable range compared to analytical bounds
        # For sigma=2, num_steps=8, expect epsilon roughly in [0.5, 3.0]
        assert 0.2 < epsilon < 4.0, (
            f"Hybrid method result outside expected range:\n"
            f"  Method: {convolution_method.name}\n"
            f"  Epsilon: {epsilon:.6f}\n"
            f"  Expected roughly in [0.2, 4.0]"
        )


# ============================================================================
# Differential Tests: Direct vs Realization-Based Paths
# ============================================================================

class TestDirectVsRealizationPath:
    """Test equivalence between direct Gaussian and realization-built Gaussian."""

    @pytest.mark.parametrize("sigma", [2.0, 3.0])
    @pytest.mark.parametrize("num_steps", [2, 5])
    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_direct_gaussian_vs_realization_gaussian(
        self,
        sigma: float,
        num_steps: int,
        bound_type: BoundType,
    ):
        """Differential test: Direct Gaussian vs realization-built Gaussian.

        A Gaussian realization should produce the same results as the
        direct Gaussian implementation (num_selected=1 case).
        """
        delta = 1e-5

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            delta=delta,
        )

        config = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        # Direct Gaussian path
        pld_direct = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=bound_type,
        )

        # Realization-based path
        realization = _gaussian_realization(sigma)
        pld_realization = general_allocation_PLD(
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
            bound_type=bound_type,
        )

        # Compare epsilon values
        epsilon_direct = float(pld_direct.get_epsilon_for_delta(delta))
        epsilon_realization = float(pld_realization.get_epsilon_for_delta(delta))

        # Should be close
        tolerance = 0.03  # 3% tolerance
        abs_diff = abs(epsilon_direct - epsilon_realization)
        rel_diff = abs_diff / max(epsilon_direct, epsilon_realization)

        assert rel_diff < tolerance, (
            f"Direct vs Realization path mismatch:\n"
            f"  Direct epsilon:       {epsilon_direct:.6f}\n"
            f"  Realization epsilon:  {epsilon_realization:.6f}\n"
            f"  Relative difference:  {rel_diff:.2%}\n"
            f"  Tolerance:            {tolerance:.2%}\n"
            f"  Parameters: sigma={sigma}, num_steps={num_steps}, bound_type={bound_type.name}"
        )

    @pytest.mark.parametrize("num_selected", [2])
    @pytest.mark.parametrize("num_epochs", [1, 2])
    def test_gaussian_vs_realization_with_allocation(
        self,
        num_selected: int,
        num_epochs: int,
    ):
        """Differential test: Gaussian vs realization for general allocation.

        Even with num_selected > 1 and num_epochs > 1, the Gaussian
        and realization paths should match.
        """
        sigma = 2.0
        num_steps = num_selected * 4  # Ensure num_steps >= num_selected
        delta = 1e-5

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            delta=delta,
        )

        config = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        # Direct Gaussian path
        pld_direct = gaussian_allocation_PLD(
            params=params,
            config=config,
        )

        # Realization-based path
        realization = _gaussian_realization(sigma)
        pld_realization = general_allocation_PLD(
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        # Compare epsilon values
        epsilon_direct = float(pld_direct.get_epsilon_for_delta(delta))
        epsilon_realization = float(pld_realization.get_epsilon_for_delta(delta))

        # Should be close (allow more tolerance for complex compositions)
        tolerance = 0.08  # 8% tolerance for multi-stage composition
        abs_diff = abs(epsilon_direct - epsilon_realization)
        rel_diff = abs_diff / max(epsilon_direct, epsilon_realization)

        assert rel_diff < tolerance, (
            f"Allocation path mismatch:\n"
            f"  Direct epsilon:       {epsilon_direct:.6f}\n"
            f"  Realization epsilon:  {epsilon_realization:.6f}\n"
            f"  Relative difference:  {rel_diff:.2%}\n"
            f"  Tolerance:            {tolerance:.2%}\n"
            f"  Parameters: num_steps={num_steps}, num_selected={num_selected}, "
            f"num_epochs={num_epochs}"
        )


# ============================================================================
# Differential Tests: Baseline Comparison Against Trusted Reference
# ============================================================================

class TestBaselineComparison:
    """Compare against trusted baseline values from master@e870048."""
    pytestmark = [pytest.mark.nightly, pytest.mark.slow]

    MASTER_BASELINE_COMMIT = "e870048"
    MASTER_BASELINE_EPSILON_AT_DELTA_1E3 = {
        (1.0, 2): 2.499438514714,
        (2.0, 4): 0.643096678546,
        (3.0, 6): 0.299278152147,
        (1.5, 10): 0.575567981226,
    }

    @pytest.mark.parametrize("sigma,num_steps", [
        (1.0, 2),
        (2.0, 4),
        (3.0, 6),
        (1.5, 10),
    ])
    def test_against_master_baseline_epsilons(
        self,
        sigma: float,
        num_steps: int,
    ):
        """Current implementation should stay close to trusted master baseline."""
        delta = 1e-3

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            delta=delta,
        )

        config = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld = gaussian_allocation_PLD(
            params=params,
            config=config,
        )

        epsilon = float(pld.get_epsilon_for_delta(delta))
        baseline_epsilon = self.MASTER_BASELINE_EPSILON_AT_DELTA_1E3[(sigma, num_steps)]
        rel_diff = abs(epsilon - baseline_epsilon) / baseline_epsilon

        assert rel_diff < 0.02, (
            f"Epsilon drifted from trusted baseline (master@{self.MASTER_BASELINE_COMMIT}):\n"
            f"  Computed epsilon: {epsilon:.6f}\n"
            f"  Baseline epsilon: {baseline_epsilon:.6f}\n"
            f"  Relative diff:    {rel_diff:.2%}\n"
            f"  Parameters: sigma={sigma}, num_steps={num_steps}, delta={delta:.1e}\n"
            f"  This may indicate drift from reference implementation"
        )

    def test_composition_scales_like_master_trend_for_large_num_steps(self):
        """The baseline trend is monotone decreasing as ``num_steps`` grows."""
        sigma = 5.0
        delta = 1e-5

        epsilons = []
        num_steps_values = [2, 4, 8, 16]

        for num_steps in num_steps_values:
            params = PrivacyParams(
                sigma=sigma,
                num_steps=num_steps,
                num_selected=1,
                num_epochs=1,
                delta=delta,
            )

            config = AllocationSchemeConfig(
                loss_discretization=5e-3,
                tail_truncation=1e-10,
                convolution_method=ConvolutionMethod.GEOM,
            )

            pld = gaussian_allocation_PLD(params=params, config=config)
            epsilon = float(pld.get_epsilon_for_delta(delta))
            epsilons.append(epsilon)

        ratio_1_2 = epsilons[1] / epsilons[0]
        ratio_2_4 = epsilons[2] / epsilons[1]
        ratio_4_8 = epsilons[3] / epsilons[2]

        # Trusted baseline (master@e870048) gives ratios around 0.68.
        for ratio in [ratio_1_2, ratio_2_4, ratio_4_8]:
            assert 0.55 < ratio < 0.8, (
                f"Composition scaling deviates from trusted baseline trend:\n"
                f"  Ratios: {ratio_1_2:.3f}, {ratio_2_4:.3f}, {ratio_4_8:.3f}\n"
                f"  Epsilons: {epsilons}\n"
                f"  Expected ratios near ~0.68 for this API's t-semantics"
            )


# ============================================================================
# Differential Tests: Symmetry Properties
# ============================================================================

class TestSymmetryProperties:
    """Test symmetry properties that should hold across implementations."""

    def test_remove_add_symmetry_for_identical_realizations(self):
        """Differential test: Identical remove/add realizations should be symmetric.

        When remove and add realizations are identical, the result should
        be consistent regardless of implementation details.
        """
        sigma = 2.0
        num_steps = 5
        delta = 1e-5

        realization = _gaussian_realization(sigma)

        config = AllocationSchemeConfig(
            loss_discretization=5e-3,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.GEOM,
        )

        # Standard allocation
        pld = general_allocation_PLD(
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )

        # Should match direct Gaussian
        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            delta=delta,
        )

        pld_gaussian = gaussian_allocation_PLD(params=params, config=config)

        epsilon_realization = float(pld.get_epsilon_for_delta(delta))
        epsilon_gaussian = float(pld_gaussian.get_epsilon_for_delta(delta))

        abs_diff = abs(epsilon_realization - epsilon_gaussian)
        assert abs_diff < 0.02, (
            f"Symmetry property violated:\n"
            f"  Realization epsilon: {epsilon_realization:.6f}\n"
            f"  Gaussian epsilon:    {epsilon_gaussian:.6f}\n"
            f"  Difference:          {abs_diff:.6f}\n"
            f"  Expected near-identical results for symmetric case"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

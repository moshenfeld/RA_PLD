"""
Unit tests for discrete_convolution.types module.

Tests DiscreteDist validation and mass conservation.
"""
import pytest
import numpy as np
from PLD_accounting.core_utils import PMF_MASS_TOL
from PLD_accounting.types import BoundType, SpacingType, ConvolutionMethod
from PLD_accounting.discrete_dist import DiscreteDist


class TestDiscreteDist:
    """Test DiscreteDist dataclass validation."""

    def test_valid_distribution(self):
        """Test that valid distribution is accepted."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        dist = DiscreteDist(x_array=x, PMF_array=pmf)
        assert np.allclose(dist.x_array, x)
        assert np.allclose(dist.PMF_array, pmf)
        assert dist.p_neg_inf == 0.0
        assert dist.p_pos_inf == 0.0

    def test_with_infinite_mass(self):
        """Test distribution with mass at infinity."""
        x = np.array([1.0, 2.0])
        pmf = np.array([0.3, 0.5], dtype=np.float64)
        dist = DiscreteDist(x_array=x, PMF_array=pmf, p_neg_inf=0.1, p_pos_inf=0.1)
        assert dist.p_neg_inf == 0.1
        assert dist.p_pos_inf == 0.1

    def test_non_increasing_x_raises(self):
        """Test that non-increasing x array raises error."""
        x = np.array([1.0, 3.0, 2.0])
        pmf = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        with pytest.raises(ValueError, match="strictly increasing"):
            DiscreteDist(x_array=x, PMF_array=pmf)

    def test_negative_pmf_raises(self):
        """Test that negative PMF values raise error."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.3, -0.1, 0.8], dtype=np.float64)
        with pytest.raises(ValueError, match="nonnegative"):
            DiscreteDist(x_array=x, PMF_array=pmf)

    def test_mismatched_shapes_raises(self):
        """Test that mismatched array shapes raise error."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.5, 0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="equal length"):
            DiscreteDist(x_array=x, PMF_array=pmf)

    def test_mass_not_conserved_raises(self):
        """Test that non-unit total mass raises error."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.2, 0.3, 0.4], dtype=np.float64)  # sums to 0.9
        dist = DiscreteDist(x_array=x, PMF_array=pmf)
        with pytest.raises(ValueError, match="MASS CONSERVATION ERROR"):
            dist.validate_mass_conservation(BoundType.DOMINATES)

    def test_mass_within_tolerance_accepted(self):
        """Test that mass within tolerance is accepted."""
        x = np.array([1.0, 2.0])
        # Create PMF that sums to 1.0 within tolerance
        pmf = np.array([0.5, 0.5 + PMF_MASS_TOL/2], dtype=np.float64)
        # Should not raise
        dist = DiscreteDist(x_array=x, PMF_array=pmf)
        dist.validate_mass_conservation(BoundType.DOMINATES)
        assert dist is not None


class TestEnsureMassConservation:
    """Test DiscreteDist.validate_mass_conservation."""

    def test_exact_conservation(self):
        """Test exact mass conservation passes."""
        dist = DiscreteDist(x_array=np.array([1.0, 2.0, 3.0]),
                            PMF_array=np.array([0.25, 0.5, 0.25], dtype=np.float64))
        dist.validate_mass_conservation(BoundType.DOMINATES)

    def test_with_infinite_mass(self):
        """Test conservation with infinite mass."""
        dist = DiscreteDist(x_array=np.array([1.0, 2.0]),
                            PMF_array=np.array([0.2, 0.3], dtype=np.float64),
                            p_neg_inf=0.1,
                            p_pos_inf=0.4)
        # This test has both infinities, which violates both bound types
        # We'll use DOMINATES to check the validation
        with pytest.raises(ValueError, match="DOMINATES.*p_neg_inf"):
            dist.validate_mass_conservation(BoundType.DOMINATES)

    def test_violation_raises(self):
        """Test that violation raises detailed error."""
        dist = DiscreteDist(x_array=np.array([1.0, 2.0]),
                            PMF_array=np.array([0.3, 0.3], dtype=np.float64))
        with pytest.raises(ValueError, match="MASS CONSERVATION ERROR") as exc_info:
            dist.validate_mass_conservation(BoundType.DOMINATES)
        # Check detailed error message
        assert "tolerance=" in str(exc_info.value)
        assert "PMF sum=" in str(exc_info.value)

    def test_high_precision_summation(self):
        """Test that high-precision summation is used."""
        # Create many small values that could accumulate rounding error
        n = 1000
        pmf = np.array([1.0/n] * n, dtype=np.float64)
        dist = DiscreteDist(x_array=np.arange(n, dtype=np.float64), PMF_array=pmf)
        dist.validate_mass_conservation(BoundType.DOMINATES)  # Should pass with fsum


class TestEnums:
    """Test enum definitions."""

    def test_bound_type_values(self):
        """Test BoundType enum values."""
        assert BoundType.DOMINATES.value == "DOMINATES"
        assert BoundType.IS_DOMINATED.value == "IS_DOMINATED"

    def test_spacing_type_values(self):
        """Test SpacingType enum values."""
        assert SpacingType.LINEAR.value == "linear"
        assert SpacingType.GEOMETRIC.value == "geometric"

    def test_convolution_method_values(self):
        """Test ConvolutionMethod enum values."""
        assert ConvolutionMethod.GEOM.value == "geometric"
        assert ConvolutionMethod.FFT.value == "fft"

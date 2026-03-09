import numpy as np
import pytest

from PLD_accounting.discrete_dist import (
    GeneralDiscreteDist,
    DenseLinearDiscreteDist,
    SparseLinearDiscreteDist,
)
from PLD_accounting.dp_accounting_support import (
    discrete_dist_to_dp_accounting_pmf,
    dp_accounting_pmf_to_discrete_dist,
)
from dp_accounting.pld.pld_pmf import DensePLDPmf, SparsePLDPmf


def _make_linear_dist() -> DenseLinearDiscreteDist:
    return DenseLinearDiscreteDist.from_x_array(
        x_array=np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float64),
        PMF_array=np.array([0.2, 0.3, 0.25, 0.15], dtype=np.float64),
        p_neg_inf=0.0,
        p_pos_inf=0.1,
    )


def test_dp_accounting_roundtrip_preserves_mass_and_grid_shape():
    original = _make_linear_dist()
    pmf = discrete_dist_to_dp_accounting_pmf(original, pessimistic_estimate=True)
    restored = dp_accounting_pmf_to_discrete_dist(pmf)

    assert restored.x_array.shape == original.x_array.shape
    total_mass = np.sum(restored.PMF_array) + restored.p_neg_inf + restored.p_pos_inf
    assert np.isclose(total_mass, 1.0, atol=1e-12)
    assert np.isclose(restored.p_pos_inf, original.p_pos_inf, atol=1e-12)


def test_discrete_dist_to_dp_accounting_handles_zero_finite_mass():
    """Test that distributions with all mass at infinity are handled correctly."""
    dist = DenseLinearDiscreteDist.from_x_array(
        x_array=np.array([0.0, 1.0], dtype=np.float64),
        PMF_array=np.array([0.0, 0.0], dtype=np.float64),
        p_neg_inf=0.0,
        p_pos_inf=1.0,
    )
    # dp_accounting allows all mass at infinity
    pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
    assert pmf._infinity_mass == 1.0
    assert np.allclose(pmf._probs, np.array([0.0, 0.0]))
class TestDenseLinearDistAdapter:
    """Test conversion from DenseLinearDiscreteDist to DensePLDPmf."""

    def test_dense_linear_to_dense_pmf(self):
        """Test that DenseLinearDiscreteDist converts to DensePLDPmf."""
        dist = DenseLinearDiscreteDist(
            x_min=0.0,
            x_gap=0.5,
            PMF_array=np.array([0.2, 0.3, 0.4, 0.1]),
            p_pos_inf=0.0
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._discretization == 0.5
        assert np.allclose(pmf._probs, dist.PMF_array)

    def test_dense_linear_with_nonzero_base(self):
        """Test DenseLinearDiscreteDist with non-zero x_min."""
        # x_min = 1.0, x_gap = 0.5 -> base_index = 2
        dist = DenseLinearDiscreteDist(
            x_min=1.0,
            x_gap=0.5,
            PMF_array=np.array([0.3, 0.4, 0.3]),
            p_pos_inf=0.0
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._discretization == 0.5
        assert pmf._lower_loss == 2  # 1.0 / 0.5 = 2

    def test_dense_linear_with_infinity_mass(self):
        """Test DenseLinearDiscreteDist with p_pos_inf."""
        dist = DenseLinearDiscreteDist(
            x_min=0.0,
            x_gap=0.25,
            PMF_array=np.array([0.2, 0.5, 0.2]),
            p_pos_inf=0.1
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._infinity_mass == 0.1

    def test_dense_linear_roundtrip(self):
        """Test DenseLinear -> PMF -> DiscreteDist roundtrip."""
        dist = DenseLinearDiscreteDist(
            x_min=0.5,
            x_gap=0.25,
            PMF_array=np.array([0.2, 0.3, 0.4, 0.1]),
            p_pos_inf=0.0
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        restored = dp_accounting_pmf_to_discrete_dist(pmf)

        # Check values match
        assert np.allclose(dist.x_array, restored.x_array)
        assert np.allclose(dist.PMF_array, restored.PMF_array)
        assert np.isclose(dist.p_pos_inf, restored.p_pos_inf)


class TestSparseLinearDistAdapter:
    """Test conversion from SparseLinearDiscreteDist to SparsePLDPmf."""

    def test_sparse_linear_to_sparse_pmf(self):
        """Test that SparseLinearDiscreteDist converts to SparsePLDPmf."""
        dist = SparseLinearDiscreteDist(
            x_min=0.0,
            x_gap=0.5,
            indices=np.array([0, 2, 5], dtype=np.int64),
            PMF_array=np.array([0.3, 0.5, 0.2]),
            p_pos_inf=0.0
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        assert isinstance(pmf, SparsePLDPmf)
        assert pmf._discretization == 0.5

        # Check that loss_probs dictionary has correct mapping
        expected_losses = {0: 0.3, 2: 0.5, 5: 0.2}
        assert pmf._loss_probs == expected_losses

    def test_sparse_linear_with_nonzero_base(self):
        """Test SparseLinearDiscreteDist with non-zero x_min."""
        # x_min = 1.0, x_gap = 0.5 -> base_index = 2
        # indices [0, 2, 5] -> absolute indices [2, 4, 7]
        dist = SparseLinearDiscreteDist(
            x_min=1.0,
            x_gap=0.5,
            indices=np.array([0, 2, 5], dtype=np.int64),
            PMF_array=np.array([0.3, 0.4, 0.3]),
            p_pos_inf=0.0
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        assert isinstance(pmf, SparsePLDPmf)

        # Check absolute loss indices
        expected_losses = {2: 0.3, 4: 0.4, 7: 0.3}
        assert pmf._loss_probs == expected_losses

    def test_sparse_linear_with_infinity_mass(self):
        """Test SparseLinearDiscreteDist with p_pos_inf."""
        dist = SparseLinearDiscreteDist(
            x_min=0.0,
            x_gap=0.25,
            indices=np.array([0, 3, 8], dtype=np.int64),
            PMF_array=np.array([0.2, 0.5, 0.2]),
            p_pos_inf=0.1
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        assert isinstance(pmf, SparsePLDPmf)
        assert pmf._infinity_mass == 0.1

    def test_sparse_linear_filters_zero_probabilities(self):
        """Test that zero probabilities are filtered out."""
        dist = SparseLinearDiscreteDist(
            x_min=0.0,
            x_gap=0.5,
            indices=np.array([0, 2, 5, 7], dtype=np.int64),
            PMF_array=np.array([0.3, 0.0, 0.5, 0.2]),  # index 2 has zero prob
            p_pos_inf=0.0
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        assert isinstance(pmf, SparsePLDPmf)

        # Zero probability at index 2 should be filtered out
        expected_losses = {0: 0.3, 5: 0.5, 7: 0.2}
        assert pmf._loss_probs == expected_losses

    def test_sparse_linear_roundtrip(self):
        """Test SparseLinear -> PMF -> DiscreteDist roundtrip."""
        dist = SparseLinearDiscreteDist(
            x_min=0.5,
            x_gap=0.25,
            indices=np.array([0, 3, 7], dtype=np.int64),
            PMF_array=np.array([0.3, 0.5, 0.2]),
            p_pos_inf=0.0
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        restored = dp_accounting_pmf_to_discrete_dist(pmf)

        # Check values match (restored will be GeneralDiscreteDist)
        assert np.allclose(dist.x_array, restored.x_array)
        assert np.allclose(dist.PMF_array, restored.PMF_array)
        assert np.isclose(dist.p_pos_inf, restored.p_pos_inf)


class TestDenseVsSparseAdapterConsistency:
    """Test that dense and sparse adapters produce equivalent results."""

    def test_equivalent_distributions_produce_same_pmf(self):
        """Test that dense and sparse versions of same distribution produce equivalent PMFs."""
        # Create dense distribution
        dist_dense = DenseLinearDiscreteDist(
            x_min=0.0,
            x_gap=0.5,
            PMF_array=np.array([0.2, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0, 0.1]),
            p_pos_inf=0.0
        )

        # Create equivalent sparse distribution (only non-zero entries)
        dist_sparse = SparseLinearDiscreteDist(
            x_min=0.0,
            x_gap=0.5,
            indices=np.array([0, 2, 5, 7], dtype=np.int64),
            PMF_array=np.array([0.2, 0.3, 0.4, 0.1]),
            p_pos_inf=0.0
        )

        # Convert both to PMF
        pmf_dense = discrete_dist_to_dp_accounting_pmf(dist_dense, pessimistic_estimate=True)
        pmf_sparse = discrete_dist_to_dp_accounting_pmf(dist_sparse, pessimistic_estimate=True)

        # Both should have same discretization
        assert pmf_dense._discretization == pmf_sparse._discretization

        # Convert both back to discrete distributions
        restored_dense = dp_accounting_pmf_to_discrete_dist(pmf_dense)
        restored_sparse = dp_accounting_pmf_to_discrete_dist(pmf_sparse)

        # Check that both distributions have same total mass
        mass_dense = np.sum(restored_dense.PMF_array) + restored_dense.p_pos_inf
        mass_sparse = np.sum(restored_sparse.PMF_array) + restored_sparse.p_pos_inf
        assert np.isclose(mass_dense, 1.0, atol=1e-12)
        assert np.isclose(mass_sparse, 1.0, atol=1e-12)

        # Check that non-zero entries match at the same x values
        # Filter to non-zero probabilities for comparison
        nonzero_dense = restored_dense.PMF_array > 1e-15
        nonzero_sparse = restored_sparse.PMF_array > 1e-15

        x_dense_nonzero = restored_dense.x_array[nonzero_dense]
        x_sparse_nonzero = restored_sparse.x_array[nonzero_sparse]
        pmf_dense_nonzero = restored_dense.PMF_array[nonzero_dense]
        pmf_sparse_nonzero = restored_sparse.PMF_array[nonzero_sparse]

        # Both should have same non-zero x values and probabilities
        assert np.allclose(x_dense_nonzero, x_sparse_nonzero)
        assert np.allclose(pmf_dense_nonzero, pmf_sparse_nonzero)

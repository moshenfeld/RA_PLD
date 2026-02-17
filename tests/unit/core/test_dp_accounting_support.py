import numpy as np
import pytest

from PLD_accounting.discrete_dist import DiscreteDist
from PLD_accounting.dp_accounting_support import (
    _align_to_common_grid,
    discrete_dist_to_dp_accounting_pmf,
    dp_accounting_pmf_to_discrete_dist,
)


def _make_linear_dist() -> DiscreteDist:
    return DiscreteDist(
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


def test_discrete_dist_to_dp_accounting_rejects_zero_finite_mass():
    dist = DiscreteDist(
        x_array=np.array([0.0, 1.0], dtype=np.float64),
        PMF_array=np.array([0.0, 0.0], dtype=np.float64),
        p_neg_inf=0.0,
        p_pos_inf=1.0,
    )
    with pytest.raises(ValueError, match="No finite probability mass"):
        discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)


def test_align_to_common_grid_uses_finer_spacing():
    coarse = DiscreteDist(
        x_array=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        PMF_array=np.array([0.2, 0.6, 0.2], dtype=np.float64),
        p_neg_inf=0.0,
        p_pos_inf=0.0,
    )
    fine = DiscreteDist(
        x_array=np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64),
        PMF_array=np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64),
        p_neg_inf=0.0,
        p_pos_inf=0.0,
    )

    aligned_coarse, aligned_fine = _align_to_common_grid(coarse, fine)
    assert np.allclose(aligned_coarse.x_array, fine.x_array)
    assert np.allclose(aligned_fine.x_array, fine.x_array)

    mass_coarse = np.sum(aligned_coarse.PMF_array) + aligned_coarse.p_neg_inf + aligned_coarse.p_pos_inf
    mass_fine = np.sum(aligned_fine.PMF_array) + aligned_fine.p_neg_inf + aligned_fine.p_pos_inf
    assert np.isclose(mass_coarse, 1.0, atol=1e-12)
    assert np.isclose(mass_fine, 1.0, atol=1e-12)

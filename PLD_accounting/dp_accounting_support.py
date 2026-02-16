"""
dp_accounting compatibility wrappers for subsampling implementation.

Provides translation between dp_accounting's PrivacyLossDistribution objects
and our DiscreteDist-based API.
"""
import math
import numpy as np
from numpy.typing import NDArray
from typing import Union

from dp_accounting.pld.privacy_loss_distribution import PrivacyLossDistribution
from dp_accounting.pld.pld_pmf import SparsePLDPmf, DensePLDPmf, PLDPmf
from dp_accounting.pld import pld_pmf

from PLD_accounting.types import BoundType, Direction
from PLD_accounting.discrete_dist import DiscreteDist
from PLD_accounting.core_utils import stable_array_equal, compute_bin_width


# ============================================================================
# Translation Functions: DiscreteDist <-> dp_accounting
# ============================================================================

def discrete_dist_to_dp_accounting_pmf(dist: DiscreteDist, pessimistic_estimate: bool = True) -> PLDPmf:
    """Convert DiscreteDist to dp_accounting PMF.

    Args:
        dist: DiscreteDist object with privacy loss distribution
        pessimistic_estimate: Whether to use pessimistic estimate in dp_accounting

    Returns:
        PLDPmf object

    Notes:
        - Infinity masses are handled via p_pos_inf
        - Loss grid must have uniform spacing
        - Discretization is computed from the grid spacing in dist.x_array
    """
    losses = dist.x_array
    probs = dist.PMF_array.astype(np.float64)

    # Filter to positive probabilities first
    pos_ind = probs > 0
    losses_filtered = losses[pos_ind]
    probs_filtered = probs[pos_ind]
    if losses.size < 2:
        raise ValueError("Less than 2 finite values - cannot convert to dp_accounting PMF")
    if losses_filtered.size == 0:
        raise ValueError("No finite probability mass - cannot convert to dp_accounting PMF")

    # Compute discretization from uniform grid spacing
    discretization = compute_bin_width(losses)
    loss_indices = np.round(losses_filtered / discretization).astype(int)

    # Convert to dp_accounting PMF with explicit infinity mass
    loss_probs_dict = dict(zip(loss_indices.tolist(), probs_filtered.tolist()))

    return pld_pmf.create_pmf(
        loss_probs=loss_probs_dict,
        discretization=discretization,
        infinity_mass=dist.p_pos_inf,
        pessimistic_estimate=pessimistic_estimate
    )



def dp_accounting_pmf_to_discrete_dist(pmf: PLDPmf) -> DiscreteDist:
    """Convert dp_accounting PMF to DiscreteDist.

    Notes:
        - Infinity mass from dp_accounting becomes p_pos_inf in DiscreteDist
        - p_neg_inf is set to 0 (not used in privacy loss distributions)
    """

    # Extract dense loss grid and probabilities from PMF
    if isinstance(pmf, DensePLDPmf):
        probs = pmf._probs.copy()
        losses = pmf._lower_loss + np.arange(np.size(probs))
    elif isinstance(pmf, SparsePLDPmf):
        loss_probs = pmf._loss_probs.copy()
        if len(loss_probs) == 0:
            raise ValueError("Empty dp_accounting PMF is not supported")
        losses_sparse = np.array(list(loss_probs.keys()), dtype=np.int64)
        probs_sparse = np.array(list(loss_probs.values()), dtype=np.float64)
        losses = np.arange(np.min(losses_sparse), np.max(losses_sparse) + 1)
        probs = np.zeros(np.size(losses))
        probs[losses_sparse - np.min(losses_sparse)] = probs_sparse
    else:
        raise AttributeError(f"Unrecognized PMF format: {type(pmf)}. Expected DensePLDPmf or SparsePLDPmf.")

    # Clip probabilities and convert losses to physical units
    probs = np.clip(probs, 0.0, 1.0)
    losses = losses.astype(np.float64) * pmf._discretization

    # Rescale probabilities to represent finite mass only (1 - infinity_mass)
    finite_target = max(0.0, 1.0 - pmf._infinity_mass)
    sum_probs = np.sum(probs, dtype=np.float64)
    if sum_probs > 0.0:
        probs = probs * (finite_target / sum_probs)

    # Get infinity mass
    infinity_mass = pmf._infinity_mass

    return DiscreteDist(
        x_array=losses,
        PMF_array=probs,
        p_neg_inf=0.0,  # Not used for privacy loss distributions
        p_pos_inf=infinity_mass
    )

def _align_to_common_grid(dist_1: DiscreteDist, dist_2: DiscreteDist) -> tuple[DiscreteDist, DiscreteDist]:
    """Align two distributions to a common grid by choosing the finer discretization.

    Returns both distributions resampled onto the grid with smaller spacing.
    """
    # If grids are already compatible, return as-is
    if stable_array_equal(dist_1.x_array, dist_2.x_array):
        return dist_1, dist_2

    # Choose the grid with finer discretization (smaller step size)
    step_1 = compute_bin_width(dist_1.x_array)
    step_2 = compute_bin_width(dist_2.x_array)

    if step_1 <= step_2:
        # dist_1 has finer grid, use it as target
        return dist_1, _resample_onto_grid(dist_2, dist_1.x_array)
    else:
        # dist_2 has finer grid, use it as target
        return _resample_onto_grid(dist_1, dist_2.x_array), dist_2

def _resample_onto_grid(dist: DiscreteDist, target_grid: NDArray[np.float64]) -> DiscreteDist:
    """Resample a distribution onto a different grid using linear interpolation."""
    # Interpolate PMF values onto the new grid
    # Use np.interp which handles out-of-range values by returning 0
    new_pmf = np.interp(target_grid, dist.x_array, dist.PMF_array, left=0.0, right=0.0)

    # Normalize to preserve total probability
    total_orig = np.sum(dist.PMF_array)
    total_new = np.sum(new_pmf)
    if total_new > 0:
        new_pmf *= (total_orig / total_new)

    return DiscreteDist(
        x_array=target_grid,
        PMF_array=new_pmf,
        p_neg_inf=dist.p_neg_inf,
        p_pos_inf=dist.p_pos_inf
    )


"""
Shared random-allocation composition helpers.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.discrete_dist import LinearDiscreteDist
from PLD_accounting.utils import exp_linear_to_geometric, log_geometric_to_linear, negate_reverse_linear_distribution
from PLD_accounting.dp_accounting_support import linear_dist_to_dp_accounting_pmf
from PLD_accounting.distribution_discretization import change_spacing_type
from PLD_accounting.geometric_convolution import geometric_convolve, geometric_self_convolve
from PLD_accounting.FFT_convolution import FFT_convolve, FFT_self_convolve

# =============================================================================
# Public API
# =============================================================================


def allocation_PLD(*,
    compute_base_pmf_remove: Callable[..., LinearDiscreteDist],
    compute_base_pmf_add: Callable[..., LinearDiscreteDist],
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """
    Orchestrate full allocation PLD construction for both directions.

    This function builds REMOVE and ADD PMFs via ``allocation_PMF(...)`` and
    then converts both PMFs to the final ``dp_accounting`` PLD object.
    """

    if bound_type == BoundType.BOTH:
        raise ValueError(
            "allocation_PLD does not support bound_type=BoundType.BOTH; "
            "build separate DOMINATES and IS_DOMINATED PLDs instead"
        )
    remove_dist = allocation_PMF(
        compute_base_pmf=compute_base_pmf_remove,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    add_dist = allocation_PMF(
        compute_base_pmf=compute_base_pmf_add,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    return _compose_pld_from_pmfs(
        remove_dist=remove_dist,
        add_dist=add_dist,
        bound_type=bound_type,
    )

def allocation_PMF(*,
    compute_base_pmf: Callable[..., LinearDiscreteDist],
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Build one-direction allocation PMF with adaptive floor/ceil decomposition.

    For divisible ``num_steps / num_selected``, this builds one component. For
    non-divisible cases, it builds floor and ceil components via
    ``_allocation_PMF_core(...)`` and combines them with one final
    ``FFT_convolve(...)``.
    """
    if num_steps < 1 or num_selected < 1 or num_epochs < 1:
        raise ValueError(f"num_steps (={num_steps}), num_selected (={num_selected}), and num_epochs (={num_epochs}) must be >= 1")
    new_num_steps_floor = int(num_steps // num_selected)
    if new_num_steps_floor < 1:
        raise ValueError("num_steps must be >= num_selected")
    num_epochs_remainder = num_steps - num_selected * new_num_steps_floor
    new_num_steps_ceil = new_num_steps_floor + 1
    new_num_epochs_floor = (num_selected - num_epochs_remainder) * num_epochs
    new_num_epochs_ceil = num_epochs_remainder * num_epochs
    tail_truncation /= 2
    
    dist_floor = None
    dist_ceil = None
    if new_num_epochs_floor > 0:
        dist_floor = _allocation_PMF_core(
            compute_base_pmf=compute_base_pmf,
            num_steps=new_num_steps_floor,
            num_epochs=new_num_epochs_floor,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    if new_num_epochs_ceil > 0:
        dist_ceil = _allocation_PMF_core(
            compute_base_pmf=compute_base_pmf,
            num_steps=new_num_steps_ceil,
            num_epochs=new_num_epochs_ceil,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )

    if dist_floor is None:
        if dist_ceil is None:
            raise RuntimeError("allocation_PMF failed to build either floor or ceil component")
        return dist_ceil
    if dist_ceil is None:
        return dist_floor
    return FFT_convolve(
        dist_1=dist_floor,
        dist_2=dist_ceil,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )


def geometric_allocation_PMF_base_remove(*,
    base_distributions_creation: Callable[..., tuple[LinearDiscreteDist, LinearDiscreteDist]],
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Build the REMOVE component PMF via exp-space geometric composition.

    The callback ``base_distributions_creation`` provides one-step
    ``(base, dual_base)`` factors, which are shifted and composed.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    loss_discretization /= int(2 * np.ceil(np.log2(num_steps)) + 1)
    tail_truncation /= 3
    base_factor_tail_truncation = tail_truncation / num_steps

    base, dual_base = base_distributions_creation(
        loss_discretization=loss_discretization,
        tail_truncation=base_factor_tail_truncation,
        bound_type=bound_type,
    )

    # Subtract the average loss
    log_num_steps = float(np.log(num_steps))
    scaled_dual = LinearDiscreteDist(
        x_min=dual_base.x_min - log_num_steps,
        x_gap=dual_base.x_gap,
        PMF_array=dual_base.PMF_array.copy(),
        p_neg_inf=dual_base.p_neg_inf,
        p_pos_inf=dual_base.p_pos_inf,
    )
    scaled_base = LinearDiscreteDist(
        x_min=base.x_min - log_num_steps,
        x_gap=base.x_gap,
        PMF_array=base.PMF_array.copy(),
        p_neg_inf=base.p_neg_inf,
        p_pos_inf=base.p_pos_inf,
    )

    # Factor preparation in exp-space.
    exp_dual = exp_linear_to_geometric(scaled_dual)
    exp_base = exp_linear_to_geometric(scaled_base)

    if num_steps == 1:
        exp_convolved = exp_base
    else:
        # V_{t-1} <- self-conv(V1, t-1, ...).
        exp_convolved_dual = geometric_self_convolve(
            dist=exp_dual,
            T=num_steps - 1,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
        # U_t <- conv(V_{t-1}, U1, ...).
        exp_convolved = geometric_convolve(
            dist_1=exp_convolved_dual,
            dist_2=exp_base,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    # L_t <- log(U_t).
    return log_geometric_to_linear(exp_convolved)


def geometric_allocation_PMF_base_add(*,
    base_distributions_creation: Callable[..., LinearDiscreteDist],
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Build the ADD component PMF via exp-space geometric self-composition.

    The callback ``base_distributions_creation`` provides the one-step ADD
    factor, which is shifted and composed before mapping back to linear loss.
    """
    loss_discretization /= int(2 * np.ceil(np.log2(num_steps)) + 1)
    tail_truncation /= 2
    base_factor_tail_truncation = tail_truncation / num_steps
 
    base = base_distributions_creation(
        loss_discretization=loss_discretization,
        tail_truncation=base_factor_tail_truncation,
        bound_type=bound_type,
    )

    log_num_steps = float(np.log(num_steps))

    scaled_base = LinearDiscreteDist(
        x_min=base.x_min - log_num_steps,
        x_gap=base.x_gap,
        PMF_array=base.PMF_array.copy(),
        p_neg_inf=base.p_neg_inf,
        p_pos_inf=base.p_pos_inf,
    )

    # Factor preparation in exp-space.
    exp_base = exp_linear_to_geometric(scaled_base)
    exp_bound_type = BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    if num_steps == 1:
        exp_convolved = exp_base
    else:
        # U_t <- self-conv(U, t, lower).
        exp_convolved = geometric_self_convolve(
            dist=exp_base,
            T=num_steps,
            tail_truncation=tail_truncation / 2,
            bound_type=exp_bound_type,
        )
    # L_t <- -log(U_t).
    log_dist = log_geometric_to_linear(exp_convolved)
    return negate_reverse_linear_distribution(log_dist)


# =============================================================================
# Helper Functions
# =============================================================================


def _allocation_PMF_core(*,
    compute_base_pmf: Callable[..., LinearDiscreteDist],
    num_steps: int,
    num_epochs: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Build and finalize one floor/ceil decomposition component.

    This function derives component-level budgets, calls
    ``compute_base_pmf(...)``, regrids to linear spacing, composes across
    epochs, and aligns to output discretization.
    """
    output_tail_truncation = tail_truncation / 3
    base_tail_truncation = output_tail_truncation / (2*num_epochs)
    output_loss_discretization = loss_discretization / 3
    base_loss_discretization = output_loss_discretization / np.sqrt(num_epochs)

    base_dist = compute_base_pmf(
        num_steps=num_steps,
        loss_discretization=base_loss_discretization,
        tail_truncation=base_tail_truncation,
        bound_type=bound_type,
    )
    rediscritized_dist = change_spacing_type(
        dist=base_dist,
        tail_truncation=base_tail_truncation,
        loss_discretization=base_loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    assert isinstance(rediscritized_dist, LinearDiscreteDist)
    
    if num_epochs == 1:
        composed_dist = rediscritized_dist
    else:
        composed_dist = FFT_self_convolve(
            dist=rediscritized_dist,
            T=num_epochs,
            tail_truncation=output_tail_truncation,
            bound_type=bound_type,
            use_direct=True,
        )

    if composed_dist.x_gap < output_loss_discretization:
        final_dist = change_spacing_type(
            dist=composed_dist,
            tail_truncation=output_tail_truncation,
            loss_discretization=output_loss_discretization,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )
        assert isinstance(final_dist, LinearDiscreteDist)
    else:
        final_dist = composed_dist
    return final_dist


def _compose_pld_from_pmfs(*,
    remove_dist: LinearDiscreteDist | None,
    add_dist: LinearDiscreteDist | None,
    bound_type: BoundType,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """
    Convert remove/add PMFs into a ``dp_accounting`` PLD.

    Args:
        remove_dist: REMOVE-direction linear PMF.
        add_dist: Optional ADD-direction linear PMF.
        bound_type: Bound direction used for pessimistic conversion.

    Returns:
        A ``dp_accounting`` privacy loss distribution.
    """
    if remove_dist is None:
        raise ValueError(
            "PLD construction requires remove-direction PMF. "
            "Provide remove_realization or use both directions."
        )
    pessimistic_estimate = bound_type == BoundType.DOMINATES
    pmf_remove = linear_dist_to_dp_accounting_pmf(dist=remove_dist, pessimistic_estimate=pessimistic_estimate)
    if add_dist is None:
        return privacy_loss_distribution.PrivacyLossDistribution(
            pmf_remove=pmf_remove,
        )
    pmf_add = linear_dist_to_dp_accounting_pmf(dist=add_dist, pessimistic_estimate=pessimistic_estimate)
    return privacy_loss_distribution.PrivacyLossDistribution(
        pmf_remove=pmf_remove,
        pmf_add=pmf_add,
    )

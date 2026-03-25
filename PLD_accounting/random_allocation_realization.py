"""
Realization-specific random-allocation accounting.
"""

from __future__ import annotations

from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.discrete_dist import LinearDiscreteDist, PLDRealization
from PLD_accounting.utils import calc_pld_dual, negate_reverse_linear_distribution
from PLD_accounting.distribution_discretization import change_spacing_type


def realization_remove_base_distributions(*,
    realization: PLDRealization,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> tuple[LinearDiscreteDist, LinearDiscreteDist]:
    """
    Prepare remove-direction factors from a loss-space realization.

    Algorithm 1 (`rand-alloc-rem`) in Appendix C.

    Args:
        realization: REMOVE-direction realization in linear loss space.
        loss_discretization: Target linear-grid spacing.
        tail_truncation: Tail truncation budget for regridding.
        bound_type: Bound direction.

    Returns:
        Tuple ``(base, dual_base)`` aligned to the requested linear grid.
    """

    dual_realization = calc_pld_dual(realization)
    neg_dual_linear = negate_reverse_linear_distribution(dual_realization)

    if realization.x_gap < loss_discretization:
        rediscritized_realization = change_spacing_type(
            dist=realization,
            tail_truncation=tail_truncation,
            loss_discretization=loss_discretization,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )
        assert isinstance(rediscritized_realization, LinearDiscreteDist)
        rediscritized_dual = change_spacing_type(
            dist=neg_dual_linear,
            tail_truncation=tail_truncation,
            loss_discretization=loss_discretization,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )
        assert isinstance(rediscritized_dual, LinearDiscreteDist)
    else:
        rediscritized_realization = realization
        rediscritized_dual = neg_dual_linear
    return rediscritized_realization, rediscritized_dual


def realization_add_base_distribution(*,
    realization: PLDRealization,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Prepare add-direction factors from a loss-space realization.

    Algorithm 2 (`rand-alloc-add`) in Appendix C.

    Args:
        realization: ADD-direction realization in linear loss space.
        loss_discretization: Target linear-grid spacing.
        tail_truncation: Tail truncation budget for regridding.
        bound_type: Bound direction.

    Returns:
        One ADD loss factor aligned to the requested linear grid.
    """

    exp_bound_type = (
        BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    )
    neg_realization = negate_reverse_linear_distribution(realization)

    if neg_realization.x_gap < loss_discretization:
        neg_coarsened = change_spacing_type(
            dist=neg_realization,
            tail_truncation=tail_truncation,
            loss_discretization=loss_discretization,
            spacing_type=SpacingType.LINEAR,
            bound_type=exp_bound_type,
        )
        assert isinstance(neg_coarsened, LinearDiscreteDist)
    else:
        neg_coarsened = neg_realization
    return neg_coarsened

"""
Public API surface for random-allocation accounting.
"""

from __future__ import annotations

from functools import partial

from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.types import AllocationSchemeConfig, BoundType, ConvolutionMethod, Direction, PrivacyParams
from PLD_accounting.discrete_dist import PLDRealization
from PLD_accounting.adaptive_random_allocation import optimize_allocation_delta_range, optimize_allocation_epsilon_range
from PLD_accounting.random_allocation_accounting import allocation_PLD, geometric_allocation_PMF_base_remove, geometric_allocation_PMF_base_add
from PLD_accounting.random_allocation_gaussian import gaussian_allocation_PMF_core
from PLD_accounting.random_allocation_realization import realization_remove_base_distributions, realization_add_base_distribution


# =============================================================================
# Gaussian-Based Random Allocation API
# =============================================================================


def gaussian_allocation_epsilon_range(
    delta: float,
    sigma: float,
    num_steps: int,
    num_selected: int = 1,
    num_epochs: int = 1,
    epsilon_accuracy: float = -1.0,
) -> tuple[float, float]:
    """
    Compute epsilon upper and lower bounds of epsilon for random-allocation 
    with the Gaussian mechanism using adaptive refinement.

    Args:
        delta: Target delta for the epsilon query.
        sigma: Gaussian noise scale.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        epsilon_accuracy: Absolute convergence target on the best upper/lower epsilon gap. 
            Negative epsilon_accuracy means ~10% of the correct epsilon value.

    Returns:
        A tuple ``(upper_bound, lower_bound)``.
    """
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        delta=delta,
    )

    result = optimize_allocation_epsilon_range(
        params=params,
        target_accuracy=epsilon_accuracy,
        pld_builder=gaussian_allocation_PLD,
    )
    return result.upper_bound, result.lower_bound


def gaussian_allocation_delta_range(
    epsilon: float,
    sigma: float,
    num_steps: int,
    num_selected: int = 1,
    num_epochs: int = 1,
    delta_accuracy: float = -1.0,
) -> tuple[float, float]:
    """
    Compute epsilon upper and lower bounds of delta for random-allocation 
    with the Gaussian mechanism using adaptive refinement.

    Args:
        epsilon: Target epsilon for the delta query.
        sigma: Gaussian noise scale.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        delta_accuracy: Absolute convergence target on the best upper/lower
            delta gap. Negative delta_accuracy means ~10% of the correct delta
            value.

    Returns:
        A tuple ``(upper_bound, lower_bound)``.
    """
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        epsilon=epsilon,
    )

    result = optimize_allocation_delta_range(
        params=params,
        target_accuracy=delta_accuracy,
        pld_builder=gaussian_allocation_PLD,
    )
    return result.upper_bound, result.lower_bound


def gaussian_allocation_epsilon_extended(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """
    Compute epsilon upper / lower bounds of epsilon for random-allocation
    with the Gaussian mechanism with more control over the accuracy parameters.

    Args:
        params: Privacy parameters with ``params.delta`` set.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated bound.

    Returns:
        The epsilon value corresponding to ``params.delta``.
    """
    if params.delta is None:
        raise ValueError("gaussian_allocation_epsilon_extended requires params.delta to be set")
    if bound_type == BoundType.BOTH:
        raise ValueError(
            "Epsilon function does not support bound_type=BoundType.BOTH; "
            "use DOMINATES or IS_DOMINATED to get a single epsilon value, "
            "or use gaussian_allocation_epsilon_range for upper/lower bounds"
        )
    pld = gaussian_allocation_PLD(
        params=params,
        config=config,
        bound_type=bound_type,
    )
    return float(pld.get_epsilon_for_delta(params.delta))


def gaussian_allocation_delta_extended(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """
    Compute epsilon upper / lower bounds of delta for random-allocation
    with the Gaussian mechanism with more control over the accuracy parameters.

    Args:
        params: Privacy parameters with ``params.epsilon`` set.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated bound.

    Returns:
        The delta value corresponding to ``params.epsilon``.
    """
    if params.epsilon is None:
        raise ValueError("gaussian_allocation_delta_extended requires params.epsilon to be set")
    if bound_type == BoundType.BOTH:
        raise ValueError(
            "Delta function does not support bound_type=BoundType.BOTH; "
            "use DOMINATES or IS_DOMINATED to get a single delta value, "
            "or use gaussian_allocation_delta_range for upper/lower bounds"
        )
    pld = gaussian_allocation_PLD(
        params=params,
        config=config,
        bound_type=bound_type,
    )
    return float(pld.get_delta_for_epsilon(params.epsilon))

def gaussian_allocation_PLD(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """
    Compute upper / lower PLD for random-allocation with the Gaussian mechanism.

    Args:
        params: Privacy parameters describing noise scale, number of steps,
            and optional delta/epsilon query target.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated discretized bound.

    Returns:
        A ``dp_accounting`` ``PrivacyLossDistribution`` for both privacy directions.
    """

    compute_base_pmf_remove = partial(
        gaussian_allocation_PMF_core,
        direction=Direction.REMOVE,
        sigma=params.sigma,
        config=config,
    )
    compute_base_pmf_add = partial(
        gaussian_allocation_PMF_core,
        direction=Direction.ADD,
        sigma=params.sigma,
        config=config,
    )
    return allocation_PLD(
        compute_base_pmf_remove=compute_base_pmf_remove,
        compute_base_pmf_add=compute_base_pmf_add,
        num_steps=params.num_steps,
        num_selected=params.num_selected,
        num_epochs=params.num_epochs,
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )


# =============================================================================
# PLD Realization-Based Random Allocation API
# =============================================================================


def general_allocation_epsilon(
    delta: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    remove_realization: PLDRealization,
    add_realization: PLDRealization,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """
    Compute epsilon from explicit PLD realizations.

    Args:
        delta: Target delta for the epsilon query.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        remove_realization: Explicit remove-direction PLD realization.
        add_realization: Explicit add-direction PLD realization.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated bound.

    Returns:
        The epsilon value corresponding to the given delta.

    Notes:
        Supports only the GEOM convolution method.
   """
    if bound_type == BoundType.BOTH:
        raise ValueError(
            "Epsilon function does not support bound_type=BoundType.BOTH; "
            "use DOMINATES or IS_DOMINATED to get a single epsilon value"
        )
    pld = general_allocation_PLD(
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        remove_realization=remove_realization,
        add_realization=add_realization,
        config=config,
        bound_type=bound_type)
    return float(pld.get_epsilon_for_delta(delta))


def general_allocation_delta(
    epsilon: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    remove_realization: PLDRealization,
    add_realization: PLDRealization,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """
    Compute delta from explicit PLD realizations.

    Args:
        epsilon: Target epsilon for the delta query.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        remove_realization: Explicit remove-direction PLD realization.
        add_realization: Explicit add-direction PLD realization.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated bound.

    Returns:
        The delta value corresponding to the given epsilon.

    Notes:
        Supports only the GEOM convolution method.
    """
    if bound_type == BoundType.BOTH:
        raise ValueError(
            "Delta function does not support bound_type=BoundType.BOTH; "
            "use DOMINATES or IS_DOMINATED to get a single delta value"
        )
    pld = general_allocation_PLD(
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        remove_realization=remove_realization,
        add_realization=add_realization,
        config=config,
        bound_type=bound_type)
    return float(pld.get_delta_for_epsilon(epsilon))


def general_allocation_PLD(
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    remove_realization: PLDRealization,
    add_realization: PLDRealization,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """
    Build a random-allocation PLD from explicit PLD realizations.

    Args:
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        remove_realization: Explicit remove-direction PLD realization.
        add_realization: Explicit add-direction PLD realization.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated discretized bound.

    Returns:
        A ``dp_accounting`` ``PrivacyLossDistribution`` for the composed realization.

    Notes:
        Supports only the GEOM convolution method.
    """
    if not isinstance(remove_realization, PLDRealization):
        raise TypeError(
            f"general_allocation_PLD requires PLDRealization, got {type(remove_realization)}"
        )
    if not isinstance(add_realization, PLDRealization):
        raise TypeError(
            f"general_allocation_PLD requires PLDRealization, got {type(add_realization)}"
        )

    # Validate that geometric convolution is used for realization path
    if config.convolution_method != ConvolutionMethod.GEOM:
        raise ValueError(
            "PLD realization-based allocation requires geometric convolution. "
            f"Got convolution_method={config.convolution_method}. "
            "Use ConvolutionMethod.GEOM."
        )


    compute_base_pmf_remove = partial(
        geometric_allocation_PMF_base_remove,
        base_distributions_creation=partial(
            realization_remove_base_distributions,
            realization=remove_realization,
        )
    )
    compute_base_pmf_add = partial(
        geometric_allocation_PMF_base_add,
        base_distributions_creation=partial(
            realization_add_base_distribution,
            realization=add_realization,
        )
    )
    return allocation_PLD(
        compute_base_pmf_remove=compute_base_pmf_remove,
        compute_base_pmf_add=compute_base_pmf_add,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )

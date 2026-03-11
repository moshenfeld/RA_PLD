"""Public API surface for random-allocation accounting."""

from __future__ import annotations

from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.adaptive_random_allocation import (
    adaptive_delta_convergence,
    adaptive_epsilon_convergence,
)
from PLD_accounting.dp_accounting_support import discrete_dist_to_dp_accounting_pmf
from PLD_accounting.random_allocation_accounting import allocation_PMF, compute_conv_params
from PLD_accounting.types import AllocationSchemeConfig, BoundType, Direction, PrivacyParams


def allocation_PLD(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    direction: Direction = Direction.BOTH,
    bound_type: BoundType = BoundType.DOMINATES,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Build a random-allocation PLD.

    Args:
        params: Privacy parameters describing noise scale, number of steps,
            and optional delta/epsilon query target.
        config: Discretization and convolution configuration.
        direction: Which privacy direction to include in the output PLD.
        bound_type: Whether to compute a dominating or dominated discretized bound.

    Returns:
        A ``dp_accounting`` ``PrivacyLossDistribution`` for the requested direction.

    Notes:
        ``Direction.ADD`` alone is not supported here because a PLD object must
        always include the remove-direction PMF.
    """
    if direction == Direction.ADD:
        raise ValueError("PLD requires REMOVE direction PMF")
    if bound_type == BoundType.BOTH:
        raise ValueError(f"Allocation PLD does not support bound_type: {bound_type}")

    conv_params = compute_conv_params(params=params, config=config)
    pessimistic = bound_type == BoundType.DOMINATES

    remove_dist = allocation_PMF(
        conv_params=conv_params,
        direction=Direction.REMOVE,
        bound_type=bound_type,
        convolution_method=config.convolution_method,
    )
    pmf_remove = discrete_dist_to_dp_accounting_pmf(
        dist=remove_dist,
        pessimistic_estimate=pessimistic,
    )
    if direction == Direction.REMOVE:
        return privacy_loss_distribution.PrivacyLossDistribution(
            pmf_remove=pmf_remove,
        )

    add_dist = allocation_PMF(
        conv_params=conv_params,
        direction=Direction.ADD,
        bound_type=bound_type,
        convolution_method=config.convolution_method,
    )
    pmf_add = discrete_dist_to_dp_accounting_pmf(
        dist=add_dist,
        pessimistic_estimate=pessimistic,
    )
    return privacy_loss_distribution.PrivacyLossDistribution(
        pmf_remove=pmf_remove,
        pmf_add=pmf_add,
    )


def numerical_allocation_epsilon(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    direction: Direction = Direction.BOTH,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """Compute epsilon from a random-allocation PLD.

    Args:
        params: Privacy parameters with ``params.delta`` set.
        config: Discretization and convolution configuration.
        direction: Which privacy direction(s) to analyze.
        bound_type: Which discretization bound to use.

    Returns:
        The epsilon value corresponding to ``params.delta``.
    """
    if bound_type == BoundType.BOTH:
        raise ValueError(f"Delta function does not support bound_type: {bound_type}")
    return allocation_PLD(
        params=params,
        config=config,
        direction=direction,
        bound_type=bound_type,
    ).get_epsilon_for_delta(params.delta)


def numerical_allocation_delta(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    direction: Direction = Direction.BOTH,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """Compute delta from a random-allocation PLD.

    Args:
        params: Privacy parameters with ``params.epsilon`` set.
        config: Discretization and convolution configuration.
        direction: Which privacy direction(s) to analyze.
        bound_type: Which discretization bound to use.

    Returns:
        The delta value corresponding to ``params.epsilon``.
    """
    return allocation_PLD(
        params=params,
        config=config,
        direction=direction,
        bound_type=bound_type,
    ).get_delta_for_epsilon(params.epsilon)


def numerical_allocation_epsilon_range(
    delta: float,
    sigma: float,
    num_steps: int,
    num_selected: int = 1,
    num_epochs: int = 1,
    epsilon_accuracy: float = -1.0,
) -> tuple[float, float]:
    """Compute epsilon upper/lower bounds with adaptive random-allocation refinement.

    Args:
        delta: Target delta for the epsilon query.
        sigma: Gaussian noise scale.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        epsilon_accuracy: Absolute convergence target on the best upper/lower
            epsilon gap. Negative epsilon_accuracy means ~10% of the correct epsilon value

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

    result = adaptive_epsilon_convergence(
        params=params,
        target_accuracy=epsilon_accuracy,
        pld_builder=allocation_PLD,
    )
    return result.upper_bound, result.lower_bound


def numerical_allocation_delta_range(
    epsilon: float,
    sigma: float,
    num_steps: int,
    num_selected: int = 1,
    num_epochs: int = 1,
    delta_accuracy: float = -1.0,
) -> tuple[float, float]:
    """Compute delta upper/lower bounds with adaptive random-allocation refinement.

    Args:
        epsilon: Target epsilon for the delta query.
        sigma: Gaussian noise scale.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        delta_accuracy: Absolute convergence target on the best upper/lower
            delta gap. Negative delta_accuracy means ~10% of the correct delta value

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

    result = adaptive_delta_convergence(
        params=params,
        target_accuracy=delta_accuracy,
        pld_builder=allocation_PLD,
    )
    return result.upper_bound, result.lower_bound

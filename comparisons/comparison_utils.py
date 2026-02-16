"""Utilities for comparing with external packages (e.g., random_allocation).

This module provides converter functions to translate between this project's types
and external package types for benchmarking and comparison purposes.
"""

from PLD_accounting.types import PrivacyParams, Direction

from random_allocation.comparisons.structs import (
    PrivacyParams as PoissonPrivacyParams,
    Direction as PoissonDirection,
)


def to_poisson_params(params: PrivacyParams) -> 'PoissonPrivacyParams':
    """Convert project PrivacyParams to random_allocation PrivacyParams.

    Args:
        params: Project's PrivacyParams object

    Returns:
        Random allocation package's PrivacyParams object with same values

    Note:
        This is only used for comparison experiments with the random_allocation package.
    """
    return PoissonPrivacyParams(
        sigma=params.sigma,
        num_steps=params.num_steps,
        num_selected=params.num_selected,
        num_epochs=params.num_epochs,
        delta=params.delta,
        epsilon=params.epsilon
    )


def to_poisson_direction(direction: Direction) -> 'PoissonDirection':
    """Convert project Direction to random_allocation Direction.

    Args:
        direction: Project's Direction enum value

    Returns:
        Random allocation package's Direction enum value

    Raises:
        ValueError: If direction is not a valid Direction enum value

    Note:
        This is only used for comparison experiments with the random_allocation package.
    """
    mapping = {
        Direction.REMOVE: PoissonDirection.REMOVE,
        Direction.ADD: PoissonDirection.ADD,
        Direction.BOTH: PoissonDirection.BOTH
    }

    if direction not in mapping:
        raise ValueError(f"Invalid direction: {direction}")

    return mapping[direction]


def from_poisson_direction(direction: 'PoissonDirection') -> Direction:
    """Convert random_allocation Direction to project Direction.

    Args:
        direction: Random allocation package's Direction enum value

    Returns:
        Project's Direction enum value

    Raises:
        ValueError: If direction is not a valid Direction enum value

    Note:
        This is only used for comparison experiments with the random_allocation package.
    """
    mapping = {
        PoissonDirection.REMOVE: Direction.REMOVE,
        PoissonDirection.ADD: Direction.ADD,
        PoissonDirection.BOTH: Direction.BOTH
    }

    if direction not in mapping:
        raise ValueError(f"Invalid direction: {direction}")

    return mapping[direction]

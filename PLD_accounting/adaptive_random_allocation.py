"""Adaptive helpers for random-allocation queries."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

from PLD_accounting.types import AllocationSchemeConfig, BoundType, ConvolutionMethod, Direction, PrivacyParams
from dp_accounting.pld import privacy_loss_distribution


# Global configuration constants
MAX_ITERATIONS = 10
POISSON_GUESS_DISCRETIZATION = 1e-4
MIN_DISCRETIZATION = 1e-6
MAX_DISCRETIZATION = 1e-1
MIN_TAIL_TRUNCATION = 1e-16
MAX_TAIL_TRUNCATION = 1e-4


def _clip_discretization(value: float) -> float:
    return min(max(value, MIN_DISCRETIZATION), MAX_DISCRETIZATION)


def _clip_tail_truncation(value: float) -> float:
    return min(max(value, MIN_TAIL_TRUNCATION), MAX_TAIL_TRUNCATION)


@dataclass
class AdaptiveResult:
    """Result from adaptive allocation computation.

    Attributes:
        upper_bound: Best upper bound found across all iterations.
        lower_bound: Best lower bound found across all iterations.
        absolute_gap: Final gap between upper and lower bounds.
        converged: Whether the algorithm converged to target accuracy.
        iterations: Number of refinement iterations performed.
        initial_discretization: Starting loss_discretization value.
        discretization: Final loss_discretization value used.
        initial_tail_truncation: Starting tail_truncation value.
        tail_truncation: Final tail_truncation value used.
        target_accuracy: Target absolute gap for convergence.
    """
    upper_bound: float
    lower_bound: float
    absolute_gap: float
    converged: bool
    iterations: int
    initial_discretization: float
    discretization: float
    initial_tail_truncation: float
    tail_truncation: float
    target_accuracy: float


def estimate_poisson_query(
    params: PrivacyParams,
    query_func: Callable[[privacy_loss_distribution.PrivacyLossDistribution], float],
) -> float:
    """Estimate the query value with a Poisson-subsampled Gaussian approximation."""
    sampling_probability = params.num_selected / params.num_steps
    compose_steps = params.num_selected * params.num_epochs

    pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=params.sigma,
        sensitivity=1.0,
        value_discretization_interval=POISSON_GUESS_DISCRETIZATION,
        pessimistic_estimate=True,
        sampling_prob=sampling_probability,
    ).self_compose(compose_steps)

    estimate = float(query_func(pld))
    if not math.isfinite(estimate) or estimate < 0.0:
        raise RuntimeError(
            "Poisson-based adaptive initialization produced an invalid estimate: "
            f"{estimate!r}"
        )
    return estimate


def adaptive_convergence(
    params: PrivacyParams,
    target_accuracy: float,
    initial_discretization: float,
    initial_tail_truncation: float,
    query_func: Callable[[privacy_loss_distribution.PrivacyLossDistribution], float],
    discretization_gap_func: Callable[
        [privacy_loss_distribution.PrivacyLossDistribution, float, float, float], float
    ],
    tail_gap_func: Callable[
        [privacy_loss_distribution.PrivacyLossDistribution, float, float, float], float
    ],
    pld_builder: Callable[..., privacy_loss_distribution.PrivacyLossDistribution],
) -> AdaptiveResult:
    """Core adaptive refinement loop.

    Args:
        params: Privacy parameters
        target_accuracy: Target absolute gap between bounds
        query_func: Function to extract the scalar query value from the computed PLD.
        discretization_gap_func: Estimates the current query gap contribution due
            to loss discretization using the upper PLD and current upper/lower values.
        tail_gap_func: Estimates the current query gap contribution due to tail
            truncation using the upper PLD and current upper/lower values.
        pld_builder: Callable that constructs the PLD for a given config/bound pair.

    Returns:
        AdaptiveResult with convergence metadata
    """
    discretization = _clip_discretization(initial_discretization)
    tail_truncation = _clip_tail_truncation(initial_tail_truncation)
    effective_initial_discretization = discretization
    effective_initial_tail_truncation = tail_truncation

    upper_bound = np.inf
    lower_bound = -np.inf
    converged = False
    iteration = 0
    previous_discretization: float | None = None
    previous_tail_truncation: float | None = None
    unchanged_discretization_iters = 0
    unchanged_tail_iters = 0

    while not converged and iteration < MAX_ITERATIONS:
        iteration += 1

        if previous_discretization == discretization:
            unchanged_discretization_iters += 1
        else:
            unchanged_discretization_iters = 1

        if previous_tail_truncation == tail_truncation:
            unchanged_tail_iters += 1
        else:
            unchanged_tail_iters = 1

        config = AllocationSchemeConfig(
            loss_discretization=discretization,
            tail_truncation=tail_truncation,
            convolution_method=ConvolutionMethod.GEOM,
        )

        # Compute upper bound (DOMINATES)
        pld_upper = pld_builder(
            params=params,
            config=config,
            direction=Direction.BOTH,
            bound_type=BoundType.DOMINATES,
        )
        new_upper = query_func(pld_upper)

        # Compute lower bound (IS_DOMINATED)
        pld_lower = pld_builder(
            params=params,
            config=config,
            direction=Direction.BOTH,
            bound_type=BoundType.IS_DOMINATED,
        )
        new_lower = query_func(pld_lower)

        if new_upper < new_lower:
            raise RuntimeError(
                "Adaptive refinement produced invalid bounds: dominating bound "
                f"{new_upper:.12g} is below dominated bound {new_lower:.12g}"
            )

        upper_bound = min(upper_bound, new_upper)
        lower_bound = max(lower_bound, new_lower)

        if upper_bound < lower_bound:
            raise RuntimeError(
                "Adaptive refinement produced invalid bounds: dominating bound "
                f"{upper_bound:.12g} is below dominated bound {lower_bound:.12g}"
            )

        # Check convergence
        gap = upper_bound - lower_bound
        if gap < target_accuracy:
            converged = True
            break

        # Compare two query-level gap estimates to decide which approximation knob
        # is currently dominating the observed upper/lower bound separation.
        discretization_gap = float(
            discretization_gap_func(pld_upper, new_upper, new_lower, discretization)
        )
        if not math.isfinite(discretization_gap) or discretization_gap < 0.0:
            raise RuntimeError(
                "Adaptive refinement produced an invalid discretization-gap estimate: "
                f"{discretization_gap!r}"
            )

        tail_gap = float(tail_gap_func(pld_upper, new_upper, new_lower, tail_truncation))
        if not math.isfinite(tail_gap) or tail_gap < 0.0:
            raise RuntimeError(
                "Adaptive refinement produced an invalid tail-gap estimate: "
                f"{tail_gap!r}"
            )

        next_discretization = _clip_discretization(discretization / 2)
        next_tail_truncation = _clip_tail_truncation(tail_truncation / 10)
        can_change_discretization = next_discretization != discretization
        can_change_tail = next_tail_truncation != tail_truncation

        force_discretization = unchanged_discretization_iters >= 3 and can_change_discretization
        force_tail = unchanged_tail_iters >= 3 and can_change_tail

        # Refine the source with the larger estimated contribution to the gap.
        previous_discretization = discretization
        previous_tail_truncation = tail_truncation

        if force_discretization and not force_tail:
            discretization = next_discretization
        elif force_tail and not force_discretization:
            tail_truncation = next_tail_truncation
        elif force_discretization and force_tail:
            if discretization_gap >= tail_gap and can_change_discretization:
                discretization = next_discretization
            elif can_change_tail:
                tail_truncation = next_tail_truncation
        elif discretization_gap > tail_gap:
            discretization = next_discretization
        else:
            tail_truncation = next_tail_truncation

    # Warn if not converged
    if not converged:
        warnings.warn(
            f"Adaptive refinement did not converge after {MAX_ITERATIONS} iterations. "
            f"Final gap: {upper_bound - lower_bound:.6e}, target: {target_accuracy:.6e}. "
            f"Returning best bounds found.",
            RuntimeWarning,
        )

    return AdaptiveResult(
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        absolute_gap=upper_bound - lower_bound,
        converged=converged,
        iterations=iteration,
        initial_discretization=effective_initial_discretization,
        discretization=discretization,
        initial_tail_truncation=effective_initial_tail_truncation,
        tail_truncation=tail_truncation,
        target_accuracy=target_accuracy,
    )

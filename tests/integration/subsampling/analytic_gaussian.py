"""
Analytical computation of subsampled Gaussian mechanism.

Provides ground-truth implementation for testing subsampling against
first principles, using closed-form formulas for privacy loss distribution.
"""
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from typing import Tuple

def stable_subsampling_loss(
    losses: NDArray[np.float64],
    sampling_prob: float,
    remove_direction: bool = True
) -> NDArray[np.float64]:
    if sampling_prob <= 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in (0, 1]")
    if sampling_prob == 1:
        return losses

    new_losses = np.zeros_like(losses)
    if not remove_direction:
        losses = -losses.copy()

    undefined_threshold = np.log(1 - sampling_prob) if sampling_prob < 1.0 else -np.inf
    undefined_ind = losses <= undefined_threshold
    new_losses[undefined_ind] = -np.inf

    small_loss_ind = ~undefined_ind & (losses < sampling_prob)
    new_losses[small_loss_ind] = np.log1p(np.expm1(losses[small_loss_ind]) / sampling_prob)

    medium_loss_ind = ~undefined_ind & ~small_loss_ind & (losses > 1)
    new_losses[medium_loss_ind] = np.log(1 + np.expm1(losses[medium_loss_ind]) / sampling_prob)

    large_loss_ind = ~undefined_ind & ~small_loss_ind & ~medium_loss_ind
    new_losses[large_loss_ind] = losses[large_loss_ind] - np.log(sampling_prob) \
        + np.log1p((sampling_prob - 1) * np.exp(-losses[large_loss_ind]))

    if not remove_direction:
        new_losses = -new_losses
    return new_losses


def gaussian_pld(
    sigma: float,
    sampling_prob: float,
    discretization: float,
    remove_direction: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute privacy loss distribution for subsampled Gaussian analytically.

    Uses closed-form formulas to compute the exact PLD for a Gaussian mechanism
    with subsampling. This serves as ground truth for validating numerical
    implementations.

    Args:
        sigma: Standard deviation of Gaussian noise
        sampling_prob: Subsampling probability (q)
        discretization: Grid spacing for privacy loss values
        remove_direction: If True, compute for REMOVE direction; else ADD

    Returns:
        Tuple of (losses, probabilities) arrays

    Raises:
        ValueError: If parameters are invalid
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not (0 < sampling_prob <= 1):
        raise ValueError("sampling_prob (q) must be in (0, 1]")
    if discretization <= 0:
        raise ValueError("discretization must be positive")

    # Create loss grid with sufficient range
    l_max = np.ceil(20.0 / (sigma * discretization)) * discretization
    losses = np.arange(-l_max, l_max + discretization, discretization)

    # Transform losses according to subsampling formula (reference)
    transformed_losses = stable_subsampling_loss(losses, sampling_prob, remove_direction)

    # Compute CCDF values using analytical formula
    # For Gaussian mechanism: P(L >= l) where L is privacy loss
    x_upper = sigma * transformed_losses - 0.5 / sigma
    x_lower = sigma * transformed_losses + 0.5 / sigma

    # Compute survival function (CCDF)
    S = np.ones_like(losses)
    if remove_direction:
        # Remove direction: mixture of base and neighbor distributions
        S = (1.0 - sampling_prob) * stats.norm.sf(x_lower) + \
            sampling_prob * stats.norm.sf(x_upper)
    else: # ADD
        # Add direction: only base distribution
        S = stats.norm.sf(x_upper)

    # Convert CCDF to PMF: P(L = l_i) = S[i-1] - S[i]
    probs = np.concatenate(([1.0], S[:-1])) - S

    # Validation checks
    if np.any(S < 0) or np.any(S > 1):
        raise ValueError(
            "CCDF out of [0,1] in subsampled_gaussian_probabilities_from_losses"
        )
    if np.any(probs < -1e-15):
        raise ValueError(
            "Negative probability in subsampled_gaussian_probabilities_from_losses"
        )
    if np.sum(probs) > 1 + 1e-12:
        raise ValueError(
            "sum(probs) > 1 in subsampled_gaussian_probabilities_from_losses"
        )
    if np.size(probs) != np.size(losses):
        raise ValueError("Length mismatch between losses and probs")

    return losses, probs


def gaussian_delta_from_epsilon(
    sigma: float,
    sampling_prob: float,
    epsilon: float,
    remove_direction: bool
) -> float:
    """Compute delta for given epsilon using analytical formula.

    Computes the delta parameter for (epsilon, delta)-DP given epsilon,
    using the analytical relationship for subsampled Gaussian mechanisms.

    Args:
        sigma: Standard deviation of Gaussian noise
        sampling_prob: Subsampling probability (q)
        epsilon: Privacy parameter epsilon
        remove_direction: If True, compute for REMOVE direction; else ADD

    Returns:
        Delta value achieving (epsilon, delta)-DP

    Raises:
        ValueError: If parameters are invalid
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not (0 < sampling_prob <= 1):
        raise ValueError("sampling_prob (q) must be in (0, 1]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    # Base case: no subsampling
    if sampling_prob == 1.0:
        return stats.norm.cdf(0.5 / sigma - epsilon * sigma) - \
               np.exp(epsilon) * stats.norm.cdf(-0.5 / sigma - epsilon * sigma)

    # Remove direction: amplification via subsampling
    if remove_direction:
        # Compute amplified epsilon
        amplified_epsilon = np.log(1.0 + (np.exp(epsilon) - 1.0) / sampling_prob)
        # Recurse with no subsampling
        return sampling_prob * gaussian_delta_from_epsilon(
            sigma, 1.0, amplified_epsilon, True
        )

    # Add direction: no amplification if epsilon too small
    if epsilon >= -np.log(1 - sampling_prob):
        return 0.0

    # Add direction with amplification
    amplified_epsilon = -np.log(1.0 + (np.exp(-epsilon) - 1.0) / sampling_prob)
    return (1.0 - np.exp(epsilon) * (1.0 - sampling_prob)) * \
           gaussian_delta_from_epsilon(sigma, 1.0, amplified_epsilon, False)


def gaussian_epsilon_for_delta(
    sigma: float,
    sampling_prob: float,
    delta: float,
    remove_direction: bool
) -> float:
    """Compute epsilon for given delta using binary search.

    Inverts the epsilon->delta relationship to find the minimum epsilon
    achieving (epsilon, delta)-DP.

    Args:
        sigma: Standard deviation of Gaussian noise
        sampling_prob: Subsampling probability (q)
        delta: Privacy parameter delta
        remove_direction: If True, compute for REMOVE direction; else ADD

    Returns:
        Minimum epsilon achieving (epsilon, delta)-DP
    """
    def delta_for_eps(eps: float) -> float:
        return gaussian_delta_from_epsilon(
            sigma, sampling_prob, eps, remove_direction
        )

    # Binary search for epsilon
    eps_low = 0.0
    eps_high = 100.0

    # Check if epsilon exists in search range
    if delta_for_eps(eps_high) > delta:
        return float('inf')

    # Binary search with tolerance 1e-6
    while eps_high - eps_low > 1e-6:
        eps_mid = (eps_low + eps_high) / 2.0
        if delta_for_eps(eps_mid) <= delta:
            eps_high = eps_mid
        else:
            eps_low = eps_mid

    return eps_high

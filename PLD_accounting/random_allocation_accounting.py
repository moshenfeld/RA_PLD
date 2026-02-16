from scipy import stats
from scipy.stats import lognorm, norm
import numpy as np

from dp_accounting.pld import privacy_loss_distribution

from dataclasses import dataclass

from PLD_accounting.types import PrivacyParams, AllocationSchemeConfig, Direction, ConvolutionMethod, BoundType, SpacingType
from PLD_accounting.discrete_dist import DiscreteDist
from PLD_accounting.core_utils import compute_bin_width
from PLD_accounting.distribution_discretization import (
    MIN_GRID_SIZE,
    change_spacing_type,
    discretize_continuous_distribution,
    discretize_continuous_to_pmf,
)
from PLD_accounting.utils import combine_distributions
from PLD_accounting.convolution_API import convolve_discrete_distributions, self_convolve_discrete_distributions
from PLD_accounting.dp_accounting_support import discrete_dist_to_dp_accounting_pmf

# =============================================================================
# Helper dataclass
# =============================================================================
@dataclass
class _ConvParams:
    num_steps: int
    compose_steps: int
    sigma: float
    output_tail_truncation: float
    pre_composition_tail_truncation: float
    discretization_tail_truncation: float
    n_grid_FFT: int
    n_grid_geom: int
    output_loss_discretization: float
    pre_composition_loss_discretization: float
    discretization_loss_discretization: float 
    max_grid_FFT: int

def _compute_conv_params(
    params: PrivacyParams,
    config: AllocationSchemeConfig
) -> _ConvParams:
    """Compute convolution parameters from privacy params and scheme config."""
    if config.max_grid_FFT <= 0:
        raise ValueError("max_grid_FFT must be positive for FFT convolution")

    num_steps_per_round = int(np.floor(params.num_steps / params.num_selected))
    if num_steps_per_round < 1:
        raise ValueError("num_steps must be >= num_selected to form at least one step per round")
    sigma_inv = 1.0 / params.sigma

    output_tail_truncation = config.tail_truncation / 3
    pre_composition_tail_truncation = output_tail_truncation / (params.num_selected * params.num_epochs)
    discretization_tail_truncation = pre_composition_tail_truncation / num_steps_per_round
    # Avoid underflow in logcdf/logsf that can collapse all finite mass to infinity.
    discretization_tail_truncation = max(discretization_tail_truncation, np.finfo(float).eps * 1e-10)
    
    output_loss_discretization = config.loss_discretization / 3
    pre_composition_loss_discretization = output_loss_discretization / np.sqrt(params.num_selected * params.num_epochs)
    discretization_loss_discretization = pre_composition_loss_discretization / (2 * np.ceil(np.log2(num_steps_per_round)) + 1)

    log_range = -stats.norm.ppf(discretization_tail_truncation / 2) * sigma_inv
    n_grid_FFT = int(np.ceil(config.max_grid_FFT / num_steps_per_round))
    n_grid_geom = max(int(2 * log_range / discretization_loss_discretization), MIN_GRID_SIZE)
    if config.max_grid_mult > 0 and n_grid_geom > config.max_grid_mult:
        n_grid_geom = config.max_grid_mult

    return _ConvParams(
        num_steps=num_steps_per_round,
        compose_steps=params.num_epochs * params.num_selected,
        sigma=sigma_inv,
        output_tail_truncation=output_tail_truncation,
        pre_composition_tail_truncation=pre_composition_tail_truncation,
        discretization_tail_truncation=discretization_tail_truncation,
        n_grid_FFT=n_grid_FFT,
        n_grid_geom=n_grid_geom,
        output_loss_discretization=output_loss_discretization,
        pre_composition_loss_discretization=pre_composition_loss_discretization,
        discretization_loss_discretization=discretization_loss_discretization,
        max_grid_FFT=config.max_grid_FFT,
    )

# =============================================================================
# Computation of random variables dominating / dominated by the PLD 
# in the add / remove direction using FFT / geometric method
# =============================================================================

def _allocation_PMF_remove(conv_params: _ConvParams,
                           bound_type: BoundType,
                           convolution_method: ConvolutionMethod
                           ) -> DiscreteDist:
    """Compute allocation PLD for REMOVE direction using the selected convolution method.

    Discretizes the log-normal factors on linear (FFT) or geometric grids and
    composes them with the corresponding convolution kernel.
    """
    num_steps = conv_params.num_steps
    if num_steps < 2:
        raise ValueError("REMOVE direction requires at least two steps per round")
    sigma = conv_params.sigma
    discretization_tail_truncation = conv_params.discretization_tail_truncation / 2
    pre_composition_tail_truncation = conv_params.pre_composition_tail_truncation / 3

    lower_norm_mean = -sigma**2 / 2 - np.log(num_steps)
    upper_norm_mean = sigma**2 / 2 - np.log(num_steps)
    exp_L_QP_neg = stats.lognorm(s=sigma, scale=np.exp(lower_norm_mean))
    exp_L_PQ = stats.lognorm(s=sigma, scale=np.exp(upper_norm_mean))
    if convolution_method == ConvolutionMethod.FFT:
        n_grid = conv_params.n_grid_FFT
        spacing_type = SpacingType.LINEAR
        lower_shift = np.exp(lower_norm_mean + sigma**2 / 2)
        upper_shift = np.exp(upper_norm_mean + sigma**2 / 2)
    elif convolution_method == ConvolutionMethod.GEOM:
        n_grid = conv_params.n_grid_geom
        spacing_type = SpacingType.GEOMETRIC
        lower_shift = 0.0
        upper_shift = 0.0
    else:
        raise ValueError(f"Invalid convolution_method: {convolution_method}")
    # Compute the random variable dominating e^{-L_{Q,P}} for upper bound,
    # dominated by e^{-L_{Q,P}} for lower bound
    base_dist_lower = discretize_continuous_distribution(
        dist=exp_L_QP_neg,
        tail_truncation=discretization_tail_truncation,
        bound_type=bound_type,
        spacing_type=spacing_type,
        n_grid=n_grid,
        align_to_multiples=(convolution_method == ConvolutionMethod.GEOM)
    )
    base_dist_lower.x_array -= lower_shift
    # Self convolve it T-1 times
    conv_dist_lower = self_convolve_discrete_distributions(dist=base_dist_lower,
                                                           T=num_steps - 1,
                                                           tail_truncation=pre_composition_tail_truncation,
                                                           bound_type=bound_type,
                                                           convolution_method=convolution_method)
    # Compute the random variable dominating e^{L_{P,Q}} for upper bound,
    # dominated by e^{L_{P,Q}} for lower bound
    if spacing_type == SpacingType.GEOMETRIC:
        base_dist_upper = discretize_continuous_distribution(
            dist=exp_L_PQ,
            tail_truncation=discretization_tail_truncation,
            bound_type=bound_type,
            spacing_type=spacing_type,
            n_grid=n_grid,
            align_to_multiples=(convolution_method == ConvolutionMethod.GEOM)
        )
    else:
        upper_grid = conv_dist_lower.x_array + upper_shift
        x_max_target = exp_L_PQ.isf(discretization_tail_truncation)
        p_right = exp_L_PQ.sf(upper_grid[-1])
        p_right_threshold = conv_params.output_tail_truncation / 10
        if np.isfinite(x_max_target) and upper_grid[-1] < x_max_target and p_right > p_right_threshold:
            if upper_grid.size > 1:
                # Extend the linear grid (FFT path) to cover the target right tail.
                step = compute_bin_width(upper_grid)
                n_extra = int(np.ceil((x_max_target - upper_grid[-1]) / step))
                if n_extra > 0:
                    upper_grid = np.concatenate(
                        [upper_grid, upper_grid[-1] + step * np.arange(1, n_extra + 1)]
                    )
        base_dist_upper = discretize_continuous_to_pmf(
            dist=exp_L_PQ,
            x_array=upper_grid,
            bound_type=bound_type,
            PMF_min_increment=discretization_tail_truncation
        )
        base_dist_upper.x_array -= upper_shift
    # Convolve it with the T-1 self convolution of the bound on e^{-L_{Q,P}}
    conv_dist = convolve_discrete_distributions(dist_1=conv_dist_lower,
                                                dist_2=base_dist_upper,
                                                tail_truncation=pre_composition_tail_truncation,
                                                bound_type=bound_type,
                                                convolution_method=convolution_method)
    if convolution_method == ConvolutionMethod.FFT:
        conv_dist.x_array += (num_steps - 1) * lower_shift + upper_shift
    # Convert to losses: ln(x_array/t)
    conv_dist.x_array = np.log(conv_dist.x_array)
    return conv_dist

def _allocation_PMF_add(conv_params: _ConvParams,
                        bound_type: BoundType,
                        convolution_method: ConvolutionMethod
                        ) -> DiscreteDist:
    """Compute allocation PLD for ADD direction using the selected convolution method.

    Builds a bound on e^{-L_{Q,P}}, composes it T times, then flips coordinates
    to express losses for the ADD neighbor relation.
    """
    if convolution_method == ConvolutionMethod.GEOM:
        n_grid = conv_params.n_grid_geom
        spacing_type = SpacingType.GEOMETRIC
    elif convolution_method == ConvolutionMethod.FFT:
        n_grid = conv_params.n_grid_FFT
        spacing_type = SpacingType.LINEAR
    else:    
        raise ValueError(f"Invalid convolution_method: {convolution_method}")

    num_steps = conv_params.num_steps
    sigma = conv_params.sigma
    discretization_tail_truncation = conv_params.discretization_tail_truncation
    pre_composition_tail_truncation = conv_params.pre_composition_tail_truncation / 2
    # Compute the random variable dominated by e^{-L_{Q,P}} for upper bound,
    # dominating e^{-L_{Q,P}} for lower bound,
    opposite_bound_type = BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    norm_mean = -sigma**2 / 2 - np.log(num_steps)
    base_dist = discretize_continuous_distribution(dist=stats.lognorm(s=sigma, scale=np.exp(norm_mean)),
                                                   tail_truncation=discretization_tail_truncation,
                                                   bound_type=opposite_bound_type,
                                                   spacing_type=spacing_type,
                                                   n_grid=n_grid,
                                                   align_to_multiples=(convolution_method == ConvolutionMethod.GEOM))
    # Self convolve it T times
    conv_dist = self_convolve_discrete_distributions(dist=base_dist,
                                                     T=num_steps,
                                                     tail_truncation=pre_composition_tail_truncation,
                                                     bound_type=opposite_bound_type,
                                                     convolution_method=convolution_method)
    # Convert to losses: -ln(x_array/t)
    return DiscreteDist(
        x_array=-np.flip(np.log(conv_dist.x_array)),
        PMF_array=np.flip(conv_dist.PMF_array),
        p_neg_inf=conv_dist.p_pos_inf,
        p_pos_inf=conv_dist.p_neg_inf
    )


def _allocation_PMF(conv_params: _ConvParams,
                    direction: Direction,
                    bound_type: BoundType,
                    convolution_method: ConvolutionMethod
                    ) -> DiscreteDist:
    """Compute ADD/REMOVE direction distribution using FFT, geometric, or combined bounds."""
    if direction == Direction.ADD:
        conv_func = _allocation_PMF_add
        if convolution_method == ConvolutionMethod.COMBINED:
            convolution_method = ConvolutionMethod.GEOM
    elif direction == Direction.REMOVE:
        conv_func = _allocation_PMF_remove
        if convolution_method == ConvolutionMethod.COMBINED:
            convolution_method = ConvolutionMethod.FFT
    else:
        raise ValueError(f"Invalid direction: {direction}")

    if convolution_method in (ConvolutionMethod.FFT, ConvolutionMethod.BEST_OF_TWO):
        FFT_based_dist = conv_func(conv_params=conv_params,
                                   bound_type=bound_type,
                                   convolution_method=ConvolutionMethod.FFT)
    if convolution_method in (ConvolutionMethod.GEOM, ConvolutionMethod.BEST_OF_TWO):
        geom_based_dist = conv_func(conv_params=conv_params,
                                    bound_type=bound_type,
                                    convolution_method=ConvolutionMethod.GEOM)

    if convolution_method == ConvolutionMethod.BEST_OF_TWO:
        result_dist = combine_distributions(FFT_based_dist, geom_based_dist, bound_type=bound_type)
    elif convolution_method == ConvolutionMethod.GEOM:
        result_dist = geom_based_dist
    elif convolution_method == ConvolutionMethod.FFT:
        result_dist = FFT_based_dist
    else:
        raise ValueError(f"Invalid convolution_method: {convolution_method}")

    result_dist = change_spacing_type(
        dist=result_dist,
        tail_truncation=conv_params.pre_composition_tail_truncation,
        loss_discretization=conv_params.pre_composition_loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    result_dist = self_convolve_discrete_distributions(
        dist=result_dist,
        T=conv_params.compose_steps,
        tail_truncation=conv_params.output_tail_truncation,
        bound_type=bound_type,
        convolution_method=ConvolutionMethod.FFT,
    )
    return change_spacing_type(
        dist=result_dist,
        tail_truncation=conv_params.output_tail_truncation,
        loss_discretization=conv_params.output_loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )

# =============================================================================
# Combined functions
# =============================================================================

def allocation_PLD(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    direction: Direction = Direction.BOTH,
    bound_type: BoundType = BoundType.DOMINATES,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Compute composed allocation PLD using project convolution and dp_accounting output format.
    """
    if direction == Direction.ADD:
        raise ValueError("PLD requires REMOVE direction PMF")
    if bound_type == BoundType.BOTH:
        raise ValueError(f"Allocation PLD does not support bound_type: {bound_type}")

    conv_params = _compute_conv_params(params=params, config=config)

    pessimistic = (bound_type == BoundType.DOMINATES)

    pmf_remove = discrete_dist_to_dp_accounting_pmf(
        dist=_allocation_PMF(
            conv_params=conv_params,
            direction=Direction.REMOVE,
            bound_type=bound_type,
            convolution_method=config.convolution_method,
        ),
        pessimistic_estimate=pessimistic,
    )
    if direction == Direction.REMOVE:
        return privacy_loss_distribution.PrivacyLossDistribution(
            pmf_remove=pmf_remove,
        )
    pmf_add = discrete_dist_to_dp_accounting_pmf(
        dist=_allocation_PMF(
            conv_params=conv_params,
            direction=Direction.ADD,
            bound_type=bound_type,
            convolution_method=config.convolution_method,
        ),
        pessimistic_estimate=pessimistic,
    )
    return privacy_loss_distribution.PrivacyLossDistribution(
        pmf_remove=pmf_remove,
        pmf_add=pmf_add,
    )


# =============================================================================
# Convenience functions for epsilon/delta queries
# =============================================================================

def numerical_allocation_epsilon(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    direction: Direction = Direction.BOTH,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """Compute epsilon for allocation scheme given delta.

    Convenience wrapper around allocation_PLD that returns epsilon directly.

    """
    if bound_type == BoundType.BOTH:
        raise ValueError(f"Delta function does not support bound_type: {bound_type}")
    return allocation_PLD(
        params=params,
        config=config,
        direction=direction,
        bound_type=bound_type
    ).get_epsilon_for_delta(params.delta)


def numerical_allocation_delta(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    direction: Direction = Direction.BOTH,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """Compute delta for allocation scheme given epsilon.

    Convenience wrapper around allocation_PLD that returns delta directly.
    """
    return allocation_PLD(
        params=params,
        config=config,
        direction=direction,
        bound_type=bound_type
    ).get_delta_for_epsilon(params.epsilon)

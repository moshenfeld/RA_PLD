from scipy import stats
import numpy as np

from dataclasses import dataclass

from PLD_accounting.types import *
from PLD_accounting.utils import *
from PLD_accounting.core_utils import compute_bin_width, stable_isclose
from PLD_accounting.distribution_discretization import *
from PLD_accounting.utils import combine_distributions
from PLD_accounting.geometric_convolution import geometric_convolve, geometric_self_convolve
from PLD_accounting.FFT_convolution import FFT_convolve, FFT_self_convolve
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


def compute_conv_params(
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
    discretization_tail_truncation = max(
        float(discretization_tail_truncation),
        float(np.finfo(float).eps * 1e-10),
    )
    
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

def _build_shared_geometric_grid(
    dist: stats.rv_continuous,
    tail_truncation: float,
    log_step: float,
) -> np.ndarray:
    """Build a geometric grid snapped to a shared log-space lattice."""
    x_min = dist.ppf(tail_truncation)
    x_max = dist.isf(tail_truncation)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError(f"Quantiles not finite: x_min={x_min}, x_max={x_max}")
    if x_min <= 0.0:
        raise ValueError(f"Geometric spacing requires positive values, got x_min={x_min}, x_max={x_max}")
    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min, got x_min={x_min}, x_max={x_max}")

    start_idx = int(np.floor(np.log(x_min) / log_step))
    stop_idx = int(np.ceil(np.log(x_max) / log_step))
    x_array = np.exp(log_step * np.arange(start_idx, stop_idx + 1, dtype=np.float64))

    support_min, support_max = dist.support()
    if np.isfinite(support_min):
        x_array = x_array[x_array > support_min]
    if np.isfinite(support_max):
        x_array = x_array[x_array < support_max]
    return x_array


def _allocation_PMF_remove_fft(
    conv_params: _ConvParams,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """Compute REMOVE-direction allocation PMF using the FFT backend only."""
    num_steps = conv_params.num_steps
    if num_steps < 2:
        raise ValueError("REMOVE direction requires at least two steps per round")

    sigma = conv_params.sigma
    discretization_tail_truncation = conv_params.discretization_tail_truncation / 2
    pre_composition_tail_truncation = conv_params.pre_composition_tail_truncation / 3

    lower_norm_mean = -sigma**2 / 2 - np.log(num_steps)
    upper_norm_mean = sigma**2 / 2 - np.log(num_steps)
    lower_shift = np.exp(lower_norm_mean + sigma**2 / 2)
    upper_shift = np.exp(upper_norm_mean + sigma**2 / 2)

    exp_L_QP_neg = stats.lognorm(s=sigma, scale=np.exp(lower_norm_mean))
    base_dist_lower = discretize_continuous_distribution(
        dist=exp_L_QP_neg,
        tail_truncation=discretization_tail_truncation,
        bound_type=bound_type,
        spacing_type=SpacingType.LINEAR,
        n_grid=conv_params.n_grid_FFT,
        align_to_multiples=False,
    )
    assert isinstance(base_dist_lower, LinearDiscreteDist)
    shifted_lower = shift_distribution(base_dist_lower, -lower_shift)
    assert isinstance(shifted_lower, LinearDiscreteDist)
    conv_dist_lower = FFT_self_convolve(
        dist=shifted_lower,
        T=num_steps - 1,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=bound_type,
        use_direct=True,
    )

    exp_L_PQ = stats.lognorm(s=sigma, scale=np.exp(upper_norm_mean))
    upper_grid = conv_dist_lower.x_array + upper_shift
    x_max_target = exp_L_PQ.isf(discretization_tail_truncation)
    p_right = exp_L_PQ.sf(upper_grid[-1])
    p_right_threshold = conv_params.output_tail_truncation / 10
    if np.isfinite(x_max_target) and upper_grid[-1] < x_max_target and p_right > p_right_threshold:
        if upper_grid.size > 1:
            # Extend the FFT grid enough to capture the right tail before shifting back.
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
        PMF_min_increment=discretization_tail_truncation,
        spacing_type=SpacingType.LINEAR,
    )
    shifted_upper = shift_distribution(base_dist_upper, -upper_shift)
    assert isinstance(shifted_upper, LinearDiscreteDist)

    conv_dist_raw = FFT_convolve(
        dist_1=conv_dist_lower,
        dist_2=shifted_upper,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=bound_type,
    )
    # FFT produces LinearDiscreteDist in exp-space; regrid to geometric then apply log
    shifted_exp = shift_distribution(
        conv_dist_raw,
        (num_steps - 1) * lower_shift + upper_shift,
    )
    exp_geom = change_spacing_type(
        dist=shifted_exp,
        tail_truncation=0.0,
        loss_discretization=conv_params.pre_composition_loss_discretization,
        spacing_type=SpacingType.GEOMETRIC,
        bound_type=bound_type,
    )
    assert isinstance(exp_geom, GeometricDiscreteDist)
    return log_geometric_to_linear(exp_geom)


def _allocation_PMF_remove_geom(
    conv_params: _ConvParams,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """Compute REMOVE-direction allocation PMF using the geometric backend only."""
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

    lower_x_min = exp_L_QP_neg.ppf(discretization_tail_truncation)
    lower_x_max = exp_L_QP_neg.isf(discretization_tail_truncation)
    upper_x_min = exp_L_PQ.ppf(discretization_tail_truncation)
    upper_x_max = exp_L_PQ.isf(discretization_tail_truncation)
    log_span = max(
        np.log(lower_x_max / lower_x_min),
        np.log(upper_x_max / upper_x_min),
    )
    shared_log_step = log_span / (conv_params.n_grid_geom - 1)

    # Build both remove-direction factors on the same geometric lattice. Independent
    # endpoint alignment can add an extra interval to only one side, which changes
    # its ratio and makes the downstream geometric convolution invalid.
    base_dist_lower = discretize_continuous_to_pmf(
        dist=exp_L_QP_neg,
        x_array=_build_shared_geometric_grid(
            dist=exp_L_QP_neg,
            tail_truncation=discretization_tail_truncation,
            log_step=shared_log_step,
        ),
        bound_type=bound_type,
        PMF_min_increment=discretization_tail_truncation,
        spacing_type=SpacingType.GEOMETRIC,
    )
    assert isinstance(base_dist_lower, GeometricDiscreteDist)
    conv_dist_lower = geometric_self_convolve(
        dist=base_dist_lower,
        T=num_steps - 1,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=bound_type,
    )

    base_dist_upper = discretize_continuous_to_pmf(
        dist=exp_L_PQ,
        x_array=_build_shared_geometric_grid(
            dist=exp_L_PQ,
            tail_truncation=discretization_tail_truncation,
            log_step=shared_log_step,
        ),
        bound_type=bound_type,
        PMF_min_increment=discretization_tail_truncation,
        spacing_type=SpacingType.GEOMETRIC,
    )
    assert isinstance(base_dist_upper, GeometricDiscreteDist)

    conv_dist_raw = geometric_convolve(
        dist_1=conv_dist_lower,
        dist_2=base_dist_upper,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=bound_type,
    )
    # Geometric convolution produces GeometricDiscreteDist; directly apply log
    return log_geometric_to_linear(conv_dist_raw)


def _allocation_PMF_remove(conv_params: _ConvParams,
                           bound_type: BoundType,
                           convolution_method: ConvolutionMethod,
                           ) -> LinearDiscreteDist:
    """Compute REMOVE-direction allocation PMF using the selected backend."""
    if convolution_method == ConvolutionMethod.FFT:
        return _allocation_PMF_remove_fft(conv_params=conv_params, bound_type=bound_type)
    if convolution_method in (ConvolutionMethod.GEOM, ConvolutionMethod.BEST_OF_TWO, ConvolutionMethod.COMBINED):
        return _allocation_PMF_remove_geom(
            conv_params=conv_params,
            bound_type=bound_type,
        )
    raise ValueError(f"Invalid convolution_method: {convolution_method}")

def _allocation_PMF_add_fft(
    conv_params: _ConvParams,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """Compute ADD-direction allocation PMF using the FFT backend only."""
    num_steps = conv_params.num_steps
    sigma = conv_params.sigma
    discretization_tail_truncation = conv_params.discretization_tail_truncation
    pre_composition_tail_truncation = conv_params.pre_composition_tail_truncation / 2
    opposite_bound_type = BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES

    norm_mean = -sigma**2 / 2 - np.log(num_steps)
    base_dist = discretize_continuous_distribution(
        dist=stats.lognorm(s=sigma, scale=np.exp(norm_mean)),
        tail_truncation=discretization_tail_truncation,
        bound_type=opposite_bound_type,
        spacing_type=SpacingType.LINEAR,
        n_grid=conv_params.n_grid_FFT,
        align_to_multiples=False,
    )
    assert isinstance(base_dist, LinearDiscreteDist)
    conv_dist = FFT_self_convolve(
        dist=base_dist,
        T=num_steps,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=opposite_bound_type,
        use_direct=True,
    )
    # FFT produces LinearDiscreteDist in exp-space; regrid to geometric then apply log
    exp_geom = change_spacing_type(
        dist=conv_dist,
        tail_truncation=0.0,
        loss_discretization=conv_params.pre_composition_loss_discretization,
        spacing_type=SpacingType.GEOMETRIC,
        bound_type=opposite_bound_type,
    )
    assert isinstance(exp_geom, GeometricDiscreteDist)
    log_dist = log_geometric_to_linear(exp_geom)
    return negate_reverse_linear_distribution(
        log_dist,
        p_neg_inf=conv_dist.p_pos_inf,
        p_pos_inf=conv_dist.p_neg_inf,
    )


def _allocation_PMF_add_geom(
    conv_params: _ConvParams,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """Compute ADD-direction allocation PMF using the geometric backend only."""
    num_steps = conv_params.num_steps
    sigma = conv_params.sigma
    discretization_tail_truncation = conv_params.discretization_tail_truncation
    pre_composition_tail_truncation = conv_params.pre_composition_tail_truncation / 2
    opposite_bound_type = BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES

    norm_mean = -sigma**2 / 2 - np.log(num_steps)
    base_dist = discretize_continuous_distribution(
        dist=stats.lognorm(s=sigma, scale=np.exp(norm_mean)),
        tail_truncation=discretization_tail_truncation,
        bound_type=opposite_bound_type,
        spacing_type=SpacingType.GEOMETRIC,
        n_grid=conv_params.n_grid_geom,
        align_to_multiples=True,
    )
    assert isinstance(base_dist, GeometricDiscreteDist)
    conv_dist = geometric_self_convolve(
        dist=base_dist,
        T=num_steps,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=opposite_bound_type,
    )
    # Geometric convolution produces GeometricDiscreteDist; directly apply log
    log_dist = log_geometric_to_linear(conv_dist)
    return negate_reverse_linear_distribution(
        log_dist,
        p_neg_inf=conv_dist.p_pos_inf,
        p_pos_inf=conv_dist.p_neg_inf,
    )


def _allocation_PMF_add(conv_params: _ConvParams,
                        bound_type: BoundType,
                        convolution_method: ConvolutionMethod,
                        ) -> LinearDiscreteDist:
    """Compute ADD-direction allocation PMF using the selected backend."""
    if convolution_method == ConvolutionMethod.FFT:
        return _allocation_PMF_add_fft(conv_params=conv_params, bound_type=bound_type)
    if convolution_method in (ConvolutionMethod.GEOM, ConvolutionMethod.BEST_OF_TWO, ConvolutionMethod.COMBINED):
        return _allocation_PMF_add_geom(
            conv_params=conv_params,
            bound_type=bound_type,
        )
    raise ValueError(f"Invalid convolution_method: {convolution_method}")


def _finalize_allocation_dist(
    selected_dist: DiscreteDistBase,
    conv_params: _ConvParams,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    if not (
        isinstance(selected_dist, LinearDiscreteDist)
        and stable_isclose(selected_dist.x_gap, conv_params.pre_composition_loss_discretization)
    ):
        selected_dist = change_spacing_type(
            dist=selected_dist,
            tail_truncation=conv_params.pre_composition_tail_truncation,
            loss_discretization=conv_params.pre_composition_loss_discretization,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )

    assert isinstance(selected_dist, LinearDiscreteDist)
    composed_dist = FFT_self_convolve(
        dist=selected_dist,
        T=conv_params.compose_steps,
        tail_truncation=conv_params.output_tail_truncation,
        bound_type=bound_type,
        use_direct=True,
    )
    final_dist = change_spacing_type(
        dist=composed_dist,
        tail_truncation=conv_params.output_tail_truncation,
        loss_discretization=conv_params.output_loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    assert isinstance(final_dist, LinearDiscreteDist)
    return final_dist


def allocation_PMF(conv_params: _ConvParams,
                   direction: Direction,
                   bound_type: BoundType,
                   convolution_method: ConvolutionMethod,
                   ) -> LinearDiscreteDist:
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
        return _finalize_allocation_dist(
            selected_dist=combine_distributions(FFT_based_dist, geom_based_dist, bound_type=bound_type),
            conv_params=conv_params,
            bound_type=bound_type,
        )
    if convolution_method == ConvolutionMethod.GEOM:
        return _finalize_allocation_dist(
            selected_dist=geom_based_dist,
            conv_params=conv_params,
            bound_type=bound_type,
        )
    if convolution_method == ConvolutionMethod.FFT:
        return _finalize_allocation_dist(
            selected_dist=FFT_based_dist,
            conv_params=conv_params,
            bound_type=bound_type,
        )
    raise ValueError(f"Invalid convolution_method: {convolution_method}")

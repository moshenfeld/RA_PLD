"""
High-level convolution API for discrete distributions.

Provides wrapper functions that dispatch to appropriate convolution implementations
(FFT or geometric) based on the convolution method parameter.
"""

from PLD_accounting.types import BoundType, ConvolutionMethod
from PLD_accounting.discrete_dist import DiscreteDist
from PLD_accounting.FFT_convolution import FFT_convolve, FFT_self_convolve
from PLD_accounting.geometric_convolution import geometric_convolve, geometric_self_convolve

def self_convolve_discrete_distributions(
    dist: DiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
    convolution_method: ConvolutionMethod
) -> DiscreteDist:
    """Convolve distribution with itself T times.

    Dispatches to geometric or FFT implementation based on convolution_method.
    Implementation:
    - GEOM: uses geometric kernel for geometric spacing
    - FFT: uses FFT-based fast convolution for linear spacing
    - Uses binary exponentiation (square-and-multiply) for efficient T-fold convolution
    """
    if convolution_method == ConvolutionMethod.GEOM:
        conv_dist = geometric_self_convolve(
            dist=dist,
            T=T,
            tail_truncation=tail_truncation,
            bound_type=bound_type
        )
    elif convolution_method == ConvolutionMethod.FFT:
        conv_dist = FFT_self_convolve(
            dist=dist,
            T=T,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            use_direct=True
        )
    else:
        raise ValueError(f"Invalid convolution_method: {convolution_method}")

    return conv_dist.validate_mass_conservation(bound_type)

def convolve_discrete_distributions(
    dist_1: DiscreteDist,
    dist_2: DiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
    convolution_method: ConvolutionMethod
) -> DiscreteDist:
    """Convolve two distributions.

    Dispatches to geometric or FFT implementation based on convolution_method.
    Implementation:
    - GEOM: uses geometric kernel for geometric spacing
    - FFT: uses FFT-based multiplication in frequency domain for linear spacing
    - Both methods handle infinite-mass tails correctly
    """
    if not convolution_method in (ConvolutionMethod.GEOM, ConvolutionMethod.FFT):
        raise Exception(f"Invalid convolution_method: {convolution_method}")
    convolution_func = geometric_convolve if convolution_method == ConvolutionMethod.GEOM else FFT_convolve
    return convolution_func(
            dist_1=dist_1,
            dist_2=dist_2,
            tail_truncation=tail_truncation,
            bound_type=bound_type
        ).validate_mass_conservation(bound_type)
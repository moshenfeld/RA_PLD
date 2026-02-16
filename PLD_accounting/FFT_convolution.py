import numpy as np
import math
import sys
import warnings
from scipy.fft import rfft, irfft, next_fast_len
from scipy.signal import fftconvolve

from dp_accounting.pld.common import compute_self_convolve_bounds

from PLD_accounting.core_utils import (
    compute_bin_width_two_arrays,
    compute_bin_width,
    convolve_infinite_masses,
    self_convolve_infinite_mass,
    enforce_mass_conservation,
)
from PLD_accounting.types import BoundType
from PLD_accounting.discrete_dist import DiscreteDist
from PLD_accounting.utils import binary_self_convolve


def calc_FFT_window_size(pmf: np.ndarray, T: int, tail_truncation: float) -> tuple[int, int]:
    """
    Calculate FFT window bounds for T-fold self-convolution with fallback protection.

    Wraps dp_accounting's compute_self_convolve_bounds with protection against
    infinite windows when tail_truncation is too small.
    """
    # Try Chernoff bounds from dp_accounting
    L, U = compute_self_convolve_bounds(pmf, T, tail_truncation)
    window_size = U - L + 1

    # Protect against infinite/overflow when tail_truncation is too small
    if not (0 < window_size < float('inf')):
        L = 0
        n = len(pmf)
        # Exact FFT formula: T-fold self-convolution of size-n produces T*(n-1) + 1 elements
        window_size = T * (n - 1) + 1

        warnings.warn(f"calc_FFT_window_size: Chernoff bounds failed (tail_truncation={tail_truncation:.3e}, T={T}). Using fallback L={0}, window_size={window_size:,} (exact FFT bound for n={n})")

    return int(L), int(window_size)


def FFT_convolve(
    dist_1: DiscreteDist,
    dist_2: DiscreteDist,
    tail_truncation: float,
    bound_type: BoundType
) -> DiscreteDist:
    """
    Convolve two discrete distributions via FFT.
    """
    if not np.any(dist_1.PMF_array) or not np.any(dist_2.PMF_array):
        raise ValueError("FFT convolution requires nonzero finite mass in both inputs")
    # --- CONVOLUTION ---
    # Length of full convolution is N + M - 1
    width = compute_bin_width_two_arrays(dist_1.x_array, dist_2.x_array)
    L = dist_1.x_array.size + dist_2.x_array.size - 1
    conv_x = dist_1.x_array[0] + dist_2.x_array[0] + width * np.arange(L, dtype=float)

    conv_PMF = fftconvolve(dist_1.PMF_array, dist_2.PMF_array, mode='full')
    
    # --- NUMERICAL STABILITY ---
    # Clip negative floating point noise immediately
    conv_PMF[conv_PMF < 0] = 0.0

    # Remove "ghost mass" from indices that are mathematically impossible
    nz1 = np.nonzero(dist_1.PMF_array)[0]
    nz2 = np.nonzero(dist_2.PMF_array)[0]
    min_idx = int(nz1[0] + nz2[0])
    max_idx = int(nz1[-1] + nz2[-1])
    conv_PMF[:min_idx] = 0.0
    if max_idx + 1 < conv_PMF.size:
        conv_PMF[max_idx + 1:] = 0.0

    # Restore finite-mass normalization (FFT/trimming can drift)
    finite_prob_1 = math.fsum(map(float, dist_1.PMF_array))
    finite_prob_2 = math.fsum(map(float, dist_2.PMF_array))
    current_finite_mass = math.fsum(map(float, conv_PMF))
    if current_finite_mass <= 0.0:
        raise ValueError("FFT convolution produced zero finite mass")
    conv_PMF *= finite_prob_1 * finite_prob_2 / current_finite_mass

    expected_neg_inf, expected_pos_inf = convolve_infinite_masses(
        dist_1.p_neg_inf, dist_1.p_pos_inf, dist_2.p_neg_inf, dist_2.p_pos_inf
    )
    conv_PMF, p_neg_inf, p_pos_inf = enforce_mass_conservation(
        PMF_array=conv_PMF,
        expected_neg_inf=expected_neg_inf,
        expected_pos_inf=expected_pos_inf,
        bound_type=bound_type,
    )

    return DiscreteDist(
        x_array=conv_x,
        PMF_array=conv_PMF,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf
    ).truncate_edges(tail_truncation, bound_type)


def FFT_self_convolve(
    dist: DiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
    use_direct: bool
) -> DiscreteDist:
    """
    T-fold self-convolution via FFT with truncation-aware exponentiation.
    """
    if use_direct:
        return _fft_self_convolve_direct(
            dist=dist,
            T=T,
            tail_truncation=tail_truncation,
            bound_type=bound_type
        )

    return binary_self_convolve(
        dist=dist,
        T=T,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=FFT_convolve
    )


def _fft_self_convolve_direct(
    dist: DiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType
) -> DiscreteDist:
    tail_truncation /= 2 #account for the double truncation
    # 1. FFT convolution
    # Computes truncation bounds for convolution using Chernoff bound
    # dp_accounting's compute_self_convolve_bounds expects a normalized PMF (sum=1.0)
    # If mass is at infinity, we normalize the PMF and rescale tail_truncation accordingly
    finite_mass = math.fsum(map(float, dist.PMF_array))
    normalized_PMF = dist.PMF_array / finite_mass
    tail_truncation_rescaled = tail_truncation / finite_mass

    # Use wrapper with fallback protection
    # Returns int, so no cast needed
    L, window_size = calc_FFT_window_size(normalized_PMF, T, tail_truncation_rescaled)

    # Increase output size to be a power of a small integer
    fft_size = next_fast_len(max(window_size, dist.PMF_array.size))
    # Perform FFT convolution
    raw_conv = irfft(rfft(dist.PMF_array, n=fft_size)**T, n=fft_size)
    raw_conv[raw_conv < 0] = 0.0
    rolled_conv = np.roll(raw_conv, -L)
    
    # 2. Tails probability
    # Analytical  calculation of convolved infinity masses
    conv_neg_inf, conv_pos_inf = self_convolve_infinite_mass(dist.p_neg_inf, dist.p_pos_inf, T)
    if bound_type == BoundType.DOMINATES:
        # Move tail_truncation probability mass from the extreme left to pos_inf
        # to compensate for the potential roll over resulting from the fact fft_size < the full range
        cumsum = np.cumsum(rolled_conv)
        left_tail_ind = int(np.searchsorted(cumsum, tail_truncation, side="right"))
        shifted_mass = math.fsum(map(float, rolled_conv[:left_tail_ind]))
        rolled_conv[:left_tail_ind] = 0.0
        # Move the right tail mass to pos inf
        right_tail_mass = math.fsum(map(float, rolled_conv[window_size:]))
        conv_pos_inf += shifted_mass + right_tail_mass
    elif bound_type == BoundType.IS_DOMINATED:
        # Move tail_truncation probability mass from the extreme right to neg_inf
        # to compensate for the potential roll over resulting from the fact fft_size < the full range
        cumsum = np.cumsum(rolled_conv[::-1])
        right_tail_ind = rolled_conv.size - 1 - int(np.searchsorted(cumsum, tail_truncation, side="right"))
        shifted_mass = math.fsum(map(float, rolled_conv[right_tail_ind+1:]))
        rolled_conv[right_tail_ind+1:] = 0.0
        conv_neg_inf += shifted_mass
        # Move the right tail mass to index window_size-1
        right_tail_mass = math.fsum(map(float, rolled_conv[window_size:]))
        rolled_conv[min(window_size, right_tail_ind)-1] += right_tail_mass
        # Truncate the PMF array
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    # 3. Finite probability
    # PMF is [L:U+1]
    width = compute_bin_width(dist.x_array)
    x_array = (dist.x_array[0] * T + L * width) + np.arange(window_size) * width
    PMF_conv = rolled_conv[:window_size]
    PMF_conv, p_neg_inf_final, p_pos_inf_final = enforce_mass_conservation(
        PMF_array=PMF_conv,
        expected_neg_inf=conv_neg_inf,
        expected_pos_inf=conv_pos_inf,
        bound_type=bound_type,
    )
    return DiscreteDist(
        x_array=x_array,
        PMF_array=PMF_conv,
        p_neg_inf=p_neg_inf_final,
        p_pos_inf=p_pos_inf_final
    ).truncate_edges(0.0, bound_type)
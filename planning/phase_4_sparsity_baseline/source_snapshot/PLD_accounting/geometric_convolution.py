from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from PLD_accounting.core_utils import (
    compute_bin_ratio_two_arrays,
    convolve_infinite_masses,
    enforce_mass_conservation,
    stable_isclose,
)
from PLD_accounting.discrete_dist import (
    DenseGeometricDiscreteDist,
    GeometricDiscreteDistBase,
    SparseGeometricDiscreteDist,
)
from PLD_accounting.discrete_dist_utils import to_dense_geometric
from PLD_accounting.types import BoundType
from PLD_accounting.utils import binary_self_convolve


# =============================================================================
# PUBLIC API
# =============================================================================


def geometric_convolve(
    dist_1: GeometricDiscreteDistBase,
    dist_2: GeometricDiscreteDistBase,
    tail_truncation: float,
    bound_type: BoundType,
) -> GeometricDiscreteDistBase:
    """
    Convolve two geometric-grid distributions.

    Dispatches to sparse kernel when both inputs are sparse geometric.
    Otherwise densifies sparse inputs and uses the dense geometric kernel.
    """
    if isinstance(dist_1, SparseGeometricDiscreteDist) and isinstance(dist_2, SparseGeometricDiscreteDist):
        return sparse_geometric_convolve(dist_1, dist_2, tail_truncation, bound_type)

    dense_1 = to_dense_geometric(dist_1)
    dense_2 = to_dense_geometric(dist_2)

    ratio = compute_bin_ratio_two_arrays(dense_1.x_array, dense_2.x_array)

    x_out, pmf_conv = _compute_geometric_convolution(
        dense_1.x_array,
        dense_1.PMF_array,
        dense_2.x_array,
        dense_2.PMF_array,
        ratio,
        bound_type,
    )

    expected_neg_inf, expected_pos_inf = convolve_infinite_masses(
        dense_1.p_neg_inf,
        dense_1.p_pos_inf,
        dense_2.p_neg_inf,
        dense_2.p_pos_inf,
    )
    pmf_conv, p_neg_inf, p_pos_inf = enforce_mass_conservation(
        PMF_array=pmf_conv,
        expected_neg_inf=expected_neg_inf,
        expected_pos_inf=expected_pos_inf,
        bound_type=bound_type,
    )

    return DenseGeometricDiscreteDist(
        x_min=float(x_out[0]),
        ratio=ratio,
        PMF_array=pmf_conv,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
    ).truncate_edges(tail_truncation, bound_type)



def geometric_self_convolve(
    dist: GeometricDiscreteDistBase,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> GeometricDiscreteDistBase:
    """Self-convolve a geometric-grid distribution T times via binary exponentiation."""
    if isinstance(dist, SparseGeometricDiscreteDist):
        return sparse_geometric_self_convolve(dist, T, tail_truncation, bound_type)

    self_conv = binary_self_convolve(
        dist=dist,
        T=T,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=geometric_convolve,
    )
    if not isinstance(self_conv, GeometricDiscreteDistBase):
        raise TypeError(f"Expected GeometricDiscreteDistBase from self-convolution, got {type(self_conv)}")
    return self_conv



def sparse_geometric_convolve(
    dist_1: SparseGeometricDiscreteDist,
    dist_2: SparseGeometricDiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
) -> GeometricDiscreteDistBase:
    """
    Convolve two sparse geometric distributions.

    Runtime contract: only SparseGeometricDiscreteDist inputs.
    """
    if not isinstance(dist_1, SparseGeometricDiscreteDist):
        raise TypeError(f"dist_1 must be SparseGeometricDiscreteDist, got {type(dist_1)}")
    if not isinstance(dist_2, SparseGeometricDiscreteDist):
        raise TypeError(f"dist_2 must be SparseGeometricDiscreteDist, got {type(dist_2)}")

    if not stable_isclose(dist_1.ratio, dist_2.ratio):
        raise ValueError(f"Ratios must match: {dist_1.ratio} vs {dist_2.ratio}")

    ratio = dist_1.ratio
    x_out_min, pmf_out = _compute_sparse_convolution_on_dense_frame(
        indices_1=dist_1.indices,
        pmf_1=dist_1.PMF_array,
        indices_2=dist_2.indices,
        pmf_2=dist_2.PMF_array,
        x_min_1=dist_1.x_min,
        x_min_2=dist_2.x_min,
        ratio=ratio,
        bound_type=bound_type,
    )

    expected_neg_inf, expected_pos_inf = convolve_infinite_masses(
        dist_1.p_neg_inf,
        dist_1.p_pos_inf,
        dist_2.p_neg_inf,
        dist_2.p_pos_inf,
    )
    pmf_out, p_neg_inf, p_pos_inf = enforce_mass_conservation(
        PMF_array=pmf_out,
        expected_neg_inf=expected_neg_inf,
        expected_pos_inf=expected_pos_inf,
        bound_type=bound_type,
    )

    dense_result = DenseGeometricDiscreteDist(
        x_min=x_out_min,
        ratio=ratio,
        PMF_array=pmf_out,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
    )
    truncated_dense = dense_result.truncate_edges(tail_truncation, bound_type)
    if isinstance(truncated_dense, DenseGeometricDiscreteDist):
        return _maybe_sparsify_dense_geometric(truncated_dense)
    return truncated_dense



def sparse_geometric_self_convolve(
    dist: SparseGeometricDiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> GeometricDiscreteDistBase:
    """Self-convolve sparse geometric distribution T times."""
    self_conv = binary_self_convolve(
        dist=dist,
        T=T,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=geometric_convolve,
    )
    if not isinstance(self_conv, GeometricDiscreteDistBase):
        raise TypeError(f"Expected GeometricDiscreteDistBase from self-convolution, got {type(self_conv)}")
    return self_conv


# =============================================================================
# INTERNAL DENSE KERNEL
# =============================================================================


def _compute_geometric_convolution(
    x1: NDArray[np.float64],
    p1: NDArray[np.float64],
    x2: NDArray[np.float64],
    p2: NDArray[np.float64],
    r: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Align dense geometric grids and invoke the Numba dense kernel."""
    if x1[0] > x2[0]:
        x1, p1, x2, p2 = x2, p2, x1, p1

    scale = x2[0] / x1[0]

    target_n = max(x1.size, x2.size)
    if x1.size < target_n:
        x1, p1 = _pad_right_geometric(x1, p1, r, target_n)
    elif x2.size < target_n:
        x2, p2 = _pad_right_geometric(x2, p2, r, target_n)

    x_base = x1.astype(np.float64, copy=False)
    pmf_base = p1.astype(np.float64, copy=False)
    pmf_scaled = p2.astype(np.float64, copy=False)

    n = x_base.size
    if n == 1:
        mass = pmf_base[0] * pmf_scaled[0]
        x_out = np.array([(scale + 1.0) * x_base[0]], dtype=np.float64)
        pmf_out = np.array([mass], dtype=np.float64)
        return x_out, pmf_out

    log_r = np.log(r)
    log_scale = np.log(scale)
    log_ap1 = np.log(scale + 1.0)

    d_vec = np.arange(n, dtype=np.float64)
    log_r_d = d_vec * log_r

    log_lohi = np.logaddexp(0.0, log_scale + log_r_d)
    tau_lohi = (log_lohi - log_ap1) / log_r

    log_hilo = np.logaddexp(log_scale, log_r_d)
    tau_hilo = (log_hilo - log_ap1) / log_r

    delta_lohi = np.zeros(n, dtype=np.int64)
    delta_hilo = np.zeros(n, dtype=np.int64)
    rounding_eps = 1e-16

    if bound_type == BoundType.DOMINATES:
        delta_lohi[1:] = np.ceil(tau_lohi[1:] - rounding_eps).astype(np.int64)
        delta_hilo[1:] = np.ceil(tau_hilo[1:] - rounding_eps).astype(np.int64)
    elif bound_type == BoundType.IS_DOMINATED:
        delta_lohi[1:] = np.floor(tau_lohi[1:] + rounding_eps).astype(np.int64)
        delta_hilo[1:] = np.floor(tau_hilo[1:] + rounding_eps).astype(np.int64)
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    pmf_out = _numba_geometric_kernel(pmf_base, pmf_scaled, delta_lohi, delta_hilo)
    x_out = x_base * (scale + 1.0)

    return x_out, pmf_out



def _pad_right_geometric(
    x: NDArray[np.float64],
    p: NDArray[np.float64],
    r: float,
    target_n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extend a geometric grid on the right to target length."""
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = x.size
    if n >= target_n:
        return x, p

    k = target_n - n
    if r == 1.0:
        tail = np.full(k, x[-1], dtype=np.float64)
    else:
        tail = x[-1] * (r ** np.arange(1, k + 1, dtype=np.float64))

    x_ext = np.concatenate([x, tail])
    p_ext = np.pad(p, (0, k), mode="constant")
    return x_ext, p_ext


@njit(cache=True)
def _numba_geometric_kernel(
    PMF_base: NDArray[np.float64],
    PMF_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Dense geometric convolution kernel with compensated summation."""
    n = PMF_base.size
    pmf_out = np.zeros(n, dtype=np.float64)
    comp = np.zeros(n, dtype=np.float64)

    for i in range(n):
        mass = PMF_base[i] * PMF_scaled[i]
        y = mass - comp[i]
        t = pmf_out[i] + y
        comp[i] = (t - pmf_out[i]) - y
        pmf_out[i] = t

    for d in range(1, n):
        imax = n - d
        kshift1 = int(delta_lohi[d])
        kshift2 = int(delta_hilo[d])

        for i in range(imax):
            k1 = i + kshift1
            mass1 = PMF_base[i] * PMF_scaled[i + d]
            if 0 <= k1 < n:
                y = mass1 - comp[k1]
                t = pmf_out[k1] + y
                comp[k1] = (t - pmf_out[k1]) - y
                pmf_out[k1] = t

            k2 = i + kshift2
            mass2 = PMF_base[i + d] * PMF_scaled[i]
            if 0 <= k2 < n:
                y = mass2 - comp[k2]
                t = pmf_out[k2] + y
                comp[k2] = (t - pmf_out[k2]) - y
                pmf_out[k2] = t

    return pmf_out


# =============================================================================
# SPARSE-INPUT DENSE-FRAME KERNEL
# =============================================================================


def _compute_sparse_convolution_on_dense_frame(
    indices_1: NDArray[np.int64],
    pmf_1: NDArray[np.float64],
    indices_2: NDArray[np.int64],
    pmf_2: NDArray[np.float64],
    x_min_1: float,
    x_min_2: float,
    ratio: float,
    bound_type: BoundType,
) -> tuple[float, NDArray[np.float64]]:
    """
    Convolve sparse inputs into the same dense output frame that dense mode would use.

    This keeps output frame width tied to dense semantics (n = max(input spans))
    while still iterating only over nonzero sparse entries.
    """
    min_idx_1 = int(indices_1[0])
    min_idx_2 = int(indices_2[0])
    x_support_min_1 = float(x_min_1 * np.power(ratio, float(min_idx_1)))
    x_support_min_2 = float(x_min_2 * np.power(ratio, float(min_idx_2)))

    if x_support_min_1 <= x_support_min_2:
        base_indices, pmf_base_sparse = indices_1, pmf_1
        scaled_indices, pmf_scaled_sparse = indices_2, pmf_2
        x_base_min = x_support_min_1
        x_scaled_min = x_support_min_2
    else:
        base_indices, pmf_base_sparse = indices_2, pmf_2
        scaled_indices, pmf_scaled_sparse = indices_1, pmf_1
        x_base_min = x_support_min_2
        x_scaled_min = x_support_min_1

    min_base = int(base_indices[0])
    max_base = int(base_indices[-1])
    min_scaled = int(scaled_indices[0])
    max_scaled = int(scaled_indices[-1])

    span_base = max_base - min_base + 1
    span_scaled = max_scaled - min_scaled + 1
    n = int(max(span_base, span_scaled))

    local_base = (base_indices - min_base).astype(np.int64, copy=False)
    local_scaled = (scaled_indices - min_scaled).astype(np.int64, copy=False)
    scale = float(x_scaled_min / x_base_min)

    delta_lohi, delta_hilo = _compute_dense_delta_arrays(n=n, ratio=ratio, scale=scale, bound_type=bound_type)
    if _can_use_no_check_sparse_kernel(local_base, local_scaled, delta_lohi, delta_hilo):
        pmf_out = _numba_sparse_to_dense_frame_kernel_no_checks(
            local_base,
            pmf_base_sparse,
            local_scaled,
            pmf_scaled_sparse,
            delta_lohi,
            delta_hilo,
        )
    else:
        pmf_out = _numba_sparse_to_dense_frame_kernel_sweepline(
            local_base,
            pmf_base_sparse,
            local_scaled,
            pmf_scaled_sparse,
            delta_lohi,
            delta_hilo,
        )

    x_out_min = float(x_base_min * (1.0 + scale))
    return x_out_min, pmf_out


def _can_use_no_check_sparse_kernel(
    indices_base: NDArray[np.int64],
    indices_scaled: NDArray[np.int64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
) -> bool:
    """
    Return True when every active sparse pair is guaranteed to map in-frame.

    Conditions:
    1. Left branch safety: delta_hilo[d] <= d for all d, so k <= i < n.
    2. Right branch safety over active d-ranges for each i in indices_base.
    """
    n = int(delta_lohi.size)
    d = np.arange(n, dtype=np.int64)

    if bool(np.any(delta_hilo > d)):
        return False

    max_scaled = int(indices_scaled[-1])
    for i_val in indices_base:
        i = int(i_val)
        if i > max_scaled:
            continue
        d_max_active = max_scaled - i
        if i + int(delta_lohi[d_max_active]) >= n:
            return False

    return True


def _compute_dense_delta_arrays(
    n: int,
    ratio: float,
    scale: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Compute dense-grid delta mappings (same formulas used by dense kernel)."""
    delta_lohi = np.zeros(n, dtype=np.int64)
    delta_hilo = np.zeros(n, dtype=np.int64)
    if n <= 1:
        return delta_lohi, delta_hilo

    log_r = np.log(ratio)
    log_scale = np.log(scale)
    log_ap1 = np.log(scale + 1.0)

    d_vec = np.arange(n, dtype=np.float64)
    log_r_d = d_vec * log_r
    log_lohi = np.logaddexp(0.0, log_scale + log_r_d)
    tau_lohi = (log_lohi - log_ap1) / log_r
    log_hilo = np.logaddexp(log_scale, log_r_d)
    tau_hilo = (log_hilo - log_ap1) / log_r

    rounding_eps = 1e-16
    if bound_type == BoundType.DOMINATES:
        delta_lohi[1:] = np.ceil(tau_lohi[1:] - rounding_eps).astype(np.int64)
        delta_hilo[1:] = np.ceil(tau_hilo[1:] - rounding_eps).astype(np.int64)
    elif bound_type == BoundType.IS_DOMINATED:
        delta_lohi[1:] = np.floor(tau_lohi[1:] + rounding_eps).astype(np.int64)
        delta_hilo[1:] = np.floor(tau_hilo[1:] + rounding_eps).astype(np.int64)
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    return delta_lohi, delta_hilo


@njit(cache=True)
def _numba_sparse_to_dense_frame_kernel_no_checks(
    indices_base: NDArray[np.int64],
    pmf_base: NDArray[np.float64],
    indices_scaled: NDArray[np.int64],
    pmf_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
) -> NDArray[np.float64]:
    """
    Fast sparse-input kernel for the in-frame case.

    Mapping matches dense semantics:
    - if i <= j: k = i + delta_lohi[j - i]
    - if i >  j: k = j + delta_hilo[i - j]
    """
    n = delta_lohi.size
    pmf_out = np.zeros(n, dtype=np.float64)
    comp = np.zeros(n, dtype=np.float64)

    n_base = indices_base.size
    n_scaled = indices_scaled.size

    for a in range(n_base):
        i = int(indices_base[a])
        p_i = pmf_base[a]
        for b in range(n_scaled):
            j = int(indices_scaled[b])
            mass = p_i * pmf_scaled[b]

            if i <= j:
                k = i + int(delta_lohi[j - i])
            else:
                k = j + int(delta_hilo[i - j])

            y = mass - comp[k]
            t = pmf_out[k] + y
            comp[k] = (t - pmf_out[k]) - y
            pmf_out[k] = t

    return pmf_out


@njit(cache=True)
def _lower_bound_sorted_int64(arr: NDArray[np.int64], value: int) -> int:
    """Return first index i such that arr[i] >= value."""
    lo = 0
    hi = arr.size
    while lo < hi:
        mid = (lo + hi) // 2
        if int(arr[mid]) < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _numba_sparse_to_dense_frame_kernel_sweepline(
    indices_base: NDArray[np.int64],
    pmf_base: NDArray[np.float64],
    indices_scaled: NDArray[np.int64],
    pmf_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
) -> NDArray[np.float64]:
    """
    Sparse-input kernel using sorted-index split and right-tail early break.

    This path is for cases where some (i, j) pairs map outside output frame.
    It avoids evaluating the entire right tail once k exceeds n-1.
    """
    n = delta_lohi.size
    pmf_out = np.zeros(n, dtype=np.float64)
    comp = np.zeros(n, dtype=np.float64)

    n_base = indices_base.size
    n_scaled = indices_scaled.size

    for a in range(n_base):
        i = int(indices_base[a])
        p_i = pmf_base[a]

        # Split sorted scaled indices into j < i and j >= i.
        split = _lower_bound_sorted_int64(indices_scaled, i)

        # Left side (j < i): k = j + delta_hilo[i - j] may or may not be in frame.
        for b in range(split):
            j = int(indices_scaled[b])
            k = j + int(delta_hilo[i - j])
            if k < n:
                mass = p_i * pmf_scaled[b]
                y = mass - comp[k]
                t = pmf_out[k] + y
                comp[k] = (t - pmf_out[k]) - y
                pmf_out[k] = t

        # Right side (j >= i): k is monotone nondecreasing in j, so break once out.
        for b in range(split, n_scaled):
            j = int(indices_scaled[b])
            k = i + int(delta_lohi[j - i])
            if k >= n:
                break
            mass = p_i * pmf_scaled[b]
            y = mass - comp[k]
            t = pmf_out[k] + y
            comp[k] = (t - pmf_out[k]) - y
            pmf_out[k] = t

    return pmf_out


def _maybe_sparsify_dense_geometric(
    dist: DenseGeometricDiscreteDist,
    density_threshold: float = 0.8,
) -> GeometricDiscreteDistBase:
    """Convert dense geometric result to sparse when exact dense zeros dominate."""
    keep = dist.PMF_array != 0.0
    keep_count = int(np.count_nonzero(keep))
    n = int(dist.PMF_array.size)
    if keep_count < 2 or keep_count == n:
        return dist

    density = float(keep_count) / float(n)
    if density > density_threshold:
        return dist

    indices = np.nonzero(keep)[0].astype(np.int64)
    pmf_sparse = dist.PMF_array[keep]
    return SparseGeometricDiscreteDist(
        x_min=dist.x_min,
        ratio=dist.ratio,
        indices=indices,
        PMF_array=pmf_sparse,
        p_neg_inf=dist.p_neg_inf,
        p_pos_inf=dist.p_pos_inf,
    )

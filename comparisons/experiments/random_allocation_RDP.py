from numba import jit
import numpy as np
import math
from typing import Tuple, List, Union
import warnings


@jit(nopython=True)
def log_factorial_sum_numba(n: int) -> float:
    """Compute log(n!) using Numba-accelerated sum of logarithms.

    Implementation sums log(1) + log(2) + ... + log(n) for numerical stability.
    """
    if n <= 1:
        return 0.0
    return np.sum(np.log(np.arange(1, n + 1)))

@jit(nopython=True)
def compute_exp_term(partition: Tuple[int, ...], n: int, sigma: float) -> float:
    """Compute the logarithmic contribution of a partition to RDP.

    Calculates log of the multinomial coefficient times the exponential term.
    Implementation uses log-space arithmetic throughout to prevent overflow.
    The formula accounts for: (n choose p) * k! / (product of factorials) * exp(sum of squares / 2σ²).
    """
    p = len(partition)
    k = sum(partition)

    # Compute log(n choose p) = log(n!) - log((n-p)!) - log(p!) in stable form
    log_n_choose_p = np.sum(np.log(np.arange(n - p + 1, n + 1)))

    # Compute log(k!)
    log_k_factorial = log_factorial_sum_numba(k)

    # Count frequency of each value in the partition
    counts = np.zeros(k + 1, dtype=np.int64)
    for x in partition:
        counts[x] += 1

    # Compute denominator: product of (count[i]! * i!^count[i]) for all i
    denominator = 0.0
    for i in range(1, k + 1):
        if counts[i] > 0:
            denominator += log_factorial_sum_numba(counts[i])
            denominator += counts[i] * log_factorial_sum_numba(i)

    # Add the squared sum term weighted by 1/(2σ²)
    squared_sum = np.sum(np.array(partition) ** 2) / (2 * sigma * sigma)
    return log_n_choose_p + log_k_factorial - denominator + squared_sum

def generate_partitions_general(n: int, max_size: int) -> List[Tuple[int, ...]]:
    """Generate all integer partitions of n with at most max_size parts.

    Uses dynamic programming to build partitions incrementally.
    Implementation ensures partitions are in descending order (canonical form).
    Constraint max_size bounds the number of parts (relevant for bounding num_steps).
    """
    partitions = [[] for _ in range(n + 1)]
    partitions[0].append(())

    for i in range(1, n + 1):
        for j in range(i, 0, -1):
            for p in partitions[i - j]:
                if (not p or j <= p[0]) and len(p) < max_size:  # Ensure descending order
                    partitions[i].append((j,) + p)
    return partitions[n]

def allocation_rdp_remove(alpha: int, sigma: float, num_steps: int) -> float:
    """Compute Rényi Differential Privacy (RDP) for random allocation.

    Calculates the RDP guarantee at order alpha using the direct combinatorial formula.
    Implementation enumerates all partitions of alpha and computes their contributions.
    Uses log-sum-exp trick for numerical stability when aggregating exponential terms.
    Returns the RDP value: (1/(α-1)) * log(E[exp((α-1) * privacy_loss)]).
    """
    # Generate all partitions of alpha with at most num_steps parts
    partitions = generate_partitions_general(alpha, num_steps)

    # Compute log of each term's contribution
    exp_terms = [compute_exp_term(p, num_steps, sigma) for p in partitions]

    # Use log-sum-exp trick: log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    max_val = max(exp_terms)
    log_sum = np.log(sum(np.exp(term - max_val) for term in exp_terms))

    # Return RDP value after subtracting normalization terms
    return (log_sum - alpha*(1/(2*sigma**2) + np.log(num_steps)) + max_val) / (alpha-1)

def calc_log_comb(alpha, ell):
    """Calculate log of binomial coefficient (alpha choose ell).

    Computes log(alpha choose ell) iteratively in log-space to avoid overflow.
    Implementation uses incremental multiplication: log(a/b * c/d * ...).
    """
    res = 0.0
    for j in range(ell):
        res += math.log((alpha - ell + j + 1)/(j+1))
    return res
        
def _log_add(logx: float, logy: float) -> float:
    """Add two numbers in log-space: log(exp(logx) + exp(logy)).

    Implementation uses log1p for numerical stability when one term dominates.
    Handles -inf (representing zero) correctly.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)

def ampl_subsampling_rdp(eps, alpha_lst, q):
    """Apply RDP amplification by subsampling (Theorem 5 of Zhu et al. 2019).

    Computes amplified RDP values for subsampling with probability q.
    Reference: http://proceedings.mlr.press/v97/zhu19c/zhu19c.pdf
    Implementation assumes alpha_lst = [2, 3, 4, ..., K] consecutive integers.
    Uses log-space arithmetic throughout to maintain numerical stability.
    """
    if q == 1:
        return eps
    eps_sub = np.zeros(len(alpha_lst) + 2)
    p = 1-q

    for j in range(len(alpha_lst)):
        alpha = alpha_lst[j]
        # Compute three terms of the amplification formula in log-space
        term1_log = (alpha-1) * math.log(p) + math.log(alpha*q - q + 1)
        term2_log = math.log(alpha*(alpha-1)/2.0) + 2 * math.log(q) + (alpha-2)*math.log(p) + eps[0]
        term3_log = None
        for ell in range(3, alpha+1):
            curr = math.log(3) + calc_log_comb(alpha, ell)
            curr += (alpha-ell) * math.log(p)
            curr += ell * math.log(q) + (ell-1)*eps[ell-2]
            if ell == 3:
                term3_log = curr
            else:
                term3_log = _log_add(term3_log, curr)
        # Aggregate all terms using log-add
        eps_curr = _log_add(term1_log, term2_log)
        if alpha > 2:
            eps_curr = _log_add(eps_curr, term3_log)
        eps_sub[alpha] = eps_curr/(alpha-1)
    return eps_sub[2:]


def get_privacy_spent(
    *, orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
) -> Tuple[float, float]:
    """Convert RDP to (ε,δ)-DP using optimal order selection.

    Computes epsilon by optimizing over all provided RDP orders (alphas).
    Reference: Balle et al. (AISTATS 2020), Theorem 21.
    https://arxiv.org/abs/1905.09982
    Implementation evaluates the conversion formula for each order and selects minimum ε.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    # Apply conversion formula: ε(α) = RDP(α) + (log((α-1)/α) - log(δ) - log(α))/(α-1)
    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )

    # Handle edge case of no privacy loss
    if np.isnan(eps).all():
        return np.inf, np.nan

    # Select order that minimizes epsilon
    idx_opt = np.nanargmin(eps)
    if idx_opt == 0 or idx_opt == len(eps) - 1:
        extreme = "smallest" if idx_opt == 0 else "largest"
        warnings.warn(f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound.")
    return eps[idx_opt], orders_vec[idx_opt]

def allocation_rdp_add(alpha: float, sigma: float, num_steps: int, num_selected: int
) -> float:
    """
    Compute an upper bound on RDP of the allocation mechanism (add direction)

    Args:
        sigma: Noise scale
        num_steps: Number of steps
        num_selected: Number of selected items
        alpha: Alpha order for RDP

    Returns:
        Upper bound on RDP
    """
    return float(alpha*num_selected**2/(2*sigma**2*num_steps) \
        + (alpha*num_selected*(num_steps-num_selected))/(2*sigma**2*num_steps*(alpha-1)) \
        - num_steps*np.log1p(alpha*(np.exp(num_selected*(num_steps-num_selected)/(sigma**2*num_steps**2))-1))/(2*(alpha-1)))

def allocation_rdp_arr(sigma, num_steps, num_selected, orders, is_remove):
    if is_remove:
        num_steps_per_round = int(np.floor(num_steps / num_selected))
        return np.array([allocation_rdp_remove(alpha=alpha, sigma=sigma, num_steps=num_steps_per_round)*num_selected for alpha in orders])
    else:
        return np.array([allocation_rdp_add(alpha=alpha, sigma=sigma, num_steps=num_steps, num_selected=num_selected) for alpha in orders])

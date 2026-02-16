"""
Subsampling methods comparison experiment.

Compares different methods for computing subsampling:
1. Analytic Gaussian mixture (ground truth) - closed-form formulas
2. dp_accounting library (TensorFlow Privacy) - reference implementation
3. Our implementation (optional) - DiscreteDist + subsample_PMF pipeline

Plots CCDFs (complementary CDFs) for visual comparison.
"""
from typing import Optional, Dict, Any, List
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.dp_accounting_support import discrete_dist_to_dp_accounting_pmf
from PLD_accounting.distribution_discretization import discretize_continuous_to_pmf
from PLD_accounting.types import BoundType, SpacingType, Direction
from PLD_accounting.discrete_dist import DiscreteDist
from PLD_accounting.subsample_PLD import subsample_PMF
from comparisons.visualization_definitions import SAVEFIG_DPI
from comparisons.plotting_utils import configure_matplotlib_for_publication, show_or_close_figure


def compute_analytical_subsampled_gaussian(
    sigma: float,
    sampling_prob: float,
    discretization: float,
    remove_direction: bool = True
):
    """
    Compute analytical subsampled Gaussian PLD - the ground truth.

    Uses closed-form formulas for the privacy loss distribution.
    First computes the base (unsubsampled) Gaussian PLD, then applies
    the subsampling transformation.
    """
    # First compute base Gaussian PLD on original loss grid
    l_max = np.ceil(20.0 / (sigma * discretization)) * discretization
    n_steps = int(np.round(l_max / discretization))
    base_losses = np.arange(-n_steps, n_steps + 1) * discretization

    transformed_losses = stable_subsampling_loss(
        base_losses, sampling_prob, remove_direction=remove_direction
    )

    # Compute CCDF values using analytical formula
    x_upper = sigma * transformed_losses - 0.5 / sigma
    x_lower = sigma * transformed_losses + 0.5 / sigma

    if remove_direction:
        # Remove direction: mixture of base and neighbor distributions
        S = (1.0 - sampling_prob) * stats.norm.sf(x_lower) + \
            sampling_prob * stats.norm.sf(x_upper)
    else:
        # Add direction: only base distribution
        S = stats.norm.sf(x_upper)

    base_probs = np.concatenate(([1.0], S[:-1])) - S
    return base_losses, base_probs


def compute_analytical_base_gaussian(
    sigma: float,
    discretization: float,
    remove_direction: bool = True
):
    """
    Compute analytical base (unsubsampled) Gaussian PLD.
    """
    return compute_analytical_subsampled_gaussian(sigma, 1.0, discretization, remove_direction)


def _pmf_to_loss_probs(pmf):
    """Convert a dp_accounting PMF to loss grid and finite-mass probabilities."""
    dense = pmf.to_dense_pmf()
    losses = (np.arange(dense.size) + dense._lower_loss) * dense._discretization  # pylint: disable=protected-access
    probs = dense._probs  # pylint: disable=protected-access
    finite_target = float(max(0.0, 1.0 - dense._infinity_mass))  # pylint: disable=protected-access
    sum_probs = float(np.sum(probs, dtype=np.float64))
    if sum_probs > 0:
        probs = probs * (finite_target / sum_probs)
    return losses, probs


def _upper_to_lower(dist: DiscreteDist) -> DiscreteDist:
    if dist.p_neg_inf > 0:
        raise ValueError("Expected p_neg_inf=0 for upper PLD")
    losses = dist.x_array
    probs = dist.PMF_array
    lower_probs = np.zeros_like(probs)
    mask = probs > 0
    lower_probs[mask] = np.exp(np.log(probs[mask]) - losses[mask])
    sum_prob = float(np.sum(lower_probs))
    return DiscreteDist(
        x_array=losses,
        PMF_array=lower_probs,
        p_neg_inf=max(0.0, 1.0 - sum_prob),
        p_pos_inf=0.0,
    )


def run_subsampling_comparison_experiment(
    sigma: float,
    sampling_prob: float,
    discretization: float,
    sensitivity: float = 1.0,
    include_our_implementation: bool = False,
    direction: str = "remove",
    delta_values: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compare subsampling methods and return versions for plotting/analysis.

    Args:
        sigma: Noise standard deviation
        sampling_prob: Subsampling probability
        discretization: Grid discretization
        sensitivity: Sensitivity of the mechanism (default 1.0)
        include_our_implementation: Whether to include method 3 (our bounds)
        direction: "remove", "add", or "both"
        delta_values: Delta grid for epsilon comparison

    Returns:
        Dictionary containing:
        - params: experiment parameters
        - remove/add: dicts with versions, eps_GT, and delta_values
    """
    if direction not in {"remove", "add", "both"}:
        raise ValueError("direction must be 'remove', 'add', or 'both'")

    if delta_values is None:
        delta_values = [1e-6, 1e-8, 1e-10]

    results: Dict[str, Any] = {
        "params": {
            "sigma": sigma,
            "sampling_prob": sampling_prob,
            "discretization": discretization,
            "sensitivity": sensitivity,
            "include_our_implementation": include_our_implementation,
            "direction": direction,
        },
    }

    def _compute_direction_results(
        direction_label: str,
        remove_direction: bool,
    ) -> Dict[str, Any]:
        # Method 1: Analytic Gaussian mixture (ground truth)
        analytic_losses, analytic_probs = compute_analytical_subsampled_gaussian(
            sigma, sampling_prob, discretization, remove_direction=remove_direction
        )
        analytic_dist = DiscreteDist(
            x_array=analytic_losses,
            PMF_array=analytic_probs,
            p_neg_inf=0.0,
            p_pos_inf=max(0.0, 1.0 - float(np.sum(analytic_probs))),
        )
        analytic_pmf = discrete_dist_to_dp_accounting_pmf(
            analytic_dist,
            pessimistic_estimate=True,
        )

        # Method 2: dp_accounting library
        dp_pld = privacy_loss_distribution.from_gaussian_mechanism(
            standard_deviation=sigma,
            sensitivity=sensitivity,
            sampling_prob=sampling_prob,
            value_discretization_interval=discretization,
            pessimistic_estimate=True,
        )
        dp_pmf = dp_pld._pmf_remove if remove_direction else dp_pld._pmf_add
        dp_losses, dp_probs = _pmf_to_loss_probs(dp_pmf)

        versions: List[Dict[str, Any]] = [
            {
                "name": "TF_TF",
                "pmf": dp_pmf,
                "losses": dp_losses,
                "probs": dp_probs,
            },
        ]

        if include_our_implementation:
            base_pld = privacy_loss_distribution.from_gaussian_mechanism(
                standard_deviation=sigma,
                sensitivity=sensitivity,
                value_discretization_interval=discretization,
                pessimistic_estimate=True,
            )
            base_pmf = base_pld._pmf_remove if remove_direction else base_pld._pmf_add
            base_losses, base_probs = _pmf_to_loss_probs(base_pmf)
            our_tf_probs = subsample_losses(
                base_losses, base_probs, sampling_prob, remove_direction
            )
            our_tf_dist = DiscreteDist(
                x_array=base_losses,
                PMF_array=our_tf_probs,
                p_neg_inf=0.0,
                p_pos_inf=max(0.0, 1.0 - float(np.sum(our_tf_probs))),
            )
            our_tf_pmf = discrete_dist_to_dp_accounting_pmf(
                our_tf_dist,
                pessimistic_estimate=True,
            )
            versions.append(
                {
                    "name": "TF_Our",
                    "pmf": our_tf_pmf,
                    "losses": base_losses,
                    "probs": our_tf_probs,
                }
            )

            gt_base_losses, gt_base_probs = compute_analytical_base_gaussian(
                sigma, discretization, remove_direction=remove_direction
            )
            gt_our_probs = subsample_losses(
                gt_base_losses, gt_base_probs, sampling_prob, remove_direction
            )
            gt_our_dist = DiscreteDist(
                x_array=gt_base_losses,
                PMF_array=gt_our_probs,
                p_neg_inf=0.0,
                p_pos_inf=max(0.0, 1.0 - float(np.sum(gt_our_probs))),
            )
            gt_our_pmf = discrete_dist_to_dp_accounting_pmf(
                gt_our_dist,
                pessimistic_estimate=True,
            )
            versions.append(
                {
                    "name": "GT_Our",
                    "pmf": gt_our_pmf,
                    "losses": gt_base_losses,
                    "probs": gt_our_probs,
                }
            )

        versions.append(
            {
                "name": "GT_GT",
                "pmf": analytic_pmf,
                "losses": analytic_losses,
                "probs": analytic_probs,
            }
        )

        for version in versions:
            version["eps"] = [version["pmf"].get_epsilon_for_delta(d) for d in delta_values]

        eps_gt = [analytic_pmf.get_epsilon_for_delta(d) for d in delta_values]
        return {
            "direction": direction_label,
            "versions": versions,
            "delta_values": np.asarray(delta_values, dtype=float),
            "eps_GT": np.asarray(eps_gt, dtype=float),
        }

    if direction in {"remove", "both"}:
        results["remove"] = _compute_direction_results(
            "remove", True
        )
    if direction in {"add", "both"}:
        results["add"] = _compute_direction_results(
            "add", False
        )

    return results


def run_subsampling_bounds_comparison_experiment(
    sigma: float,
    sampling_prob: float,
    discretization: float,
    sensitivity: float = 1.0,
    direction: str = "remove",
    delta_values: Optional[List[float]] = None,
    tail_truncation: float = 1e-12,
    loss_discretization: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare analytic, dp_accounting, and subsample_PMF-based bounds."""
    if direction not in {"remove", "add", "both"}:
        raise ValueError("direction must be 'remove', 'add', or 'both'")
    if delta_values is None:
        delta_values = [1e-6, 1e-8, 1e-10]

    if loss_discretization is None:
        loss_discretization = discretization

    results: Dict[str, Any] = {
        "params": {
            "sigma": sigma,
            "sampling_prob": sampling_prob,
            "discretization": discretization,
            "sensitivity": sensitivity,
            "direction": direction,
            "tail_truncation": tail_truncation,
            "loss_discretization": loss_discretization,
        },
    }

    def _compute_direction_results(remove_direction: bool) -> Dict[str, Any]:
        analytic_losses, analytic_probs = compute_analytical_subsampled_gaussian(
            sigma, sampling_prob, discretization, remove_direction=remove_direction
        )
        analytic_dist = DiscreteDist(
            x_array=analytic_losses,
            PMF_array=analytic_probs,
            p_neg_inf=0.0,
            p_pos_inf=max(0.0, 1.0 - float(np.sum(analytic_probs))),
        )
        analytic_pmf = discrete_dist_to_dp_accounting_pmf(
            analytic_dist,
            pessimistic_estimate=True,
        )

        dp_pld = privacy_loss_distribution.from_gaussian_mechanism(
            standard_deviation=sigma,
            sensitivity=sensitivity,
            sampling_prob=sampling_prob,
            value_discretization_interval=discretization,
            pessimistic_estimate=True,
        )
        dp_pmf = dp_pld._pmf_remove if remove_direction else dp_pld._pmf_add
        dp_losses, dp_probs = _pmf_to_loss_probs(dp_pmf)

        # Build discretized base distributions for both bounds.
        base_dist = stats.norm(
            loc=1 / (2 * sigma**2),
            scale=1 / sigma,
        )
        x_min = base_dist.ppf(tail_truncation)
        x_max = base_dist.isf(tail_truncation)
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            raise ValueError(f"Quantiles not finite: x_min={x_min}, x_max={x_max}")
        n_grid = max(int(np.ceil((x_max - x_min) / loss_discretization)), 2)
        base_x_array = np.linspace(x_min, x_max, n_grid)
        base_upper = discretize_continuous_to_pmf(
            dist=base_dist,
            x_array=base_x_array,
            bound_type=BoundType.DOMINATES,
            PMF_min_increment=tail_truncation,
        )

        direction_enum = Direction.REMOVE if remove_direction else Direction.ADD

        def _to_pmf_entry(name, dist, pessimistic):
            pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=pessimistic)
            losses, probs = _pmf_to_loss_probs(pmf)
            return {"name": name, "pmf": pmf, "losses": losses, "probs": probs}

        upper_dist = subsample_PMF(
            base_pld=base_upper,
            sampling_prob=sampling_prob,
            direction=direction_enum,
            bound_type=BoundType.DOMINATES,
        )
        lower_dist = subsample_PMF(
            base_pld=base_upper,
            sampling_prob=sampling_prob,
            direction=direction_enum,
            bound_type=BoundType.IS_DOMINATED,
        )

        versions = [
            {
                "name": "ANALYTIC",
                "pmf": analytic_pmf,
                "losses": analytic_losses,
                "probs": analytic_probs,
            },
            {
                "name": "DP_ACCOUNTING",
                "pmf": dp_pmf,
                "losses": dp_losses,
                "probs": dp_probs,
            },
            _to_pmf_entry("OUR_UPPER", upper_dist, pessimistic=True),
            _to_pmf_entry("OUR_LOWER", lower_dist, pessimistic=False),
        ]

        for version in versions:
            version["eps"] = [version["pmf"].get_epsilon_for_delta(d) for d in delta_values]
        eps_gt = [analytic_pmf.get_epsilon_for_delta(d) for d in delta_values]
        return {
            "versions": versions,
            "delta_values": np.asarray(delta_values, dtype=float),
            "eps_GT": np.asarray(eps_gt, dtype=float),
        }

    if direction in {"remove", "both"}:
        results["remove"] = _compute_direction_results(True)
    if direction in {"add", "both"}:
        results["add"] = _compute_direction_results(False)

    return results


def plot_subsampling_comparison(
    results: dict,
    direction: str = "remove",
    show_plots: bool = True,
    save_plots: bool = False,
    plots_dir: Optional[str] = None,
    filename_prefix: str = "subsampling_comparison"
):
    """
    Plot CDF and epsilon-ratio for the subsampling methods.

    Args:
        results: Results dictionary from run_subsampling_comparison_experiment
        show_plots: Whether to display plots
        save_plots: Whether to save plots to file
        plots_dir: Directory to save plots (required if save_plots=True)
        filename_prefix: Prefix for saved plot filenames
    """
    configure_matplotlib_for_publication(dpi=600, savefig_dpi=SAVEFIG_DPI)

    if direction not in {"remove", "add"}:
        raise ValueError("direction must be 'remove' or 'add'")
    params = results["params"]
    if direction not in results:
        raise KeyError(f"Missing '{direction}' results. Run experiment with direction='{direction}' or 'both'.")
    direction_results = results[direction]
    versions = direction_results["versions"]
    title_suffix = (
        f"sigma={params['sigma']}, q={params['sampling_prob']}, "
        f"disc={params['discretization']:.0e}, dir={direction.upper()}"
    )
    fig_cdf = create_pmf_cdf_plot(versions=versions, title_suffix=title_suffix)
    fig_eps = create_epsilon_delta_plot(
        delta_values=direction_results["delta_values"],
        versions=versions,
        eps_GT=direction_results["eps_GT"],
        log_x_axis=True,
        log_y_axis=False,
        title_suffix=title_suffix,
    )

    if save_plots:
        if plots_dir is None:
            raise ValueError("plots_dir must be provided when save_plots=True")
        cdf_name = f"{filename_prefix}_{direction}_cdf.png"
        eps_name = f"{filename_prefix}_{direction}_eps.png"
        fig_cdf.savefig(f"{plots_dir}/{cdf_name}")
        fig_eps.savefig(f"{plots_dir}/{eps_name}")

    show_or_close_figure(fig_cdf, show_plots=show_plots)
    show_or_close_figure(fig_eps, show_plots=show_plots)

# ============================
# Subsampling reference utils
# ============================

def stable_subsampling_loss(
    losses: np.ndarray,
    sampling_prob: float,
    remove_direction: bool = True,
) -> np.ndarray:
    """Stable inverse subsampling transform used by the reference GT code."""
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


def exclusive_padded_ccdf_from_pdf(probs: np.ndarray) -> np.ndarray:
    """Compute padded CCDF for numerical stability."""
    padded_probs = np.concatenate(([0.0], probs, [1.0 - np.sum(probs)]))
    flipped_padded_probs = np.flip(padded_probs)
    padded_cumsum = np.cumsum(flipped_padded_probs) - flipped_padded_probs
    flipped_padded_cumsum = np.flip(padded_cumsum)
    return flipped_padded_cumsum


def subsample_losses(
    losses: np.ndarray,
    probs: np.ndarray,
    sampling_prob: float,
    remove_direction: bool,
) -> np.ndarray:
    """Reference subsampling over a PMF grid (testing/visualization only)."""
    if sampling_prob < 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in [0, 1]")
    if sampling_prob == 1:
        return probs

    if remove_direction:
        lower_probs = np.zeros_like(probs)
        mask = probs > 0
        lower_probs[mask] = np.exp(np.log(probs[mask]) - losses[mask])
        lower_probs /= np.sum(lower_probs)
        mixed_probs = (1.0 - sampling_prob) * lower_probs + sampling_prob * probs
    else:
        mixed_probs = probs

    mix_ccdf = exclusive_padded_ccdf_from_pdf(mixed_probs)
    transformed_losses = stable_subsampling_loss(losses, sampling_prob, remove_direction)
    losses_step = float(np.mean(np.diff(losses)))
    indices = np.floor((transformed_losses - float(losses[0])) / losses_step)
    if remove_direction:
        indices = np.clip(indices, -1, losses.size - 1).astype(int)
    else:
        indices = np.clip(indices, -1, losses.size).astype(int)
    prev_indices = np.concatenate(([-1], indices[:-1]))
    return mix_ccdf[prev_indices + 1] - mix_ccdf[indices + 1]


# ============================
# Subsampling plot utilities
# ============================

def create_pmf_cdf_plot(versions: List[dict], title_suffix: str = ""):
    losses_list = []
    for entry in versions:
        losses = np.asarray(entry["losses"], dtype=np.float64)
        finite_mask = np.isfinite(losses)
        losses_list.append(losses[finite_mask])
    union_losses = np.unique(np.concatenate(losses_list)) if losses_list else np.array([], dtype=np.float64)
    union_losses = np.sort(union_losses[np.isfinite(union_losses)])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax_main = fig.add_subplot(gs[0, :])

    lines = []
    colors = ["r", "b", "g", "m", "c"]
    for idx, entry in enumerate(versions):
        name = entry.get("name", f"PMF {idx+1}")
        losses = np.asarray(entry["losses"], dtype=np.float64)
        probs = np.asarray(entry["probs"], dtype=np.float64)
        pmf_map = dict(zip(losses, probs))
        grid_probs = np.array([pmf_map.get(x, 0.0) for x in union_losses])
        cdf_vals = np.cumsum(grid_probs)
        color = colors[idx % len(colors)]
        lines.append({"label": name, "cdf": cdf_vals, "color": color, "style": "--", "alpha": 0.85})

    for line in lines:
        ax_main.plot(
            union_losses,
            line["cdf"],
            linestyle=line["style"],
            color=line["color"],
            alpha=line["alpha"],
            label=line["label"],
        )

    title = "CDF Comparison"
    if title_suffix:
        title += f" — {title_suffix}"
    ax_main.set_title(title)
    ax_main.set_xlabel("Privacy Loss")
    ax_main.set_ylabel("CDF")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    ax_main.set_ylim(-0.02, 1.02)

    finite_losses = union_losses
    tiny = 1e-16
    if finite_losses.size and len(lines) >= 1:
        cdfs = np.vstack([ln["cdf"] for ln in lines]) if len(lines) > 1 else np.vstack([lines[0]["cdf"], lines[0]["cdf"]])
        mean_cdf = np.mean(cdfs, axis=0)
        if union_losses.size > 1:
            min_loss = float(union_losses[0])
            max_loss = float(union_losses[-1])
            step_main = float(union_losses[1] - union_losses[0])
            trans_mask = (mean_cdf >= 1e-6) & (mean_cdf <= 1.0 - 1e-6)
            if np.any(trans_mask):
                idxs = np.where(trans_mask)[0]
                left_base = int(idxs[0])
                right_base = int(idxs[-1])
            else:
                grad = np.abs(np.diff(mean_cdf))
                if grad.size and np.max(grad) > 0:
                    mask = grad >= (1e-3 * np.max(grad))
                    idxs = np.where(mask)[0]
                    left_base = int(idxs[0])
                    right_base = int(idxs[-1] + 1)
                else:
                    left_base = 0
                    right_base = union_losses.size - 1
            n = union_losses.size
            left_base = max(0, left_base)
            right_base = min(n - 1, right_base)
            base_span = max(1e-12, float(union_losses[right_base] - union_losses[left_base]))
            buffer = max(0.05 * base_span, 5.0 * step_main)
            x_left = max(min_loss, float(union_losses[left_base]) - buffer)
            x_right = min(max_loss, float(union_losses[right_base]) + buffer)
            if x_right > x_left:
                ax_main.set_xlim(x_left, x_right)
        log_cdfs = np.log(np.maximum(cdfs, tiny))
        left_mask = mean_cdf <= 0.5
        left_range = np.max(log_cdfs, axis=0) - np.min(log_cdfs, axis=0)
        left_weighted = np.where(left_mask, left_range, -np.inf)
        ccdfs = np.maximum(1.0 - cdfs, tiny)
        mean_ccdf = np.mean(ccdfs, axis=0)
        log_ccdf = np.log(ccdfs)
        right_metric = np.max(log_ccdf, axis=0) - np.min(log_ccdf, axis=0)
        tail_thresholds = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
        right_mask = np.zeros_like(mean_ccdf, dtype=bool)
        for thr in tail_thresholds:
            cand_mask = mean_ccdf <= thr
            if np.any(cand_mask):
                right_mask = cand_mask
                break
        if not np.any(right_mask):
            right_mask = mean_cdf > 0.5
        right_weighted = np.where(right_mask, right_metric, -np.inf)
        left_idx = int(np.nanargmax(left_weighted)) if np.any(left_mask) else 0
        right_idx = int(np.nanargmax(right_weighted)) if np.any(right_mask) else (len(finite_losses) - 1)
        left_loss = float(finite_losses[left_idx])
        right_loss = float(finite_losses[right_idx])
    else:
        left_loss = 0.0
        right_loss = 0.0
        left_weighted = np.array([])
        right_weighted = np.array([])
        left_mask = np.array([], dtype=bool)
        right_mask = np.array([], dtype=bool)

    def window_from_metric(metric: np.ndarray, center_idx: int, mask: np.ndarray, frac: float = 0.1, pad_bins: int = 50):
        if metric.size == 0 or center_idx < 0:
            return None
        peak = float(metric[center_idx]) if np.isfinite(metric[center_idx]) else -np.inf
        if not np.isfinite(peak) or peak <= 0:
            return None
        thresh = peak * frac
        active = (metric >= thresh) & mask
        left_b = center_idx
        while left_b - 1 >= 0 and active[left_b - 1]:
            left_b -= 1
        right_b = center_idx
        n = metric.size
        while right_b + 1 < n and active[right_b + 1]:
            right_b += 1
        left_b = max(0, left_b - pad_bins)
        right_b = min(n - 1, right_b + pad_bins)
        return float(finite_losses[left_b]), float(finite_losses[right_b])

    ax_left = fig.add_subplot(gs[1, 0])
    for line in lines:
        ax_left.plot(finite_losses, line["cdf"], linestyle=line["style"], color=line["color"], alpha=0.8)
    ax_left.set_yscale("log")
    left_win = window_from_metric(left_weighted, left_idx, left_mask) if finite_losses.size else None
    if left_win is not None:
        left_min, left_max = left_win
    else:
        left_min = left_loss - 1.0
        left_max = left_loss + 1.0
    left_width = max(1e-12, left_max - left_min)
    left_min = left_loss - 0.2 * left_width
    left_max = left_loss + 0.8 * left_width
    if finite_losses.size:
        ax_left.set_xlim(max(float(finite_losses.min()), left_min), min(float(finite_losses.max()), left_max))
        mask_left = (finite_losses >= ax_left.get_xlim()[0]) & (finite_losses <= ax_left.get_xlim()[1])
        if np.any(mask_left):
            stacked = np.vstack([ln["cdf"][mask_left] for ln in lines]) if lines else np.zeros((1, np.sum(mask_left)))
            y_left_vals = stacked.flatten()
            pos = y_left_vals[y_left_vals > 0.0]
            y_min = float(np.max([np.min(pos) if pos.size else 1e-16, 1e-16]))
            y_max = float(np.max(y_left_vals)) if y_left_vals.size else 1.0
            ax_left.set_ylim(max(y_min * 0.8, 1e-16), min(y_max * 1.25, 1.0))
    ax_left.set_title("Left extreme (log CDF)")
    ax_left.set_xlabel("Privacy Loss")
    ax_left.set_ylabel("CDF (log)")
    ax_left.grid(True, which="both", alpha=0.3)

    ax_right = fig.add_subplot(gs[1, 1])
    for line in lines:
        ax_right.plot(finite_losses, line["cdf"], linestyle=line["style"], color=line["color"], alpha=0.8)
    if finite_losses.size and np.any(np.isfinite(right_weighted)):
        n = finite_losses.size
        pad_bins = max(5, min(100, n // 50))
        l_idx = max(0, right_idx - pad_bins)
        r_idx = min(n - 1, right_idx + pad_bins)
        right_min = float(finite_losses[l_idx])
        right_max = float(finite_losses[r_idx])
    else:
        right_min = right_loss - 1.0
        right_max = right_loss + 1.0
    right_width = max(1e-12, right_max - right_min)
    right_min = right_loss - 0.8 * right_width
    right_max = right_loss + 0.2 * right_width
    if finite_losses.size:
        ax_right.set_xlim(max(float(finite_losses.min()), right_min), min(float(finite_losses.max()), right_max))
        mask_right = (finite_losses >= ax_right.get_xlim()[0]) & (finite_losses <= ax_right.get_xlim()[1])
        if np.any(mask_right) and lines:
            stacked = np.vstack([ln["cdf"][mask_right] for ln in lines])
            one_minus = 1.0 - stacked
            pos = one_minus[one_minus > 0.0]
            if pos.size:
                min_one = float(np.max([np.min(pos), 1e-16]))
                max_one = float(np.max(pos))
                y_low = max(0.0, 1.0 - 1.25 * max_one)
                y_high = min(1.0, 1.0 - 0.8 * min_one)
                if y_high > y_low:
                    ax_right.set_ylim(y_low, y_high)

    def forward_cdf_to_log1mcdf(y):
        return -np.log10(np.maximum(1e-16, 1.0 - y))

    def inverse_log1mcdf_to_cdf(z):
        return 1.0 - np.power(10.0, -z)

    ax_right.set_yscale("function", functions=(forward_cdf_to_log1mcdf, inverse_log1mcdf_to_cdf))
    ax_right.set_title("Right extreme (CDF; scale: log(1−CDF))")
    ax_right.set_xlabel("Privacy Loss")
    ax_right.set_ylabel("CDF")
    try:
        ax_right.set_ylim(0.0, 1.0)
    except Exception:
        pass
    ax_right.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    return fig


def create_epsilon_delta_plot(delta_values, versions, eps_GT, log_x_axis: bool, log_y_axis: bool, title_suffix: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["r", "b", "g", "m", "c"]

    for idx, entry in enumerate(versions):
        name = entry.get("name", f"Method {idx+1}")
        eps_vals = entry.get("eps", [])
        color = colors[idx % len(colors)]
        if eps_vals is None or len(eps_vals) == 0:
            continue
        ax.plot(delta_values, eps_vals, label=name, color=color, alpha=0.8)

    if eps_GT is not None:
        ax.plot(delta_values, eps_GT, label="Ground Truth", color="k", linestyle="--", alpha=0.7)

    if log_x_axis:
        ax.set_xscale("log")
    if log_y_axis:
        ax.set_yscale("log")

    title = "Epsilon vs Delta"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Delta")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig

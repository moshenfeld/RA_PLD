import sys
import os
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (random_allocation)
parent_dir = os.path.dirname(current_dir)

# Get the parent of parent directory (the project root)
project_root = os.path.dirname(parent_dir)

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to sys.path")

from random_allocation.comparisons.utils import *
from random_allocation.comparisons.structs import *
from random_allocation.other_schemes.poisson import *
from random_allocation.random_allocation_scheme.recursive import *
from random_allocation.random_allocation_scheme.decomposition import *

from PLD_accounting.types import (
    PrivacyParams as ConvPrivacyParams,
    AllocationSchemeConfig as ConvAllocationSchemeConfig,
    Direction as ConvDirection,
    BoundType
)
from PLD_accounting.random_allocation_accounting import numerical_allocation_epsilon
from comparisons.data_management import save_experiment_data_json, load_experiment_data_json
from comparisons.plotting_utils import finalize_plot


POISSON_KEY = "Poisson (PLD)"
ALLOCATION_LEGACY_KEY = "Random allocation (FS25)"
ALLOCATION_NUMERICAL_KEY = "Random allocation (PLD)"

DEFAULT_METHOD_STYLES = {
    POISSON_KEY: {"color": "tab:blue", "marker": "o"},
    ALLOCATION_LEGACY_KEY: {"color": "tab:orange", "marker": "s"},
    ALLOCATION_NUMERICAL_KEY: {"color": "tab:green", "marker": "^"},
}


def Poisson_mean_estimation_vectorized(data, num_steps, sigma):
    """
    Vectorized implementation of Poisson mean estimation.
    
    Args:
        data: Array of shape (num_experiments, sample_size) containing the data
        num_steps: Number of steps in the scheme
        sigma: Noise parameter
    """
    num_experiments, sample_size = data.shape
    sampling_probability = 1.0/num_steps
    # Generate participation counts for all experiments at once
    num_participations = np.random.binomial(num_steps, sampling_probability, size=(num_experiments, sample_size))
    # Calculate mean for each experiment
    sums = np.sum(num_participations * data, axis=1)
    # Add noise to each mean
    noise = np.random.normal(0, sigma*np.sqrt(num_steps), size=num_experiments)
    return (sums + noise)/sample_size


def Poisson_accuracy(sampling_func, sample_size, num_experiments, num_steps, sigma, true_mean):
    """
    Calculates the accuracy of Poisson scheme, returning mean and standard deviation of squared errors.
    
    Args:
        sampling_func: Function to generate sample data
        sample_size: Size of each sample
        num_experiments: Number of experiments to run
        num_steps: Number of steps in the scheme
        sigma: Noise parameter
        true_mean: True mean to compare against
        
    Returns:
        tuple: (mean_error, std_error)
    """
    data = sampling_func(sample_size, num_experiments)
    # Get estimates for all experiments at once
    estimates = Poisson_mean_estimation_vectorized(data, num_steps, sigma)
    # Calculate squared errors
    errors = (estimates - true_mean) ** 2
    return np.mean(errors), np.std(errors)

def Poisson_epsilon(num_steps, sigma, delta):
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=1,
        num_epochs=1,
        delta=delta
)
    config = SchemeConfig()
    return Poisson_epsilon_PLD(params, config, direction=Direction.BOTH)

def Poisson_sigma(num_steps, epsilon, delta, lower = 0.1, upper = 10):
    optimization_func = lambda sig: Poisson_epsilon(num_steps=num_steps, sigma=sig, delta=delta) 
    
    sigma = search_function_with_bounds(
        func=optimization_func, 
        y_target=epsilon,
        bounds=(lower, upper),
        tolerance=0.05,
        function_type=FunctionType.DECREASING
    )
    if sigma is None:
        lower_epsilon = Poisson_epsilon(num_steps=num_steps, sigma=upper, delta=delta)
        upper_epsilon = Poisson_epsilon(num_steps=num_steps, sigma=lower, delta=delta)
        print(f"Poisson_sigma: lower_epsilon={lower_epsilon}, upper_epsilon={upper_epsilon}, target_epsilon={epsilon}")
        return np.inf
    return sigma


def allocation_mean_estimation_vectorized(data, num_steps, sigma):
    """
    Vectorized implementation of Random Allocation mean estimation.
    
    Args:
        data: Array of shape (num_experiments, sample_size) containing the data
        num_steps: Number of steps in the scheme
        sigma: Noise parameter
    """
    # Calculate means for each experiment
    sums = np.sum(data, axis=1)
    # Add noise to each mean
    noise = np.random.normal(0, sigma*np.sqrt(num_steps), size=sums.shape)
    return (sums + noise) / data.shape[1]

def allocation_accuracy(sampling_func, sample_size, num_experiments, num_steps, sigma, true_mean):
    """
    Calculates the accuracy of Random Allocation scheme, returning mean and standard deviation of errors.
    
    Args:
        sampling_func: Function to generate sample data
        sample_size: Size of each sample
        num_experiments: Number of experiments to run
        num_steps: Number of steps in the scheme
        sigma: Noise parameter
        true_mean: True mean to compare against
        
    Returns:
        tuple: (mean_error, std_error)
    """
    data = sampling_func(sample_size, num_experiments)
    # Get estimates for all experiments at once using vectorized implementation
    estimates = allocation_mean_estimation_vectorized(data, num_steps, sigma)
    # Calculate squared errors
    errors = (estimates - true_mean) ** 2
    return np.mean(errors), np.std(errors)

def allocation_epsilon(num_steps, sigma, delta):
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=1,
        num_epochs=1,
        delta=delta
)
    config = SchemeConfig()
    return allocation_epsilon_recursive(params, config, direction=Direction.BOTH)

def allocation_sigma(num_steps, epsilon, delta, lower = 0.1, upper = 10):
    optimization_func = lambda sig: allocation_epsilon(num_steps=num_steps, sigma=sig, delta=delta)
    sigma = search_function_with_bounds(
        func=optimization_func, 
        y_target=epsilon,
        bounds=(lower, upper),
        tolerance=0.05,
        function_type=FunctionType.DECREASING
    )
    if sigma is None:
        lower_epsilon = allocation_epsilon(num_steps=num_steps, sigma=upper, delta=delta)
        upper_epsilon = allocation_epsilon(num_steps=num_steps, sigma=lower, delta=delta)
        print(f"Allocation_sigma: lower_epsilon={lower_epsilon}, upper_epsilon={upper_epsilon}, target_epsilon={epsilon}")
        return np.inf    
    return sigma

def allocation_sigma_conv(
    num_steps: int,
    epsilon: float,
    delta: float,
    *,
    lower: float = 0.1,
    upper: float = 10.0,
    config: Optional[ConvAllocationSchemeConfig] = None,
    bound_type: BoundType = BoundType.DOMINATES
) -> float:
    """Invert numerical_allocation_epsilon to obtain sigma for numerical PLD bounds."""
    conv_config = config or ConvAllocationSchemeConfig()
    def optimization_func(sig: float) -> float:
        params = ConvPrivacyParams(
            sigma=sig,
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            delta=delta
)
        return numerical_allocation_epsilon(
            params=params,
            config=conv_config,
            direction=ConvDirection.BOTH,
            bound_type=bound_type
)

    sigma_val = search_function_with_bounds(
        func=optimization_func,
        y_target=epsilon,
        bounds=(lower, upper),
        tolerance=0.05,
        function_type=FunctionType.DECREASING
)
    if sigma_val is None:
        lower_epsilon = optimization_func(upper)
        upper_epsilon = optimization_func(lower)
        print(f"Allocation_sigma_conv: lower_epsilon={lower_epsilon}, upper_epsilon={upper_epsilon}, target_epsilon={epsilon}")
        return np.inf
    return sigma_val

def run_utility_experiments(
    epsilon: float,
    delta: float,
    num_steps: int,
    dimension: int,
    true_mean: float,
    num_experiments: int,
    sample_size_arr: np.ndarray,
    *,
    include_numerical_allocation: bool = True,
    allocation_conv_config: Optional[ConvAllocationSchemeConfig] = None
) -> Dict[str, Dict[str, np.ndarray | float]]:
    """Run utility simulation for Poisson and random allocation schemes.

    Returns accuracy/std arrays for each sampling scheme under the requested privacy bounds.
    """
    poisson_sigma_val = Poisson_sigma(num_steps, epsilon, delta) * np.sqrt(dimension)
    allocation_sigma_val = allocation_sigma(num_steps, epsilon, delta) * np.sqrt(dimension)
    allocation_conv_sigma_val = None
    if include_numerical_allocation:
        allocation_conv_sigma_val = allocation_sigma_conv(
            num_steps,
            epsilon,
            delta,
            config=allocation_conv_config
) * np.sqrt(dimension)

    sampling_func = lambda sample_size, num_experiments: np.random.binomial(1, true_mean, size=(num_experiments, sample_size))

    def _run_method(accuracy_fn, sigma_val, analytic_fn):
        means, stds = [], []
        for sample_size in sample_size_arr:
            mean_val, std_val = accuracy_fn(
                sampling_func,
                sample_size,
                num_experiments,
                num_steps,
                sigma_val,
                true_mean
)
            means.append(mean_val)
            stds.append(std_val)
        return {
            "accuracy": np.array(means),
            "std": np.array(stds),
            "analytic": analytic_fn(sample_size_arr, sigma_val),
            "sigma": sigma_val,
        }

    poisson_results = _run_method(
        Poisson_accuracy,
        poisson_sigma_val,
        lambda n, sigma_val: true_mean * (1 - true_mean) / n + true_mean / n + sigma_val**2 * num_steps / n**2
)
    allocation_results = _run_method(
        allocation_accuracy,
        allocation_sigma_val,
        lambda n, sigma_val: true_mean * (1 - true_mean) / n + sigma_val**2 * num_steps / n**2
)
    results: Dict[str, Dict[str, np.ndarray | float]] = {
        POISSON_KEY: poisson_results,
        ALLOCATION_LEGACY_KEY: allocation_results,
    }

    if include_numerical_allocation and allocation_conv_sigma_val is not None:
        results[ALLOCATION_NUMERICAL_KEY] = _run_method(
            allocation_accuracy,
            allocation_conv_sigma_val,
            lambda n, sigma_val: true_mean * (1 - true_mean) / n + sigma_val**2 * num_steps / n**2
)

    return results


def run_utility_comparison_suite(
    epsilon_values: List[float],
    dimension_values: List[int],
    sample_size_arr: np.ndarray,
    *,
    num_steps: int,
    delta: float,
    true_mean: float,
    num_experiments: int,
    num_std: int = 3,
    include_numerical_allocation: bool = True,
    allocation_conv_config: Optional[ConvAllocationSchemeConfig] = None,
    titles: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run utility comparisons across epsilon/dimension settings."""
    if len(epsilon_values) != len(dimension_values):
        raise ValueError("epsilon_values and dimension_values must have the same length")
    sample_size_arr = np.array(sample_size_arr)
    if allocation_conv_config is None:
        allocation_conv_config = ConvAllocationSchemeConfig(
            loss_discretization=0.002,
            tail_truncation=delta * 0.01,
            max_grid_mult=50_000
    )

    if titles is None:
        titles = [
            r"$\varepsilon$ = " + f"{eps:.1f}" + r", $d$ = " + f"{dim:,}"
            for eps, dim in zip(epsilon_values, dimension_values)
        ]

    experiments = []
    for eps, dim, title in zip(epsilon_values, dimension_values, titles):
        print(f"Running utility experiment: epsilon={eps}, dimension={dim}")
        data = run_utility_experiments(
            epsilon=eps,
            delta=delta,
            num_steps=num_steps,
            dimension=dim,
            true_mean=true_mean,
            num_experiments=num_experiments,
            sample_size_arr=sample_size_arr,
            include_numerical_allocation=include_numerical_allocation,
            allocation_conv_config=allocation_conv_config
)
        experiments.append(
            {"epsilon": eps, "dimension": dim, "title": title, "data": data}
        )

    return {
        "params": {
            "sample_size_arr": sample_size_arr,
            "num_steps": num_steps,
            "num_experiments": num_experiments,
            "delta": delta,
            "true_mean": true_mean,
            "num_std": num_std,
            "epsilon_values": epsilon_values,
            "dimension_values": dimension_values,
            "include_numerical_allocation": include_numerical_allocation,
            "allocation_conv_config": allocation_conv_config,
        },
        "titles": titles,
        "experiments": experiments,
    }

def plot_subplot_with_ci(ax, x_data, data, title, xlabel, ylabel, num_experiments, C=3, show_ci=False, method_styles=None):
    """
    Create a subplot with confidence interval-based visualization of error distributions.
    Uses standard deviation and standard error for confidence intervals.
    
    Args:
        ax: Matplotlib axis to plot on
        x_data: Array of x values (sample sizes)
        data: Dictionary containing results data per method
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        num_experiments: Number of experiments run
        C: Multiplier for the confidence interval (standard error = std/sqrt(n))
        show_ci: Whether to display confidence interval bands (default: True)
    """
    styles = {**DEFAULT_METHOD_STYLES, **(method_styles or {})}
    fallback_colors = iter(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))

    def _style_for(method_name):
        style = styles.get(method_name, {})
        color = style.get("color")
        marker = style.get("marker", "o")
        if color is None:
            color = next(fallback_colors, "gray")
        return color, marker

    std_error = lambda std: std / np.sqrt(num_experiments)

    for method_name, method_data in data.items():
        color, marker = _style_for(method_name)
        accuracy = np.array(method_data.get("accuracy", []))
        label_sigma = method_data.get("sigma")
        label = method_name

        ax.plot(
            x_data,
            accuracy,
            marker,
            color=color,
            markersize=8,
            label=label
)

        if show_ci and "std" in method_data:
            se = std_error(np.array(method_data["std"]))

        analytic = method_data.get("analytic")
        if analytic is not None:
            ax.plot(
                x_data,
                analytic,
                "--",
                color=color,
                label=f"{method_name} (analytic)"
)
    
    # Annotate per-method sigma in the subplot (inside axes to avoid x-label overlap)
    sigma_lines = []
    for method_name, method_data in data.items():
        sigma_val = method_data.get("sigma")
        if sigma_val is None or not np.isfinite(sigma_val):
            continue
        short_name = method_name.replace("Random allocation", "Allocation")
        sigma_lines.append(f"{short_name}: σ={sigma_val:.2f}")
    if sigma_lines:
        ax.text(
            0.02,
            0.02,
            "\n".join(sigma_lines),
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=14,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
)

    # Finalize plot formatting
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(xlabel, labelpad=1, fontsize=16)  # Small padding to keep label close to axis
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Remove individual legends for each subplot - we'll add one common legend
    # ax.legend()
    ax.grid(True, alpha=0.3)


# Comprehensive plot function that creates the complete visualization
def plot_utility_comparison(sample_size_arr, experiment_data_list, titles, num_steps, num_experiments, C=3, show_ci=False, method_styles=None):
    """
    Creates a comprehensive plot with subplots comparing Poisson and Random Allocation
    schemes using standard deviation-based confidence intervals.
    
    Args:
        sample_size_arr: Array of sample sizes 
        experiment_data_list: List of dictionaries containing results from experiments
        titles: List of titles for each subplot
        num_steps: Number of steps used in the experiment
        num_experiments: Number of experiments run
        C: Multiplier for the confidence interval (standard error = std/sqrt(n))
        show_ci: Whether to display confidence interval bands (default: True)
    """
    # Create figure and subplots
    fig_width = 18 if len(experiment_data_list) <= 3 else 20
    fig_height = 4.6
    fig, axs = plt.subplots(1, len(experiment_data_list), figsize=(fig_width, fig_height))
    
    # Plot each experiment in its own subplot
    for i, (data, title) in enumerate(zip(experiment_data_list, titles)):
        plot_subplot_with_ci(
            axs[i], sample_size_arr, data, 
            title, "Sample Size", "Square Error", 
            num_experiments=num_experiments,
            C=C,
            show_ci=show_ci,
            method_styles=method_styles
)

    # Create a better positioned legend with clearer organization
    handles, labels = axs[0].get_legend_handles_labels()

    # Add a single legend below all subplots (slightly larger and lower)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        fontsize=15,
        frameon=True,
        framealpha=0.9,
        handlelength=2.8,
        handletextpad=0.7,
        columnspacing=1.4
)

    # Tight layout with extra bottom margin for the legend
    plt.tight_layout(rect=(0.02, 0.12, 1, 0.95))
    
    return fig


def plot_utility_comparison_results(
    results: Dict[str, Any],
    *,
    show_ci: bool = False,
    save_plots: bool = False,
    plots_dir: Optional[str] = None,
    filename: str = "utility_comparison.png",
    method_styles=None
):
    """Wrapper to plot results produced by run_utility_comparison_suite."""
    params = results.get("params", {})
    sample_size_arr = np.array(params.get("sample_size_arr"))
    num_steps = params.get("num_steps")
    num_experiments = params.get("num_experiments")
    num_std = params.get("num_std", 3)
    # Remap legacy method names in stored results (for backward-compatible pickles)
    name_map = {
        "Poisson": POISSON_KEY,
        "Random allocation (legacy)": ALLOCATION_LEGACY_KEY,
        "Random allocation (numerical)": ALLOCATION_NUMERICAL_KEY,
    }
    experiment_data_list = []
    for exp in results.get("experiments", []):
        data = exp["data"]
        remapped = {name_map.get(k, k): v for k, v in data.items()}
        experiment_data_list.append(remapped)
    titles = [exp.get("title", f"ε={exp.get('epsilon')}, d={exp.get('dimension')}") for exp in results.get("experiments", [])]

    fig = plot_utility_comparison(
        sample_size_arr=sample_size_arr,
        experiment_data_list=experiment_data_list,
        titles=titles,
        num_steps=num_steps,
        num_experiments=num_experiments,
        C=num_std,
        show_ci=show_ci,
        method_styles=method_styles
)

    finalize_plot(fig, save_plots=save_plots, show_plots=True,
                  plots_dir=plots_dir, filename=filename)

    return fig

def save_utility_experiment_data(results: Dict[str, Any], experiment_name: str):
    """Save utility comparison results to JSON/CSV for reuse outside pickles."""
    def _write_csvs(results, experiment_name):
        """Write per-experiment CSVs for quick inspection."""
        params = results.get("params", {})
        sample_sizes = np.array(params.get("sample_size_arr"))
        experiments = results.get("experiments", [])

        def _slugify(name: str) -> str:
            return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")

        for idx, exp in enumerate(experiments):
            title = exp.get("title", f"exp_{idx}")
            epsilon = exp.get("epsilon")
            dimension = exp.get("dimension")
            for method_name, method_data in exp.get("data", {}).items():
                df = pd.DataFrame(
                    {
                        "sample_size": sample_sizes,
                        "accuracy": method_data.get("accuracy"),
                        "std": method_data.get("std"),
                        "analytic": method_data.get("analytic"),
                        "epsilon": epsilon,
                        "dimension": dimension,
                        "method": method_name,
                        "title": title,
                    }
                )
                csv_path = f"{experiment_name}_{idx}_{_slugify(method_name)}.csv"
                df.to_csv(csv_path, index=False)
        print(f"Saved CSV files.")

    save_experiment_data_json(results, experiment_name, data_converter=_write_csvs)


def load_utility_experiment_data(experiment_name: str) -> Optional[Dict[str, Any]]:
    """Load JSON-saved utility experiment results."""
    return load_experiment_data_json(experiment_name)

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from random_allocation.comparisons.structs import SchemeConfig
from random_allocation.random_allocation_scheme.RDP_DCO import allocation_epsilon_RDP_DCO
from random_allocation.random_allocation_scheme.combined import allocation_epsilon_combined
from random_allocation.comparisons.experiments import Poisson_epsilon_PLD

from PLD_accounting.types import AllocationSchemeConfig, PrivacyParams, Direction, BoundType
from PLD_accounting.random_allocation_accounting import numerical_allocation_epsilon
from comparisons.comparison_utils import to_poisson_params
from comparisons.visualization_definitions import *
from comparisons.plotting_utils import create_method_comparison_figure

def run_epsilon_from_sigma(num_steps_arr: list[int],
                           sigma_arr: list[float],
                           num_selected: int,
                           num_epochs: int,
                           delta: float,
                           allocation_RDP_config: SchemeConfig,
                           allocation_conv_config: AllocationSchemeConfig,
                           Poisson_config: SchemeConfig) -> dict:
    """Run epsilon comparisons across sigma ranges for given num_steps values.

    Compares privacy guarantees for Poisson sampling vs. allocation schemes.
    Implementation:
    - Evaluates epsilon across sigma_arr for each num_steps value in num_steps_arr
    - Methods: Poisson PLD, Allocation RDP-DCO, Allocation FS25, Allocation conv (upper/lower)
    - Times each method and stores results
    - Returns dictionary with configuration parameters and per-num_steps epsilon arrays
    
    Args:
        num_steps_arr: List of num_steps values to evaluate. Can be a single element for single experiment.
    """
    results = {'params': {'num_selected': num_selected,
                          'num_epochs': num_epochs,
                          'delta': delta,
                          'allocation_loss_discretization': allocation_conv_config.loss_discretization,
                          'allocation_tail_truncation': allocation_conv_config.tail_truncation,
                          'Poisson_discretization': Poisson_config.discretization}}
    
    for num_steps in num_steps_arr:
        print(f"Running for num_steps = {num_steps}")
        start_time = time.time()
        Poisson_eps_arr = [Poisson_epsilon_PLD(params=to_poisson_params(PrivacyParams(sigma=sigma,
                                                                    num_steps=num_steps,
                                                                    num_selected=num_selected,
                                                                    num_epochs=num_epochs,
                                                                    delta=delta)),
                                               config=Poisson_config)
                           for sigma in sigma_arr]
        curr_time = time.time()
        print(f"\t Poisson_epsilon_PLD time: {curr_time - start_time:.0f}")

        start_time = time.time()
        allocation_DCO_eps_arr = [allocation_epsilon_RDP_DCO(params=to_poisson_params(PrivacyParams(sigma=sigma,
                                                                                  num_steps=num_steps,
                                                                                  num_selected=num_selected,
                                                                                  num_epochs=num_epochs,
                                                                                  delta=delta)),
                                                             config=allocation_RDP_config)
                                  for sigma in sigma_arr]
        curr_time = time.time()
        print(f"\t allocation_epsilon_RDP_DCO time: {curr_time - start_time:.0f}")

        start_time = time.time()
        allocation_FS25_eps_arr = [allocation_epsilon_combined(params=to_poisson_params(PrivacyParams(sigma=sigma,
                                                                                    num_steps=num_steps,
                                                                                    num_selected=num_selected,
                                                                                    num_epochs=num_epochs,
                                                                                    delta=delta)),
                                                               config=allocation_RDP_config)
                                   for sigma in sigma_arr]
        curr_time = time.time()
        print(f"\t allocation_epsilon_combined time: {curr_time - start_time:.0f}")

        start_time = time.time()
        # Compute allocation PLD upper bounds (pessimistic estimates)
        allocation_conv_upper_eps_arr = [numerical_allocation_epsilon(params=PrivacyParams(sigma=sigma,
                                                                                            num_steps=num_steps,
                                                                                            num_selected=num_selected,
                                                                                            num_epochs=num_epochs,
                                                                                            delta=delta),
                                                                       config=allocation_conv_config,
                                                                       bound_type=BoundType.DOMINATES)
                                         for sigma in sigma_arr]
        curr_time = time.time()
        print(f"\t numerical_allocation_epsilon (upper bound) time: {curr_time - start_time:.0f}")

        start_time = time.time()
        # Compute allocation PLD lower bounds (optimistic estimates)
        allocation_conv_lower_eps_arr = [numerical_allocation_epsilon(params=PrivacyParams(sigma=sigma,
                                                                                            num_steps=num_steps,
                                                                                            num_selected=num_selected,
                                                                                            num_epochs=num_epochs,
                                                                                            delta=delta),
                                                                       config=allocation_conv_config,
                                                                       bound_type=BoundType.IS_DOMINATED)
                                         for sigma in sigma_arr]
        curr_time = time.time()
        print(f"\t numerical_allocation_epsilon (lower bound) time: {curr_time - start_time:.0f}")

        results[num_steps] = {
            'sigma_arr': sigma_arr,
            'Poisson (PLD)': Poisson_eps_arr,
            'Allocation (DCO25)': allocation_DCO_eps_arr,
            'Allocation (FS25)': allocation_FS25_eps_arr,
            'Allocation (PLD, upper)': allocation_conv_upper_eps_arr,
            'Allocation (PLD, lower)': allocation_conv_lower_eps_arr,
        }
    return results


def run_epsilon_from_sigma_by_k(num_selected_arr: list[int],
                                sigma_arr: list[float],
                                num_steps: int,
                                num_epochs: int,
                                delta: float,
                                allocation_RDP_config: SchemeConfig,
                                allocation_conv_config: AllocationSchemeConfig,
                                Poisson_config: SchemeConfig) -> dict:
    """Run epsilon-from-sigma experiments for multiple k values at fixed t.

    Executes the original run_epsilon_from_sigma workflow for each k (num_selected)
    while keeping num_steps fixed, and packages the per-k results together so they
    can be visualized side-by-side.
    """
    params = {
        'num_steps': num_steps,
        'sigma_arr': sigma_arr,
        'num_selected_values': num_selected_arr,
        'num_epochs': num_epochs,
        'delta': delta,
        'allocation_loss_discretization': allocation_conv_config.loss_discretization,
        'allocation_tail_truncation': allocation_conv_config.tail_truncation,
        'Poisson_discretization': Poisson_config.discretization
    }

    results = {'params': params}
    for k in num_selected_arr:
        print(f"Running epsilon_from_sigma for k = {k}, t = {num_steps}")
        single_result = run_epsilon_from_sigma(
            num_steps_arr=[num_steps],
            sigma_arr=sigma_arr,
            num_selected=k,
            num_epochs=num_epochs,
            delta=delta,
            allocation_RDP_config=allocation_RDP_config,
            allocation_conv_config=allocation_conv_config,
            Poisson_config=Poisson_config
)
        results[k] = single_result[num_steps]

    return results


def run_epsilon_vs_k(num_selected_arr: list[int],
                     sigma: float,
                     num_steps: int,
                     num_epochs: int,
                     delta: float,
                     allocation_RDP_config: SchemeConfig,
                     allocation_conv_config: AllocationSchemeConfig,
                     Poisson_config: SchemeConfig) -> dict:
    """Run epsilon comparisons across varying k for fixed sigma and t."""
    params = {
        'sigma': sigma,
        'num_steps': num_steps,
        'num_selected_values': num_selected_arr,
        'num_epochs': num_epochs,
        'delta': delta,
        'allocation_loss_discretization': allocation_conv_config.loss_discretization,
        'allocation_tail_truncation': allocation_conv_config.tail_truncation,
        'Poisson_discretization': Poisson_config.discretization
    }

    method_names = [
        'Poisson (PLD)',
        'Allocation (DCO25)',
        'Allocation (FS25)',
        'Allocation (PLD, upper)',
        'Allocation (PLD, lower)',
    ]
    results = {'params': params, 'num_selected_arr': num_selected_arr}
    for name in method_names:
        results[name] = []

    for k in num_selected_arr:
        print(f"Running epsilon_vs_k for k = {k}, t = {num_steps}")
        poisson_params = to_poisson_params(PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=k,
            num_epochs=num_epochs,
            delta=delta
))
        allocation_params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=k,
            num_epochs=num_epochs,
            delta=delta
)

        results['Poisson (PLD)'].append(Poisson_epsilon_PLD(params=poisson_params, config=Poisson_config))
        results['Allocation (DCO25)'].append(allocation_epsilon_RDP_DCO(params=poisson_params, config=allocation_RDP_config))
        results['Allocation (FS25)'].append(allocation_epsilon_combined(params=poisson_params, config=allocation_RDP_config))
        results['Allocation (PLD, upper)'].append(
            numerical_allocation_epsilon(params=allocation_params, config=allocation_conv_config, bound_type=BoundType.DOMINATES)
        )
        results['Allocation (PLD, lower)'].append(
            numerical_allocation_epsilon(params=allocation_params, config=allocation_conv_config, bound_type=BoundType.IS_DOMINATED)
        )

    return results


def plot_epsilon_from_sigma(results, save_plots: bool = False, show_plots: bool = True,
                           plots_dir: str = None, filename: str = 'epsilon_from_sigma.png'):
    """Visualize epsilon vs. sigma across different num_steps configurations."""
    if 'params' not in results:
        raise KeyError("Results dictionary must contain 'params' key")

    num_steps_values = sorted([t for t in results.keys() if t != 'params'])
    if not num_steps_values:
        raise ValueError("Results dictionary must contain at least one num_steps entry")

    num_selected = results['params'].get('num_selected', None)

    # Prepare data for plotting
    data_by_config = {}
    x_arrays_by_config = {}
    config_titles = {}

    for num_steps in num_steps_values:
        sigma_arr = results[num_steps].get('sigma_arr')
        if sigma_arr is None:
            raise KeyError(f"Results for num_steps={num_steps} missing 'sigma_arr'")

        x_arrays_by_config[num_steps] = np.array(sigma_arr)
        method_data = {m: np.array(results[num_steps][m]) for m in results[num_steps].keys() if m != 'sigma_arr'}
        data_by_config[num_steps] = method_data

        title_suffix = f", $k$ = {num_selected}" if num_selected is not None else ""
        config_titles[num_steps] = f'$t$ = {num_steps}{title_suffix}'

    # Use the high-level plotting function
    fig = create_method_comparison_figure(
        data_by_config=data_by_config,
        x_arrays_by_config=x_arrays_by_config,
        config_titles=config_titles,
        xlabel='σ',
        ylabel='ε',
        layout='horizontal' if len(num_steps_values) <= 3 else 'auto',
        figsize_per_plot=(6, 5),
        linewidth=1.5,
        markersize=4,
        legend_fontsize=14,
        legend_ncol=5,
        legend_bottom_margin=0.03,
        save_plots=save_plots,
        show_plots=show_plots,
        plots_dir=plots_dir,
        filename=filename
    )

    return fig


def plot_epsilon_from_sigma_by_k(results, save_plots: bool = False, show_plots: bool = True,
                                 plots_dir: str = None, filename: str = 'epsilon_from_sigma_by_k.png'):
    """Visualize epsilon vs. sigma for multiple k values (fixed t) in subplots."""
    if 'params' not in results:
        raise KeyError("Results dictionary must contain 'params' key")

    k_values = sorted([k for k in results.keys() if k != 'params'])
    if not k_values:
        raise ValueError("Results dictionary must contain at least one k entry")

    num_steps = results['params'].get('num_steps', None)

    # Prepare data for plotting
    data_by_config = {}
    x_arrays_by_config = {}
    config_titles = {}

    for k in k_values:
        sigma_arr = results[k].get('sigma_arr')
        if sigma_arr is None:
            raise KeyError(f"Results for k={k} missing 'sigma_arr'")

        x_arrays_by_config[k] = np.array(sigma_arr)
        method_data = {m: np.array(results[k][m]) for m in results[k].keys() if m != 'sigma_arr'}
        data_by_config[k] = method_data

        title_suffix = f", $t$ = {num_steps}" if num_steps is not None else ""
        config_titles[k] = f'$k$ = {k}{title_suffix}'

    # Use the high-level plotting function
    fig = create_method_comparison_figure(
        data_by_config=data_by_config,
        x_arrays_by_config=x_arrays_by_config,
        config_titles=config_titles,
        xlabel='σ',
        ylabel='ε',
        layout='horizontal' if len(k_values) <= 3 else 'auto',
        figsize_per_plot=(6, 5),
        linewidth=1.5,
        markersize=4,
        legend_fontsize=12,
        legend_ncol=5,
        legend_bottom_margin=0.03,
        save_plots=save_plots,
        show_plots=show_plots,
        plots_dir=plots_dir,
        filename=filename
    )

    return fig


def plot_epsilon_vs_k(results, save_plots: bool = False, show_plots: bool = True,
                      plots_dir: str = None, filename: str = 'epsilon_vs_k.png'):
    """Plot epsilon vs. k for a fixed sigma and t."""
    if 'params' not in results:
        raise KeyError("Results dictionary must contain 'params' key")

    k_arr = results.get('num_selected_arr')
    if k_arr is None:
        raise KeyError("Results dictionary must contain 'num_selected_arr' key")

    # Extract method data
    method_data = {m: np.array(results[m]) for m in results.keys() if m not in ('params', 'num_selected_arr')}
    if not method_data:
        raise ValueError("Results dictionary must contain method data for plotting")

    # Create title from parameters
    sigma = results['params'].get('sigma')
    num_steps = results['params'].get('num_steps')
    title_parts = []
    if sigma is not None:
        title_parts.append(f'$\\sigma$ = {sigma}')
    if num_steps is not None:
        title_parts.append(f'$t$ = {num_steps}')
    title = ', '.join(title_parts) if title_parts else 'ε vs k'

    # Use the high-level plotting function with single config
    fig = create_method_comparison_figure(
        data_by_config={'epsilon_vs_k': method_data},
        x_arrays_by_config={'epsilon_vs_k': np.array(k_arr)},
        config_titles={'epsilon_vs_k': title},
        xlabel='k',
        ylabel='ε',
        layout='auto',
        figsize_per_plot=(6, 5),
        linewidth=1.5,
        markersize=4,
        legend_fontsize=10,
        legend_ncol=2,
        legend_bottom_margin=0.03,
        save_plots=save_plots,
        show_plots=show_plots,
        plots_dir=plots_dir,
        filename=filename
    )

    return fig

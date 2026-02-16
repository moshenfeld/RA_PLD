import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Optional

from random_allocation.comparisons.structs import SchemeConfig, PrivacyParams
from random_allocation.random_allocation_scheme.combined import allocation_delta_combined
from random_allocation.random_allocation_scheme.lower_bound import allocation_delta_lower_bound
from random_allocation.random_allocation_scheme.Monte_Carlo import Monte_Carlo_estimation
from random_allocation.random_allocation_scheme.Monte_Carlo_external import AdjacencyType
from random_allocation.comparisons.experiments import Poisson_delta_PLD

from PLD_accounting.types import AllocationSchemeConfig, Direction, PrivacyParams
from PLD_accounting.random_allocation_accounting import allocation_PLD
from PLD_accounting.types import BoundType
from comparisons.comparison_utils import to_poisson_params
from comparisons.visualization_definitions import *
from comparisons.plotting_utils import configure_matplotlib_for_publication, finalize_plot, create_method_comparison_figure

def calc_method_delta(name: str,
                     func: Callable,
                     params_arr: List,
                     config) -> List[float]:
    """
    Calculate delta values for a specific method.

    Args:
        name: Name of the method
        func: Function to calculate delta
        params_arr: Array of PrivacyParams objects
        config: Scheme configuration parameters
        direction: The direction of privacy. Can be ADD, REMOVE, or BOTH.

    Returns:
        List of delta values for each set of parameters
    """
    time_start = time.perf_counter()
    results = [func(params=params, config=config) for params in params_arr]
    time_stop = time.perf_counter()
    print(f'{name} delta done in {time_stop - time_start: .0f} seconds')
    return results


def calc_all_methods_delta(project_params: PrivacyParams,
                           poisson_config: SchemeConfig,
                           allocation_RDP_config: SchemeConfig,
                           allocation_conv_config: AllocationSchemeConfig,
                           epsilon_arr: List[float]) -> Dict[str, List[float]]:
    """
    Calculate delta values for all methods.

    Args:
        poisson_params_arr: Array of Poisson package's PrivacyParams objects
        project_params: project's PrivacyParams object
        poisson_config: Configuration for Poisson sampling
        allocation_RDP_config: Configuration for RDP-based allocation methods
        allocation_conv_config: Configuration for convolution-based allocation methods

    Returns:
        Dictionary mapping method names to lists of delta values
    """
    print(f'Calculating deltas with sigma = {project_params.sigma}, t = {project_params.num_steps} for all methods...')
    deltas_dict = {}

    # Poisson sampling baseline - uses Poisson package params
    poisson_params_arr = [to_poisson_params(PrivacyParams(num_steps=project_params.num_steps,
                                                          sigma=project_params.sigma,
                                                          num_selected=project_params.num_selected,
                                                          num_epochs=project_params.num_epochs,
                                                          epsilon=epsilon)) 
                                            for epsilon in epsilon_arr]
    deltas_dict['Poisson (PLD)'] = calc_method_delta('Poisson (PLD)', Poisson_delta_PLD, poisson_params_arr, poisson_config)

    # Allocation - FS25 method (our combined method from the paper) - uses Poisson package params
    deltas_dict['Allocation (FS25)'] = calc_method_delta('Allocation (FS25)', allocation_delta_combined, poisson_params_arr, allocation_RDP_config)

    # Monte Carlo estimation - returns both HP and mean - uses Poisson package params
    time_start = time.perf_counter()
    MC_dict_arr = [Monte_Carlo_estimation(params, allocation_RDP_config, adjacency_type=AdjacencyType.REMOVE) for params in poisson_params_arr]
    time_stop = time.perf_counter()
    deltas_dict['Allocation (MC, high prob)'] = [MC_dict['high prob'] for MC_dict in MC_dict_arr]
    deltas_dict['Allocation (MC, mean)'] = [MC_dict['mean'] for MC_dict in MC_dict_arr]
    print(f'Allocation (MC, high prob and mean) delta done in {time_stop - time_start: .0f} seconds')

    # Allocation - Lower bound - uses Poisson package params
    deltas_dict['Allocation (lower bound)'] = calc_method_delta('Allocation (lower bound)', allocation_delta_lower_bound, poisson_params_arr, allocation_RDP_config)

    # Allocation - Numerical (convolution-based) - uses PROJECT params
    time_start = time.perf_counter()
    PLD = allocation_PLD(params=project_params, config=allocation_conv_config, direction=Direction.BOTH)
    time_stop = time.perf_counter()
    deltas_dict['Allocation (PLD)'] = [PLD.get_delta_for_epsilon(epsilon) for epsilon in epsilon_arr]
    print(f'Allocation (PLD) delta done in {time_stop - time_start: .0f} seconds')

    print(f'Calculation done for {len(deltas_dict)} methods')
    return deltas_dict


def run_delta_comparison(num_steps_arr: list[int],
                        sigma_arr: list[float],
                        epsilon_arr_list: list,
                        num_selected: int,
                        num_epochs: int,
                        allocation_RDP_config: SchemeConfig,
                        allocation_conv_config: AllocationSchemeConfig,
                        Poisson_config: SchemeConfig,
                        direction: Direction = Direction.BOTH) -> dict:
    """
    Run delta comparison experiment across multiple parameter configurations.

    Similar to epsilon_from_sigma but computes delta given epsilon instead of epsilon given delta.
    This follows the structure of the Chau et al. Monte Carlo comparison experiments.

    Args:
        num_steps_arr: Array of num_steps values
        sigma_arr: Array of sigma values (one per num_steps)
        epsilon_arr_list: List of epsilon arrays (one per configuration)
        num_selected: Number of selected samples
        num_epochs: Number of epochs
        direction: Privacy direction
        allocation_RDP_config: Config for RDP-based allocation methods
        allocation_conv_config: Config for convolution-based allocation methods
        Poisson_config: Config for Poisson sampling

    Returns:
        Dictionary with configuration parameters and per-configuration delta arrays
    """
    results = {
        'params': {
            'num_selected': num_selected,
            'num_epochs': num_epochs,
            'direction': direction,
            'allocation_loss_discretization': allocation_conv_config.loss_discretization,
            'allocation_tail_truncation': allocation_conv_config.tail_truncation,
            'Poisson_discretization': Poisson_config.discretization
        }
    }

    for num_steps, sigma, epsilon_arr in zip(num_steps_arr, sigma_arr, epsilon_arr_list):
        print(f"\nRunning for num_steps = {num_steps}, sigma = {sigma}")

        # Create project parameter arrays
        project_params = PrivacyParams(num_steps=num_steps,
                                       sigma=sigma,
                                       num_selected=num_selected,
                                       num_epochs=num_epochs)

        # Calculate deltas for all methods
        deltas_dict = calc_all_methods_delta(project_params, Poisson_config, allocation_RDP_config, allocation_conv_config, epsilon_arr)

        # Store results
        key = f"t={num_steps}_sigma={sigma}"
        results[key] = {
            'epsilon_arr': epsilon_arr,
            'num_steps': num_steps,
            'sigma': sigma,
            **deltas_dict
        }

    return results

def plot_delta_comparison(results,
                         save_plots: bool = False,
                         show_plots: bool = True,
                         plots_dir: Optional[str] = None,
                         filename: str = 'delta_comparison.png'):
    """
    Create delta comparison plot with subplots for each configuration.

    Args:
        results: Dictionary containing delta comparison results
        save_plots: Whether to save the plot
        show_plots: Whether to display the plot
        plots_dir: Directory to save plots (required if save_plots=True)
        filename: Filename for the saved plot

    Returns:
        The created matplotlib figure
    """
    # Validate results structure
    if 'params' not in results:
        raise KeyError("Results dictionary must contain 'params' key")

    # Extract configuration keys (excluding 'params')
    config_keys = [k for k in results.keys() if k != 'params']

    if len(config_keys) == 0:
        raise ValueError("Results dictionary has no configuration data (only 'params' key found)")

    # Prepare data for the high-level plotting function
    data_by_config = {}
    x_arrays_by_config = {}
    config_titles = {}

    for config_key in config_keys:
        config_data = results[config_key]

        # Validate required keys
        required_keys = ['epsilon_arr', 'num_steps', 'sigma']
        for req_key in required_keys:
            if req_key not in config_data:
                raise KeyError(f"Results[{config_key}] missing required key: {req_key}")

        # Extract x-axis data (epsilon values)
        x_arrays_by_config[config_key] = np.array(config_data['epsilon_arr'])

        # Extract method data (all keys except the required ones)
        method_data = {k: np.array(v) for k, v in config_data.items() if k not in required_keys}
        data_by_config[config_key] = method_data

        # Create title with formatted num_steps
        num_steps = config_data['num_steps']
        sigma = config_data['sigma']
        config_titles[config_key] = f"t = {num_steps:,}, σ = {sigma}"

    # Use the high-level plotting function
    fig = create_method_comparison_figure(
        data_by_config=data_by_config,
        x_arrays_by_config=x_arrays_by_config,
        config_titles=config_titles,
        xlabel='ε',
        ylabel='δ',
        yscale='log',  # Delta plots use log scale
        layout='auto',
        figsize_per_plot=(8, 6),  # Slightly larger than epsilon plots
        linewidth=2.0,
        markersize=5,
        legend_fontsize=16,  # Match original delta plot legend size
        legend_ncol=2,  # Match original delta plot
        save_plots=save_plots,
        show_plots=show_plots,
        plots_dir=plots_dir,
        filename=filename
    )

    return fig

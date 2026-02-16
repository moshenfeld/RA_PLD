import os
import time
import numpy as np
import matplotlib.pyplot as plt

from PLD_accounting.types import AllocationSchemeConfig, Direction, PrivacyParams
from PLD_accounting.random_allocation_accounting import allocation_PLD
from PLD_accounting.types import BoundType
from random_allocation.random_allocation_scheme.direct import allocation_RDP_remove
from PLD_accounting.dp_accounting_support import dp_accounting_pmf_to_discrete_dist
from comparisons.plotting_utils import finalize_plot


def calc_PMF_RDP(PLD_PMF, alpha):
    """Compute RDP at order alpha from privacy loss distribution PMF.

    Converts PLD PMF to (loss, prob) pairs and evaluates RDP formula:
    RDP(α) = log(E[exp((α-1)*L)]) / (α-1) where L is privacy loss.
    Implementation filters to positive probabilities and uses log-space for stability.
    """
    # Convert PMF to DiscreteDist using the standard conversion function
    dist = dp_accounting_pmf_to_discrete_dist(PLD_PMF)

    losses = dist.x_array
    probs = np.array(dist.PMF_array, dtype=np.float64)

    pos_ind = probs > 0
    return np.log(np.sum(np.exp(losses[pos_ind]*(alpha-1) + np.log(probs[pos_ind]))))/(alpha-1)


def run_runtime_experiment(loss_discretization_arr: list[float],
                          num_steps_arr: list[int],
                          sigma: float,
                          num_selected: int,
                          num_epochs: int,
                          delta: float,
                          tail_truncation: float,
                          alpha: float = 2.0) -> dict:
    """Run runtime experiment for the geometric method.

    Runs all combinations of loss_discretization_arr × num_steps_arr and measures runtime.
    Also computes RDP values for comparison with analytical results.
    Returns a dictionary with raw runtime measurements that can be plotted in different ways.

    Args:
        loss_discretization_arr: Array of discretization values to test
        num_steps_arr: Array of num_steps values to test
        sigma: Fixed sigma value for all tests
        num_selected: Number of selected samples
        num_epochs: Number of epochs
        delta: Privacy delta parameter
        tail_truncation: Tail truncation parameter for allocation config
        alpha: RDP order for RDP calculations (default: 2.0)

    Returns:
        Dictionary containing:
        - params: Configuration parameters
        - loss_discretization_arr: The discretization values tested
        - num_steps_arr: The num_steps values tested
        - runtimes: 2D array where runtimes[i][j] is runtime for loss_discretization_arr[i] and num_steps_arr[j]
        - rdp_values: 2D array where rdp_values[i][j] is RDP for loss_discretization_arr[i] and num_steps_arr[j]
        - analytical_rdp: Dictionary mapping num_steps to analytical RDP values
    """
    # Calculate analytical RDP for each num_steps value
    analytical_rdp = {}
    for num_steps in num_steps_arr:
        analytical_rdp[int(num_steps)] = allocation_RDP_remove(int(alpha), sigma, int(num_steps))

    results = {
        'params': {
            'sigma': sigma,
            'num_selected': num_selected,
            'num_epochs': num_epochs,
            'delta': delta,
            'tail_truncation': tail_truncation,
            'alpha': alpha
        },
        'loss_discretization_arr': loss_discretization_arr,
        'num_steps_arr': num_steps_arr,
        'runtimes': [],
        'rdp_values': [],
        'analytical_rdp': analytical_rdp
    }

    # Run all combinations and store in a 2D array
    for i, discretization in enumerate(loss_discretization_arr):
        print(f"discretization = {discretization:.0e} ({i+1}/{len(loss_discretization_arr)})")
        runtimes_row = []
        rdp_row = []

        for j, num_steps in enumerate(num_steps_arr):
            config = AllocationSchemeConfig(loss_discretization=discretization, tail_truncation=tail_truncation)
            params = PrivacyParams(
                sigma=sigma,
                num_steps=num_steps,
                num_selected=num_selected,
                num_epochs=num_epochs,
                delta=delta
            )

            start_time = time.perf_counter()
            pld = allocation_PLD(
                params=params,
                config=config,
                direction=Direction.REMOVE
            )
            end_time = time.perf_counter()

            runtime = end_time - start_time
            rdp_value = calc_PMF_RDP(pld._pmf_remove, alpha)

            runtimes_row.append(runtime)
            rdp_row.append(rdp_value)
            print(f"  num_steps = {num_steps}: {runtime:.3f}s, RDP = {rdp_value:.6f}")

        results['runtimes'].append(runtimes_row)
        results['rdp_values'].append(rdp_row)

    return results


def plot_runtime_experiment(results, save_plots=False, show_plots=True, plots_dir=None, filename='runtime_experiment.png'):
    """Visualize runtime experiment results with two subplots.

    Creates a figure with two side-by-side subplots:
    - Left: Runtime vs. 1/discretization for different num_steps
    - Right: Runtime vs. num_steps for different discretization values

    Args:
        results: Dictionary containing 'varying_discretization' and 'varying_num_steps' experiment results
        save_plots: Whether to save the plot
        show_plots: Whether to display the plot
        plots_dir: Directory to save plots (required if save_plots=True)
        filename: Filename for the saved plot

    Returns:
        The created matplotlib figure
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Runtime vs. 1/discretization (from experiment with many discretization values)
    ax1 = axs[0]
    exp1 = results['varying_discretization']
    loss_discretization_arr = exp1['loss_discretization_arr']
    num_steps_arr = exp1['num_steps_arr']
    runtimes = exp1['runtimes']  # 2D array: runtimes[i][j] for disc[i], steps[j]

    # Plot each num_steps as a separate line
    for j, num_steps in enumerate(num_steps_arr):
        # Extract runtimes for this num_steps across all discretizations
        runtimes_for_steps = [runtimes[i][j] for i in range(len(loss_discretization_arr))]
        inv_discretization = [1.0 / d for d in loss_discretization_arr]

        ax1.plot(inv_discretization, runtimes_for_steps, marker='o', linewidth=2.5,
                markersize=8, label=f't = {num_steps:,}')

    ax1.set_xlabel(r'$1/\alpha$', fontsize=14)
    ax1.set_ylabel('Runtime (seconds)', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot 2: Runtime vs. num_steps (from experiment with many num_steps values)
    ax2 = axs[1]
    exp2 = results['varying_num_steps']
    loss_discretization_arr2 = exp2['loss_discretization_arr']
    num_steps_arr2 = exp2['num_steps_arr']
    runtimes2 = exp2['runtimes']  # 2D array: runtimes[i][j] for disc[i], steps[j]

    # Plot each discretization as a separate line (reverse order for better legend)
    for i in range(len(loss_discretization_arr2) - 1, -1, -1):
        discretization = loss_discretization_arr2[i]
        # Extract runtimes for this discretization across all num_steps
        runtimes_for_disc = runtimes2[i]

        ax2.plot(num_steps_arr2, runtimes_for_disc, marker='s', linewidth=2.5,
                markersize=8, label=f'α = {discretization:.0e}')

    ax2.set_xlabel(r'$t$', fontsize=14)
    ax2.set_ylabel('Runtime (seconds)', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()

    # Save and show/close plot
    finalize_plot(fig, save_plots=save_plots, show_plots=show_plots,
                  plots_dir=plots_dir, filename=filename)

    return fig


def plot_runtime_with_rdp_experiment(results, save_plots=False, show_plots=True, plots_dir=None,
                                     runtime_filename='runtime_experiment.png',
                                     rdp_filename='rdp_experiment.png'):
    """Visualize runtime and RDP experiment results.

    Creates two separate figures:
    1. Runtime plot with two subplots (left: runtime vs 1/discretization, right: runtime vs num_steps)
    2. RDP plot with three subplots (one per num_steps value, showing RDP vs 1/discretization)

    Args:
        results: Dictionary containing 'varying_discretization' and 'varying_num_steps' experiment results
        save_plots: Whether to save the plots
        show_plots: Whether to display the plots
        plots_dir: Directory to save plots (required if save_plots=True)
        runtime_filename: Filename for the runtime plot
        rdp_filename: Filename for the RDP plot

    Returns:
        Tuple of (runtime_fig, rdp_fig)
    """
    # Extract the experiment data
    exp1 = results['varying_discretization']
    loss_discretization_arr = exp1['loss_discretization_arr']
    num_steps_arr = exp1['num_steps_arr']
    runtimes = exp1['runtimes']  # 2D array: runtimes[i][j] for disc[i], steps[j]
    rdp_values = exp1['rdp_values']  # 2D array: rdp_values[i][j] for disc[i], steps[j]
    analytical_rdp = exp1['analytical_rdp']  # Dict: {num_steps: analytical_rdp_value}

    exp2 = results['varying_num_steps']
    loss_discretization_arr2 = exp2['loss_discretization_arr']
    num_steps_arr2 = exp2['num_steps_arr']
    runtimes2 = exp2['runtimes']  # 2D array: runtimes[i][j] for disc[i], steps[j]

    # ====================================================================
    # Figure 1: Runtime plot with TWO subplots
    # ====================================================================
    runtime_fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Left subplot: Runtime vs. 1/discretization
    ax1 = axs[0]
    inv_discretization = [1.0 / d for d in loss_discretization_arr]

    for j, num_steps in enumerate(num_steps_arr):
        # Extract runtimes for this num_steps across all discretizations
        runtimes_for_steps = [runtimes[i][j] for i in range(len(loss_discretization_arr))]

        ax1.plot(inv_discretization, runtimes_for_steps, marker='o', linewidth=2.5,
                markersize=8, label=f't = {num_steps:,}')

    ax1.set_xlabel(r'$1/\alpha$', fontsize=14)
    ax1.set_ylabel('Runtime (seconds)', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Right subplot: Runtime vs. num_steps
    ax2 = axs[1]

    for i in range(len(loss_discretization_arr2) - 1, -1, -1):
        discretization = loss_discretization_arr2[i]
        # Extract runtimes for this discretization across all num_steps
        runtimes_for_disc = runtimes2[i]

        ax2.plot(num_steps_arr2, runtimes_for_disc, marker='s', linewidth=2.5,
                markersize=8, label=f'α = {discretization:.0e}')

    ax2.set_xlabel(r'$t$', fontsize=14)
    ax2.set_ylabel('Runtime (seconds)', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    runtime_fig.tight_layout()

    # ====================================================================
    # Figure 2: RDP plot with three subplots (one per num_steps)
    # ====================================================================
    rdp_fig, rdp_axs = plt.subplots(1, 3, figsize=(18, 5))

    # Compute y-axis limits across all subplots for consistency
    all_rdp_values = []
    for j, num_steps in enumerate(num_steps_arr):
        rdp_for_steps = [rdp_values[i][j] for i in range(len(loss_discretization_arr))]
        all_rdp_values.extend(rdp_for_steps)
        all_rdp_values.append(analytical_rdp[int(num_steps)])
    min_values = min(all_rdp_values)
    max_values = max(all_rdp_values)
    range_values = max_values - min_values
    y_min = min_values - 0.1*range_values
    y_max = max_values + 0.1*range_values

    # Plot each num_steps in its own subplot
    for j, num_steps in enumerate(num_steps_arr):
        ax = rdp_axs[j]

        # Extract RDP values for this num_steps across all discretizations
        rdp_for_steps = [rdp_values[i][j] for i in range(len(loss_discretization_arr))]

        # Plot numerical RDP
        ax.plot(inv_discretization, rdp_for_steps, marker='o', linewidth=2.5,
               markersize=8, label='Geometric Conv', color='b')

        # Plot analytical RDP as horizontal line
        ax.axhline(analytical_rdp[int(num_steps)], color='black', linestyle='--',
                  linewidth=2, label='Analytical')

        ax.set_xlabel(r'$1/\alpha$', fontsize=14)
        ax.set_ylabel(f'RDP({int(exp1["params"]["alpha"])})', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xscale('log')
        ax.set_ylim(y_min, y_max)

    # Add a joint legend at the bottom of the figure
    handles, labels = rdp_axs[0].get_legend_handles_labels()
    rdp_fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12,
                   bbox_to_anchor=(0.5, -0.05))

    rdp_fig.tight_layout()

    # ====================================================================
    # Save and show/close plots
    # ====================================================================
    finalize_plot(runtime_fig, save_plots=save_plots, show_plots=show_plots,
                  plots_dir=plots_dir, filename=runtime_filename)
    finalize_plot(rdp_fig, save_plots=save_plots, show_plots=show_plots,
                  plots_dir=plots_dir, filename=rdp_filename)

    return runtime_fig, rdp_fig

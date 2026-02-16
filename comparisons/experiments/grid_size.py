import time
import numpy as np
import matplotlib.pyplot as plt

from PLD_accounting.types import PrivacyParams, AllocationSchemeConfig, Direction
from PLD_accounting.random_allocation_accounting import allocation_PLD
from PLD_accounting.types import BoundType
from PLD_accounting.types import ConvolutionMethod
from dp_accounting.pld import privacy_loss_distribution
from PLD_accounting.dp_accounting_support import dp_accounting_pmf_to_discrete_dist

# ============================================================================
# Core PLD Computation Infrastructure
# ============================================================================

def compute_PLD(params: PrivacyParams,
                config: AllocationSchemeConfig,
                direction: Direction):
    """Compute PLD using the config's convolution method.

    Args:
        params: Privacy parameters (sigma, num_steps, num_selected, num_epochs)
        config: Allocation scheme configuration (loss_discretization, max_grid_FFT, max_grid_mult, tail_truncation)
        direction: Direction.REMOVE, Direction.ADD, or Direction.BOTH

    Returns:
        PrivacyLossDistribution object with _pmf_remove and _pmf_add
    """
    return allocation_PLD(params=params,
                          config=config,
                          direction=direction)

def compute_PLDs_both_methods(params: PrivacyParams,
                              mult_loss_discretization: float,
                              fft_loss_discretization: float,
                              max_grid_mult: int,
                              max_grid_FFT: int,
                              direction: Direction,
                              tail_truncation: float = 0.01):
    """Compute PLDs using both geometric and FFT methods.

    Args:
        params: Privacy parameters
        mult_loss_discretization: Discretization for geometric method
        fft_loss_discretization: Discretization for FFT method
        max_grid_mult: Max grid size for geometric method
        max_grid_FFT: Max grid size for FFT method
        direction: Direction to compute (REMOVE, ADD, or BOTH)
        tail_truncation: Tail truncation for discretization

    Returns:
        Dictionary with keys:
            - 'mult_PLD': Geometric PLD
            - 'fft_PLD': FFT PLD
            - 'mult_time': Geometric computation time
            - 'fft_time': FFT computation time
    """
    results = {}

    # Geometric convolution
    config_mult = AllocationSchemeConfig(
        loss_discretization=mult_loss_discretization,
        tail_truncation=tail_truncation,
        max_grid_FFT=1,  # dummy value
        max_grid_mult=max_grid_mult,
        convolution_method=ConvolutionMethod.GEOM
    )

    start_time = time.time()
    mult_PLD = compute_PLD(params, config_mult, direction)
    results['mult_time'] = time.time() - start_time
    results['mult_PLD'] = mult_PLD

    # FFT convolution
    config_fft = AllocationSchemeConfig(
        loss_discretization=fft_loss_discretization,
        tail_truncation=tail_truncation,
        max_grid_FFT=max_grid_FFT,
        max_grid_mult=-1,  # unused
        convolution_method=ConvolutionMethod.FFT
    )

    start_time = time.time()
    fft_PLD = compute_PLD(params, config_fft, direction)
    results['fft_time'] = time.time() - start_time
    results['fft_PLD'] = fft_PLD

    return results

def calc_epsilon_from_PLD(PLD, delta, direction: Direction):
    """Compute epsilon for a given delta and direction from a PLD object.

    Uses the get_epsilon_for_delta method on individual PMFs for specific directions.

    Args:
        PLD: PrivacyLossDistribution object with _pmf_remove and _pmf_add
        delta: Target delta value
        direction: Direction.REMOVE, Direction.ADD, or Direction.BOTH

    Returns:
        Epsilon value for the specified direction and delta
    """
    if direction == Direction.REMOVE:
        # For remove direction, compute epsilon from remove PMF only
        epsilon = PLD._pmf_remove.get_epsilon_for_delta(delta)
        return epsilon
    elif direction == Direction.ADD:
        # For add direction, compute epsilon from add PMF only
        epsilon = PLD._pmf_add.get_epsilon_for_delta(delta)
        return epsilon
    else:  # Direction.BOTH
        # For both directions, compute epsilon for both PMFs and take worst case
        epsilon_remove = PLD._pmf_remove.get_epsilon_for_delta(delta)
        epsilon_add = PLD._pmf_add.get_epsilon_for_delta(delta)
        return max(epsilon_remove, epsilon_add)

# ============================================================================
# Experiment Functions
# ============================================================================

def run_epsilon_grid_size_comparison(sigma: float,
                                      mult_grid_arr: list[int],
                                      fft_grid_arr: list[int],
                                      num_steps: int,
                                      num_selected: int,
                                      num_epochs: int,
                                      delta: float,
                                      mult_loss_discretization: float = 1e-6,
                                      fft_loss_discretization: float = 1e-4,
                                      tail_truncation: float = 1e-12) -> dict:
    """Compare epsilon-delta DP across grid sizes for a single sigma value.

    Evaluates impact of grid size on epsilon estimates for all directions.
    Uses fixed discretization and varies max_grid for both methods.

    Args:
        sigma: Noise scale value
        mult_grid_arr: Array of geometric grid sizes to test
        fft_grid_arr: Array of FFT grid sizes to test
        num_steps: Number of steps
        num_selected: Number of selected items
        num_epochs: Number of epochs
        delta: Target delta value for epsilon computation
        mult_loss_discretization: Fixed discretization value for geometric method
        fft_loss_discretization: Fixed discretization value for FFT method
        tail_truncation: Tail truncation for discretization

    Returns:
        Dictionary with sigma as key containing grid size sweeps and timings
    """
    print(f"Computing epsilon for sigma={sigma}, delta={delta}")

    params = PrivacyParams(sigma=sigma,
                           num_steps=num_steps,
                           num_selected=num_selected,
                           num_epochs=num_epochs)

    n_points = len(mult_grid_arr)

    # Initialize result arrays for each direction
    mult_epsilon_remove_arr = np.zeros(n_points)
    mult_epsilon_add_arr = np.zeros(n_points)
    mult_epsilon_both_arr = np.zeros(n_points)
    fft_epsilon_remove_arr = np.zeros(n_points)
    fft_epsilon_add_arr = np.zeros(n_points)
    fft_epsilon_both_arr = np.zeros(n_points)
    mult_times_arr = np.zeros(n_points)
    fft_times_arr = np.zeros(n_points)

    # Compute for each grid size
    for j, (mult_grid, fft_grid) in enumerate(zip(mult_grid_arr, fft_grid_arr)):
        print(f"  Point {j+1}/{n_points}: mult_grid={mult_grid}, fft_grid={fft_grid}")

        # Compute PLDs using both methods
        pld_results = compute_PLDs_both_methods(
            params=params,
            mult_loss_discretization=mult_loss_discretization,
            fft_loss_discretization=fft_loss_discretization,
            max_grid_mult=mult_grid,
            max_grid_FFT=fft_grid,
            direction=Direction.BOTH,
            tail_truncation=tail_truncation
        )

        # Extract epsilon values for all directions
        mult_epsilon_remove_arr[j] = calc_epsilon_from_PLD(pld_results['mult_PLD'], delta, Direction.REMOVE)
        mult_epsilon_add_arr[j] = calc_epsilon_from_PLD(pld_results['mult_PLD'], delta, Direction.ADD)
        mult_epsilon_both_arr[j] = calc_epsilon_from_PLD(pld_results['mult_PLD'], delta, Direction.BOTH)

        fft_epsilon_remove_arr[j] = calc_epsilon_from_PLD(pld_results['fft_PLD'], delta, Direction.REMOVE)
        fft_epsilon_add_arr[j] = calc_epsilon_from_PLD(pld_results['fft_PLD'], delta, Direction.ADD)
        fft_epsilon_both_arr[j] = calc_epsilon_from_PLD(pld_results['fft_PLD'], delta, Direction.BOTH)

        # Store timing
        mult_times_arr[j] = pld_results['mult_time']
        fft_times_arr[j] = pld_results['fft_time']

    # Package results with sigma as key
    result_dict = {
        'sigma': sigma,
        'num_steps': num_steps,
        'num_selected': num_selected,
        'num_epochs': num_epochs,
        'delta': delta,
        'mult_loss_discretization': mult_loss_discretization,
        'fft_loss_discretization': fft_loss_discretization,
        'mult_grid_arr': mult_grid_arr,
        'fft_grid_arr': fft_grid_arr,
        'mult_epsilon_remove_arr': mult_epsilon_remove_arr,
        'mult_epsilon_add_arr': mult_epsilon_add_arr,
        'mult_epsilon_both_arr': mult_epsilon_both_arr,
        'fft_epsilon_remove_arr': fft_epsilon_remove_arr,
        'fft_epsilon_add_arr': fft_epsilon_add_arr,
        'fft_epsilon_both_arr': fft_epsilon_both_arr,
        'mult_times_arr': mult_times_arr,
        'fft_times_arr': fft_times_arr,
    }

    return {sigma: result_dict}

# ============================================================================

def plot_grid_size_combined_all_directions(results, delta):
    """Plot grid size comparison for all three directions in a 3-column subplot.

    Creates a single figure with 3 columns (REMOVE, ADD, BOTH) and N rows (one per sigma).
    Uses dual x-axis: bottom for geometric grid, top for FFT grid.

    Args:
        results: Results dict with sigma as keys
        delta: Delta value for epsilon (used in title)

    Returns:
        Figure and axes array
    """
    # ========================================================================
    # Styling Configuration - Define all colors and line styles in one place
    # ========================================================================
    STYLES = {
        'geometric': {
            'color': 'blue',         # Blue - matches bottom axis
            'marker': 'o',
            'linestyle': ':',
            'linewidth': 2,
            'markersize': 6,
            'alpha': 1.0,
            'label': 'Geometric'
        },
        'fft': {
            'color': 'red',         # Orange - matches top axis
            'marker': 's',
            'linestyle': '--',
            'linewidth': 2,
            'markersize': 6,
            'alpha': 0.8,
            'label': 'FFT'
        },
        'combined': {
            'color': 'black',
            'marker': 'D',
            'linestyle': '-',
            'linewidth': 2.5,
            'markersize': 5,
            'alpha': 0.9,
            'label': 'Combined'
        }
    }

    # Get sigma values (keys that are floats)
    sigma_keys = sorted([k for k in results.keys() if isinstance(k, (int, float))])
    n_rows = len(sigma_keys)

    # Create 3-column subplot (one per direction)
    fig, axs = plt.subplots(n_rows, 3, figsize=(21, 6*n_rows))
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    directions = [
        (Direction.REMOVE, 'REMOVE', 'remove'),
        (Direction.ADD, 'ADD', 'add'),
        (Direction.BOTH, 'BOTH', 'both')
    ]

    # Compute global y-range across all rows and columns
    global_y_min = float('inf')
    global_y_max = float('-inf')
    for sigma in sigma_keys:
        res = results[sigma]
        for _, _, dir_suffix in directions:
            mult_metric = res[f'mult_epsilon_{dir_suffix}_arr']
            fft_metric = res[f'fft_epsilon_{dir_suffix}_arr']
            global_y_min = min(global_y_min, np.min(mult_metric), np.min(fft_metric))
            global_y_max = max(global_y_max, np.max(mult_metric), np.max(fft_metric))

    # Add 5% padding to global y-range
    y_padding = (global_y_max - global_y_min) * 0.05
    global_y_min -= y_padding
    global_y_max += y_padding

    for row, sigma in enumerate(sigma_keys):
        res = results[sigma]
        mult_grid = res['mult_grid_arr']
        fft_grid = res['fft_grid_arr']

        for col, (direction, dir_name, dir_suffix) in enumerate(directions):
            ax1 = axs[row, col]
            ax2 = ax1.twiny()  # Top axis for FFT

            # Get metric arrays
            mult_metric = res[f'mult_epsilon_{dir_suffix}_arr']
            fft_metric = res[f'fft_epsilon_{dir_suffix}_arr']

            # Plot geometric (bottom axis) using STYLES config
            mult_style = STYLES['geometric']
            ax1.plot(mult_grid, mult_metric,
                    f"{mult_style['marker']}{mult_style['linestyle']}",
                    label=mult_style['label'],
                    linewidth=mult_style['linewidth'],
                    markersize=mult_style['markersize'],
                    color=mult_style['color'],
                    alpha=mult_style['alpha'])

            # Plot FFT (top axis) using STYLES config
            fft_style = STYLES['fft']
            ax2.plot(fft_grid, fft_metric,
                    f"{fft_style['marker']}{fft_style['linestyle']}",
                    label=fft_style['label'],
                    linewidth=fft_style['linewidth'],
                    markersize=fft_style['markersize'],
                    color=fft_style['color'],
                    alpha=fft_style['alpha'])

            # For BOTH direction, add combined line: max(min(FFT_add, mult_add), min(FFT_rem, mult_rem))
            if direction == Direction.BOTH:
                mult_add = res['mult_epsilon_add_arr']
                mult_remove = res['mult_epsilon_remove_arr']
                fft_add = res['fft_epsilon_add_arr']
                fft_remove = res['fft_epsilon_remove_arr']

                # Compute combined: max(min(FFT_add, mult_add), min(FFT_rem, mult_rem))
                combined = np.maximum(
                    np.minimum(fft_add, mult_add),
                    np.minimum(fft_remove, mult_remove)
                )

                # Plot combined on bottom axis using STYLES config
                combined_style = STYLES['combined']
                ax1.plot(mult_grid, combined,
                        f"{combined_style['marker']}{combined_style['linestyle']}",
                        label=combined_style['label'],
                        linewidth=combined_style['linewidth'],
                        markersize=combined_style['markersize'],
                        color=combined_style['color'],
                        alpha=combined_style['alpha'])

            # Configure bottom axis (geometric) - use color from STYLES
            ax1.set_xscale('log')
            ax1.set_xlabel('Geometric Grid Size', fontsize=16,
                          color=STYLES['geometric']['color'])
            ax1.tick_params(axis='x', labelcolor=STYLES['geometric']['color'],
                           labelsize=14)
            ax1.tick_params(axis='y', labelsize=14)
            ax1.grid(True, alpha=0.3)

            # Set y-limits to be the same across all subplots
            ax1.set_ylim(global_y_min, global_y_max)

            # Configure top axis (FFT) - use color from STYLES
            ax2.set_xscale('log')
            ax2.tick_params(axis='x', labelcolor=STYLES['fft']['color'], labelsize=14)

            # Y-axis label on leftmost column
            if col == 0:
                ax1.set_ylabel(r'$\varepsilon$', fontsize=18)

            # FFT x-axis label on top for first row
            if row == 0:
                ax2.set_xlabel('FFT Grid Size', fontsize=16,
                              color=STYLES['fft']['color'])
                # Add column title
                ax1.set_title(dir_name, fontsize=18, fontweight='bold', pad=40)

            # Store the twin axis reference for legend extraction (need one from each column type)
            if row == 0:
                if col == 0:
                    ax2_for_legend = ax2
                elif col == 2:  # BOTH column - has the combined line
                    ax1_both_for_legend = ax1

    # Add a single legend below all subplots
    # Get legend handles from first subplot (geometric), its twin (FFT), and BOTH column (combined)
    lines1, labels1 = axs[0, 0].get_legend_handles_labels()
    lines2, labels2 = ax2_for_legend.get_legend_handles_labels()
    lines3, labels3 = ax1_both_for_legend.get_legend_handles_labels()

    # Combine all handles and labels, removing duplicates while preserving order
    all_lines = lines1 + lines2 + lines3
    all_labels = labels1 + labels2 + labels3

    # Create dict to deduplicate while maintaining order
    unique_legend = {}
    for line, label in zip(all_lines, all_labels):
        if label not in unique_legend:
            unique_legend[label] = line

    fig.legend(unique_legend.values(), unique_legend.keys(), loc='lower center',
               ncol=len(unique_legend), fontsize=14,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Bottom space for legend only

    return fig, axs

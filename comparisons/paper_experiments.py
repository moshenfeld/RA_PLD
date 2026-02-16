import numpy as np
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from random_allocation.comparisons.structs import SchemeConfig, Verbosity

from PLD_accounting.types import AllocationSchemeConfig, Direction, PrivacyParams, BoundType, ConvolutionMethod
from comparisons.data_management import save_results, merge_with_historical_results
from comparisons.experiments.epsilon_from_sigma import run_epsilon_from_sigma, run_epsilon_from_sigma_by_k, run_epsilon_vs_k
from comparisons.experiments.PREAMBLE import run_PREAMBLE_experiment, run_preamble_epsilon_experiment
from comparisons.experiments.runtime import run_runtime_experiment
from comparisons.experiments.delta_comparison import run_delta_comparison
from comparisons.experiments.utility_comparison import run_utility_comparison_suite
from experiments.grid_size import run_epsilon_grid_size_comparison
from comparisons.experiments.subsampling_comparison import run_subsampling_bounds_comparison_experiment


# ====================================================================
# Experiment functions
# ====================================================================

def run_epsilon_from_sigma_experiment():
    """Run epsilon from sigma experiment with multiple num_steps values."""
    # Parameters
    sigma_arr = np.linspace(1.0, 4.0, 10)
    num_steps_arr = [10, 100, 1000]
    num_epochs = 1
    num_selected = 1
    delta = 1e-6
    beta = delta * 0.01

    # Configurations
    allocation_RDP_config = SchemeConfig(allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)])
    allocation_conv_config = AllocationSchemeConfig(tail_truncation=beta)
    Poisson_config = SchemeConfig(discretization=1e-4)

    # Run experiment
    results = run_epsilon_from_sigma(num_steps_arr, sigma_arr, num_selected, num_epochs, delta, 
                                     allocation_RDP_config, allocation_conv_config, Poisson_config)
    save_results(results, 'epsilon_from_sigma')


def run_epsilon_from_sigma_k_experiment():
    """Run epsilon from sigma experiment for multiple k values at t=1000."""
    sigma_arr = np.linspace(1.0, 4.0, 10)
    num_steps = 1000
    num_epochs = 1
    num_selected_arr = [10, 100]
    delta = 1e-6
    beta = delta * 0.01

    allocation_RDP_config = SchemeConfig(allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)])
    allocation_conv_config = AllocationSchemeConfig(loss_discretization=0.01, tail_truncation=beta, max_grid_mult=50_000)
    Poisson_config = SchemeConfig(discretization=1e-4)

    results = run_epsilon_from_sigma_by_k(
        num_selected_arr=num_selected_arr,
        sigma_arr=sigma_arr,
        num_steps=num_steps,
        num_epochs=num_epochs,
        delta=delta,
        allocation_RDP_config=allocation_RDP_config,
        allocation_conv_config=allocation_conv_config,
        Poisson_config=Poisson_config
)
    save_results(results, 'epsilon_from_sigma_by_k')


def run_epsilon_vs_k_experiment():
    """Run epsilon vs k experiment for sigma=1 and t=1000."""
    sigma = 5.0
    num_steps = 1000
    num_epochs = 1
    num_selected_arr = np.round(np.linspace(1, 250, 10)).astype(int).tolist()
    delta = 1e-6
    beta = delta/num_epochs*0.01

    allocation_RDP_config = SchemeConfig(allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)])
    allocation_conv_config = AllocationSchemeConfig(tail_truncation=beta, max_grid_mult=50_000)
    Poisson_config = SchemeConfig(discretization=1e-4)

    results = run_epsilon_vs_k(
        num_selected_arr=num_selected_arr,
        sigma=sigma,
        num_steps=num_steps,
        num_epochs=num_epochs,
        delta=delta,
        allocation_RDP_config=allocation_RDP_config,
        allocation_conv_config=allocation_conv_config,
        Poisson_config=Poisson_config
)
    save_results(results, 'epsilon_vs_k')



def run_epsilon_discretization_experiment():
    """Run epsilon discretization experiment for sigma=1.

    Tests epsilon-delta DP across varying grid sizes for geometric vs FFT methods.
    Single sigma value, no RDP computation.
    """
    # Parameters
    sigma = 1.0
    num_steps = 1000
    num_selected = 1
    num_epochs = 1
    delta = 1e-6
    beta = 1e-12
    mult_discretization = 1e-6
    fft_discretization = 1e-4

    # Grid sizes
    mult_grid_arr = np.geomspace(1e3, 1e5, 10, dtype=int).tolist()
    fft_grid_arr = np.geomspace(1e7, 1e8, 10, dtype=int).tolist()

    # Run experiment
    results = run_epsilon_grid_size_comparison(
        sigma=sigma,
        mult_grid_arr=mult_grid_arr,
        fft_grid_arr=fft_grid_arr,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        delta=delta,
        mult_loss_discretization=mult_discretization,
        fft_loss_discretization=fft_discretization,
        tail_truncation=beta
    )

    save_results(results, 'epsilon_discretization')


def run_preamble_experiment():
    """Run PREAMBLE experiment."""

    # Parameters
    settings_dict = {
        'sample size': int(6*1e5),
        'D': 2**20,
        'communication constant': 2**15,
        'SGD_num_epochs': 10,
        'batch size array': [512, 1028, 4096, int(6*1e5)],
        #'batch size array': [int(6*1e5)],
        'B array': (2**np.arange(11,12)).astype(int),
        'delta': 1e-6,
        'target epsilon': 1.0,
        'clip scale': 1.02,
    }

    # Configuration for the numerical method
    numerical_config = AllocationSchemeConfig(
        loss_discretization=settings_dict['target epsilon'] * 1e-3,
        tail_truncation=settings_dict['delta'] * 0.01,
        max_grid_mult=100_000,
        max_grid_FFT=100_000_000,
        convolution_method=ConvolutionMethod.GEOM  # GEOM, FFT, or COMBINED
    )
    Gaussian_config = SchemeConfig(discretization=1e-4)

    bound_type = BoundType.DOMINATES  # DOMINATES for upper bound, IS_DOMINATED for lower bound

    # Run experiment with optional caching (RDP is always run first)
    results = run_PREAMBLE_experiment(
        settings_dict=settings_dict,
        numerical_config=numerical_config,
        Gaussian_config=Gaussian_config,
        bound_type=bound_type,
        num_processes = 6,
    )
    results = merge_with_historical_results('PREAMBLE_experiment', results)
    
    save_results(results, 'PREAMBLE_experiment')


def run_paper_preamble_epsilon_experiment():
    """Run PREAMBLE epsilon experiment with paper-specific parameters."""
    settings_dict = {
        'sample size': int(6*1e5),
        'D': 2**20,
        'communication constant': 2**15,
        'SGD_num_epochs': 10,
        'batch size array': [512],#, 4096, int(6*1e5)],
        # 'sigma array': [0.75, 0.9, 1.2, 12.5],
        # 'B array': (2**np.arange(2, 12)).astype(int),
        #'batch size array': [512],
        'sigma array': [1.0],
        'B array': (2**np.arange(2, 12)).astype(int),
        'delta': 1e-6,
        'clip scale': 1.02,
    }

    numerical_config = AllocationSchemeConfig(
        loss_discretization=1e-3,
        tail_truncation=settings_dict['delta'] * 0.01,
        max_grid_mult=200_000,
        max_grid_FFT=100_000_000,
        convolution_method=ConvolutionMethod.COMBINED #GEOM  # GEOM, FFT, or COMBINED
    )
    Gaussian_config = SchemeConfig(discretization=1e-4)

    results = run_preamble_epsilon_experiment(
        settings_dict=settings_dict,
        numerical_config=numerical_config,
        Gaussian_config=Gaussian_config,
    )
    save_results(results, 'preamble_epsilon_experiment')


def run_runtime_comparison_experiment():
    """Run runtime and RDP experiment.

    This experiment uses the same parameters for both runtime and RDP measurements,
    making it easy to compare the runtime cost of different discretization levels
    with the accuracy of RDP estimates.
    """
    # Parameters for varying discretization experiment (left plot + RDP plot)
    runtime_discretization_arr_varying_disc = np.geomspace(0.1, 0.01, 10)  # 10 discretization values
    runtime_num_steps_arr_varying_disc = np.geomspace(100, 10000, 3, dtype=int)  # 3 num_steps values

    # Parameters for varying num_steps experiment (right plot)
    runtime_discretization_arr_varying_steps = np.geomspace(0.1, 0.01, 3)  # 3 fixed discretization values
    runtime_num_steps_arr_varying_steps = 2**np.arange(6,14)  # 8 num_steps values

    runtime_sigma = 1.0
    runtime_num_selected = 1
    runtime_num_epochs = 1
    runtime_delta = 1e-6
    runtime_beta = runtime_delta * 0.01
    runtime_alpha = 2.0  # RDP order

    # Run experiment with many discretization values (for left plot and RDP plot)
    print("Running Runtime & RDP Experiment: varying discretization...")
    results_varying_disc = run_runtime_experiment(
        loss_discretization_arr=runtime_discretization_arr_varying_disc,
        num_steps_arr=runtime_num_steps_arr_varying_disc,
        sigma=runtime_sigma,
        num_selected=runtime_num_selected,
        num_epochs=runtime_num_epochs,
        delta=runtime_delta,
        tail_truncation=runtime_beta,
        alpha=runtime_alpha
    )

    # Run experiment with many num_steps values (for right plot)
    print("\nRunning Runtime Experiment: varying num_steps...")
    results_varying_steps = run_runtime_experiment(
        loss_discretization_arr=runtime_discretization_arr_varying_steps,
        num_steps_arr=runtime_num_steps_arr_varying_steps,
        sigma=runtime_sigma,
        num_selected=runtime_num_selected,
        num_epochs=runtime_num_epochs,
        delta=runtime_delta,
        tail_truncation=runtime_beta,
        alpha=runtime_alpha
    )

    # Package results
    results = {
        'params': results_varying_disc['params'],
        'varying_discretization': results_varying_disc,
        'varying_num_steps': results_varying_steps
    }

    save_results(results, 'runtime_experiment')


def run_delta_comparison_experiment():
    """Run delta comparison experiment."""
    # Parameters
    delta_num_steps_arr = [35_938, 4_492, 12_500, 1_563]
    delta_sigma_arr = [0.3, 0.4, 0.3, 0.4]
    delta_epsilon_arr_list = [
        np.linspace(0.1, 8, 20),
        np.linspace(0.1, 8, 20),
        np.linspace(0.1, 8, 20),
        np.linspace(0.1, 8, 20)
    ]
    delta_num_selected = 1
    delta_num_epochs = 1
    delta_beta = 1e-10 / delta_num_epochs * 0.01

    # Configurations
    delta_allocation_RDP_config = SchemeConfig(
        discretization=1e-4,
        allocation_direct_alpha_orders=[int(i) for i in np.arange(2, 61, dtype=int)],
        delta_tolerance=1e-15,
        epsilon_tolerance=1e-3,
        MC_use_order_stats=True,
        MC_use_mean=False,
        MC_conf_level=0.99,
        MC_sample_size=500_000,
        verbosity=Verbosity.NONE
)
    delta_allocation_conv_config = AllocationSchemeConfig(loss_discretization=1e-1, tail_truncation=delta_beta)
    delta_Poisson_config = SchemeConfig(discretization=1e-4)

    # Run experiment
    results = run_delta_comparison(
        num_steps_arr=delta_num_steps_arr,
        sigma_arr=delta_sigma_arr,
        epsilon_arr_list=delta_epsilon_arr_list,
        num_selected=delta_num_selected,
        num_epochs=delta_num_epochs,
        allocation_RDP_config=delta_allocation_RDP_config,
        allocation_conv_config=delta_allocation_conv_config,
        Poisson_config=delta_Poisson_config
    )
    save_results(results, 'delta_comparison')


def run_utility_comparison_experiment():
    """Run utility comparison between Poisson and allocation using legacy and numerical bounds."""
    small_eps = 0.1
    large_eps = 1.0
    small_dim = 1
    large_dim = 1_000

    num_steps = 1_000
    num_experiments = 10_000
    true_mean = 0.9
    delta = 1e-10
    sample_size_arr = np.logspace(2, 5, num=7, dtype=int)
    num_std = 3

    epsilon_values = [large_eps, small_eps, small_eps]
    dimension_values = [small_dim, small_dim, large_dim]
    titles = [
        r"$\varepsilon$ = " + f"{large_eps:.1f}" + r", $d$ = " + f"{small_dim:,}",
        r"$\varepsilon$ = " + f"{small_eps:.1f}" + r", $d$ = " + f"{small_dim:,}",
        r"$\varepsilon$ = " + f"{small_eps:.1f}" + r", $d$ = " + f"{large_dim:,}",
    ]

    allocation_conv_config = AllocationSchemeConfig(
        loss_discretization=0.002,
        tail_truncation=delta * 0.01,
        max_grid_mult=50_000
    )

    results = run_utility_comparison_suite(
        epsilon_values=epsilon_values,
        dimension_values=dimension_values,
        sample_size_arr=sample_size_arr,
        num_steps=num_steps,
        delta=delta,
        true_mean=true_mean,
        num_experiments=num_experiments,
        num_std=num_std,
        include_numerical_allocation=True,
        allocation_conv_config=allocation_conv_config,
        titles=titles
    )

    save_results(results, 'utility_comparison')


def run_subsampling_comparison():
    """Run subsampling methods comparison experiment."""
    # Parameters
    sigma = 1.0
    sampling_prob = 0.1
    discretization = 0.002  # q * 0.02
    sensitivity = 1.0

    # Run experiment
    results = run_subsampling_bounds_comparison_experiment(
        sigma=sigma,
        sampling_prob=sampling_prob,
        discretization=discretization,
        sensitivity=sensitivity,
        direction="both",
    )
    save_results(results, 'subsampling_comparison')


# ====================================================================
# Experiment control flags
# ====================================================================
RUN_EPSILON_FROM_SIGMA          = False
RUN_EPSILON_FROM_SIGMA_K        = False
RUN_EPSILON_VS_K                = False
RUN_PREAMBLE_EXPERIMENT         = False
RUN_PREAMBLE_EPSILON_EXPERIMENT = False
RUN_RUNTIME_EXPERIMENT          = False
RUN_DELTA_COMPARISON            = True
RUN_EPSILON_DISCRETIZATION      = True
RUN_UTILITY_COMPARISON          = True
RUN_SUBSAMPLING_COMPARISON      = True


# ====================================================================
# Main execution
# ====================================================================

if __name__ == '__main__':
    if RUN_EPSILON_FROM_SIGMA:
        run_epsilon_from_sigma_experiment()

    if RUN_EPSILON_FROM_SIGMA_K:
        run_epsilon_from_sigma_k_experiment()

    if RUN_EPSILON_VS_K:
        run_epsilon_vs_k_experiment()

    if RUN_PREAMBLE_EXPERIMENT:
        run_preamble_experiment()
        
    if RUN_PREAMBLE_EPSILON_EXPERIMENT:
        run_paper_preamble_epsilon_experiment()

    if RUN_RUNTIME_EXPERIMENT:
        run_runtime_comparison_experiment()

    if RUN_DELTA_COMPARISON:
        run_delta_comparison_experiment()

    if RUN_EPSILON_DISCRETIZATION:
        run_epsilon_discretization_experiment()

    if RUN_UTILITY_COMPARISON:
        run_utility_comparison_experiment()

    if RUN_SUBSAMPLING_COMPARISON:
        run_subsampling_comparison()

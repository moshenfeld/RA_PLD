import time
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from multiprocessing import Pool, cpu_count
from enum import Enum

from dp_accounting import pld
from comparisons.data_management import load_results_with_matching_params

if TYPE_CHECKING:
    from random_allocation.comparisons.structs import SchemeConfig
else:
    SchemeConfig = object

from PLD_accounting.types import ConvolutionMethod, BoundType, PrivacyParams, AllocationSchemeConfig, Direction
from PLD_accounting.random_allocation_accounting import allocation_PLD
from PLD_accounting.subsample_PLD import subsample_PLD
from comparisons.experiments.random_allocation_RDP import allocation_rdp_arr, ampl_subsampling_rdp, get_privacy_spent

def epsilon_from_sigma_gaussian(discretization: float,
                                sigma: float,
                                sampling_probability: float,
                                num_rounds: int,
                                delta: float) -> float:
    """Compute epsilon for standard Gaussian mechanism with subsampling.

    Evaluates composed privacy guarantee for Gaussian noise with Poisson subsampling.
    Implementation uses dp-accounting with connect-dots optimization.
    """
    amplified_PLD = pld.privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=sigma,
        value_discretization_interval=discretization,
        sampling_prob=sampling_probability,
        pessimistic_estimate=True,
        use_connect_dots=True
    )
    composed_pld = amplified_PLD.self_compose(num_rounds)
    return composed_pld.get_epsilon_for_delta(delta)

def epsilon_from_sigma_allocation_RDP(params: PrivacyParams,
                                      sampling_probability: float,
                                      num_rounds: int) -> float:
    """Compute epsilon for allocation scheme via RDP accounting.

    Evaluates privacy using direct RDP formulas, subsampling amplification, composition.
    Implementation computes RDP across orders 2-50, applies subsampling, converts to (ε,δ)-DP.
    """
    orders = range(2, 51)

    eps_rdp_per_batch_rem = allocation_rdp_arr(
        sigma=params.sigma,
        num_steps=params.num_steps,
        num_selected=params.num_selected,
        orders=orders,
        is_remove=True
    )
    eps_rdp_subsampling_rem = ampl_subsampling_rdp(
        eps=eps_rdp_per_batch_rem,
        alpha_lst=orders,
        q=sampling_probability
    )
    eps_rdp_total_rem = num_rounds * eps_rdp_subsampling_rem
    epsilon_composed_rem, _ = get_privacy_spent(
        orders=list(orders),
        rdp=list(eps_rdp_total_rem),
        delta=params.delta
    )

    eps_rdp_per_batch_add = allocation_rdp_arr(
        sigma=params.sigma,
        num_steps=params.num_steps,
        num_selected=params.num_selected,
        orders=orders,
        is_remove=False
    )
    eps_rdp_subsampling_add = ampl_subsampling_rdp(
        eps=eps_rdp_per_batch_add,
        alpha_lst=orders,
        q=sampling_probability
    )
    eps_rdp_total_add = num_rounds * eps_rdp_subsampling_add
    epsilon_composed_add, _ = get_privacy_spent(
        orders=list(orders),
        rdp=list(eps_rdp_total_add),
        delta=params.delta
    )

    return max(epsilon_composed_rem, epsilon_composed_add)

def epsilon_from_sigma_allocation(params: PrivacyParams,
                                  config: AllocationSchemeConfig,
                                  sampling_probability: float,
                                  num_rounds: int,
                                  bound_type: BoundType,
                                  ) -> float:
    """Compute epsilon for allocation scheme via dual subsampling on upper PMFs."""
    if bound_type != BoundType.DOMINATES:
        raise ValueError("Dual subsampling route supports DOMINATES bounds only")

    # Create a copy of config with scaled parameters to avoid modifying the original.
    # The tail bound scales with both composition and subsampling.
    # The discretization (~std) scales with the sqrt of the final composition
    # scaled down by the sampling rate.
    config_copy = AllocationSchemeConfig(
        loss_discretization=config.loss_discretization / np.sqrt(num_rounds * sampling_probability),
        tail_truncation=config.tail_truncation / (num_rounds * sampling_probability),
        max_grid_mult=config.max_grid_mult,
        max_grid_FFT=config.max_grid_FFT,
        convolution_method=config.convolution_method,
    )

    base_PLD = allocation_PLD(
        params=params,
        config=config_copy,
        direction=Direction.BOTH,
    )

    subsampled_PLD = subsample_PLD(
        pld=base_PLD,
        sampling_probability=sampling_probability,
        bound_type=BoundType.DOMINATES,
    )

    composed_pld = subsampled_PLD.self_compose(num_rounds)

    epsilon = composed_pld.get_epsilon_for_delta(params.delta)
    print(
        f"\t\t num_steps={params.num_steps}, num_allocations={params.num_selected}, "
        f"num_rounds={num_rounds}, sigma={params.sigma}, epsilon={epsilon}",
        flush=True
    )
    return epsilon

# lin_func:  function making lin_func(func(x)) close to linear in x 
# rel_range_delta: parameter tolerance relative to the entire range
def lin_bin_search_function(func, y_target, bounds, target_tolerance, domination: BoundType, rel_range_delta = 1/64., lin_func=lambda x: x):
    """Binary search to find x where func(x) satisfies domination constraint.

    Assumes func is monotonic decreasing.
    
    Args:
        func: Monotonic decreasing function
        y_target: Target value
        bounds: (left, right) bounds for search
        target_tolerance: Tolerance for convergence
        domination: BoundType.DOMINATES ensures func(x) <= y_target,
                    BoundType.IS_DOMINATED ensures func(x) >= y_target
    
    Returns:
        x such that:
        - If DOMINATES: func(x) <= y_target
        - If IS_DOMINATED: func(x) >= y_target
    """
    #Set resolution based on relative range delta
    left, right = bounds
    range_delta = (right - left) * rel_range_delta

    # Make the bound fully conservative
    if domination == BoundType.DOMINATES:
        y_target -= target_tolerance
    else:
        y_target += target_tolerance

    f_right = func(right)   
    if f_right > y_target:
        return right
      
    f_left = np.inf
    best_point = right
    best_value = f_right
    best_delta = abs(lin_func(best_value) - lin_func(y_target))

    # Use one point linear approximation: 
    mid = max(left, right*f_right/y_target)

    switch_to_binary = False

    iterations = 1    
    while right - left > range_delta:
    
        f_mid = func(mid)
        iterations += 1

        if (abs(f_mid - y_target) <= target_tolerance):
            print(f"Converged at iteration {iterations}: f_mid={f_mid:.4f},  mid={mid:.4f}. Value within target tolerance.")
            return mid
        #if domination == BoundType.DOMINATES and f_mid <= y_target and f_mid + target_tolerance > y_target:
        #    print(f"Converged at iteration {iterations}: f_mid={f_mid:.4f}, target={y_target}, mid={mid:.4f}")
        #    return mid
        #if domination == BoundType.IS_DOMINATED and f_mid >= y_target and f_mid + target_tolerance < y_target:
        #    print(f"Converged at iteration {iterations}: f_mid={f_mid:.4f}, target={y_target}, mid={mid:.4f}")
        #    return mid


        # Detect non-monotonicity
        if (f_mid < f_right) or (f_mid > f_left):
            print(f"Non-monotonic behavior detected: f_mid={f_mid:.4f}, mid={mid:.4f}, f_left={f_left:.4f}, left={left:.4f}, f_right={f_right:.4f}, right={right:.4f}")
            break

        if f_mid < y_target:
            range_frac = (mid - left) / (right - left)
            right = mid
            f_right = f_mid
        else:
            range_frac = (right - mid) / (right - left)
            left = mid
            f_left = f_mid

        
        if (switch_to_binary or np.isclose(f_mid, best_value, atol=1e-4)):
            mid = (left + right) / 2.0
            continue

        value_frac = abs(1./f_mid - 1./y_target) / best_delta
        # If not improving fast enough, switch to binary search to ensure convergence
        if value_frac > 0.5 and range_frac > 0.5:
            print(f"Switching to binary search: value_frac={value_frac:.4f}, range_frac={range_frac:.4f}")
            switch_to_binary = True
            mid = (left + right) / 2.0
            continue

        old_mid = mid
        mid = old_mid + (lin_func(y_target) - lin_func(f_mid)) * (best_point-old_mid) / (lin_func(best_value) - lin_func(f_mid)) 
        mid = max(left, min(right, mid))  # Ensure mid stays within bounds
        if (abs(lin_func(f_mid) - lin_func(y_target)) < best_delta):
            best_point = old_mid
            best_value = f_mid
            best_delta = abs(lin_func(best_value) - lin_func(y_target))
    print(f"Converged at iteration {iterations}. Range based.")
    return right if domination == BoundType.DOMINATES else left


def bin_search_function(func, y_target, bounds, target_tolerance, domination: BoundType):
    """Binary search to find x where func(x) satisfies domination constraint.

    Assumes func is monotonic decreasing.
    
    Args:
        func: Monotonic decreasing function
        y_target: Target value
        bounds: (left, right) bounds for search
        target_tolerance: Tolerance for convergence
        domination: BoundType.DOMINATES ensures func(x) <= y_target,
                    BoundType.IS_DOMINATED ensures func(x) >= y_target
    
    Returns:
        x such that:
        - If DOMINATES: func(x) <= y_target
        - If IS_DOMINATED: func(x) >= y_target
    """
    left, right = bounds
    f_right = func(right)   
    if f_right > y_target:
        return right
        
    while right - left > 1e-2:
        mid = (left + right) / 2
        f_mid = func(mid)
        if domination == BoundType.DOMINATES and f_mid <= y_target and f_mid + target_tolerance > y_target:
            return mid
        if domination == BoundType.IS_DOMINATED and f_mid >= y_target and f_mid + target_tolerance < y_target:
            return mid
        if f_mid < y_target:
            right = mid
        else:
            left = mid
    return right if domination == BoundType.DOMINATES else left

def _worker_wrapper(args):
    """Wrapper function to unpack task tuple for multiprocessing."""
    task_dict, config = args
    return sigma_from_epsilon_allocation(
        task_dict=task_dict,
        config=config
    )

def sigma_from_epsilon_allocation(task_dict: dict,
                                  config: AllocationSchemeConfig
) -> dict:
    """Find sigma that achieves target epsilon for allocation scheme.

    Args:
        task_dict: Dictionary containing all parameters including batch_size, B, bound_type
        config: Configuration including the convolution method to use

    Returns:
        Dictionary with batch_size, B, bound_type, convolution_method, sigma, and time
    """
    # Print start message for this task
    start_time = time.time()

    batch_size = task_dict['batch_size']
    B = task_dict['B']
    is_rdp = task_dict['is_rdp']
    bound_type = task_dict.get('bound_type')
    num_steps = task_dict['num steps']
    block_norm = task_dict['block norm']
    num_selected = task_dict['num selected']
    sampling_probability = task_dict['sampling probability']
    num_rounds = task_dict['num rounds']
    delta = task_dict['delta']
    target_epsilon = task_dict['target epsilon']
    # Determine bounds based on whether this is RDP or numerical
    if is_rdp:
        # For RDP, use wider bounds based on Gaussian sigma
        min_sigma = 0.9*task_dict['Gaussian sigma']
        max_sigma = 10*task_dict['Gaussian sigma']
        method_str = "RDP"
    else:
        # For numerical methods, use tighter bounds from Gaussian and RDP
        min_sigma = task_dict['Gaussian sigma']
        max_sigma = task_dict.get('RDP sigma', 2.5*task_dict['Gaussian sigma']) 
        method_str = f"{config.convolution_method} ({bound_type})"

    print(f"[STARTED at {time.strftime('%H:%M:%S', time.localtime(start_time))}] batch_size={batch_size}, B={B}, method={method_str}", flush=True)

    if is_rdp:
        optimization_func = lambda sigma: epsilon_from_sigma_allocation_RDP(
            params=PrivacyParams(
                sigma=sigma/block_norm,
                num_steps=num_steps,
                num_selected=num_selected,
                delta=delta
            ),
            sampling_probability=sampling_probability,
            num_rounds=num_rounds
        )
        # RDP experiment: use DOMINATES by default (conservative)
        domination = BoundType.DOMINATES
    else:
        optimization_func = lambda sigma: epsilon_from_sigma_allocation(
            params=PrivacyParams(
                sigma=sigma/block_norm,
                num_steps=num_steps,
                num_selected=num_selected,
                delta=delta
            ),
            config=config,
            sampling_probability=sampling_probability,
            num_rounds=num_rounds,
            bound_type=bound_type
        )

        # Determine domination type based on bound_type
        domination = bound_type

    epsilon_tolerance = task_dict.get('epsilon tolerance')
    if is_rdp:
        sigma_value = lin_bin_search_function(
            func=optimization_func,
            y_target=target_epsilon,
            bounds=(min_sigma, max_sigma),
            target_tolerance=epsilon_tolerance,
            domination=domination,
            rel_range_delta=1./2**10,
            lin_func=lambda x: 1./x
        )
    else:
        sigma_value = lin_bin_search_function(
            func=optimization_func,
            y_target=target_epsilon,
            bounds=(min_sigma, max_sigma),
            target_tolerance=epsilon_tolerance,
            domination=domination,
            rel_range_delta=1./2**6,
            lin_func=lambda x: 1./x
        )

    end_time = time.time()
    computation_time = end_time - start_time
    print(f"[COMPLETED at {time.strftime('%H:%M:%S', time.localtime(end_time))}] batch_size={batch_size}, B={B}, method={method_str}, sigma={sigma_value}, time={computation_time:.1f}s", flush=True)
    return {'batch_size': batch_size,
            'B': B,
            'is_rdp': is_rdp,
            'bound_type': bound_type,
            'convolution_method': config.convolution_method if not is_rdp else None,
            'method': 'pld',
            'sigma': sigma_value,
            'time': computation_time}

def run_PREAMBLE_experiment(settings_dict: dict,
                            numerical_config: AllocationSchemeConfig,
                            Gaussian_config: SchemeConfig,
                            bound_type: BoundType,
                            num_processes: int = None,
                            load_cached_results: bool = False,
                            experiment_name: str = "PREAMBLE_experiment"):
    """Run PREAMBLE experiment comparing Gaussian vs allocation schemes.

    First runs Gaussian and RDP experiments in parallel, then uses those bounds
    for the numerical PLD method.

    Args:
        settings_dict: Dictionary containing experiment parameters
        numerical_config: Configuration for the chosen numerical convolution method
        Gaussian_config: Configuration for Gaussian baseline
        bound_type: BoundType.DOMINATES for upper bound, BoundType.IS_DOMINATED for lower bound
        num_processes: Number of parallel processes to use. If None, uses cpu_count() - 1.
        load_cached_results: Whether to reuse already-saved Gaussian/RDP data when params match.
        experiment_name: Name of the saved experiment whose results can be reused when parameters match
    """

    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)

    print(f"Running PREAMBLE experiment with {num_processes} parallel processes", flush=True)

    results = {'params': {'sample size'            : settings_dict['sample size'],
                          'D'                      : settings_dict['D'],
                          'communication_constant' : settings_dict['communication constant'],
                          'SGD_num_epochs'         : settings_dict['SGD_num_epochs'],
                          'batch size array'       : settings_dict['batch size array'],
                          'B array'                : settings_dict['B array'],
                          'delta'                  : settings_dict['delta'],
                          'target epsilon'         : settings_dict['target epsilon'],
                          'clip scale'             : settings_dict['clip scale'],
                          'numerical config'       : {'loss_discretization': numerical_config.loss_discretization,
                                                      'tail_truncation'   : numerical_config.tail_truncation,
                                                      'max_grid_mult'     : numerical_config.max_grid_mult,
                                                      'max_grid_FFT'      : numerical_config.max_grid_FFT},
                          'Gaussian config'        : {'loss_discretization': Gaussian_config.discretization},
                          'bound_type'             : bound_type,
                          'convolution_method'     : numerical_config.convolution_method,
                          }
              }

    target_epsilon = settings_dict['target epsilon']
    B_array = settings_dict['B array']
    relative_epsilon_accuracy = 0.01
    epsilon_search_resolution = target_epsilon * relative_epsilon_accuracy

    # Step 1: Load or compute Gaussian and RDP results
    gaussian_rdp_results = {}
    batch_size_array = settings_dict['batch size array']

    cached_results, cached_path = None, None
    if load_cached_results:
        cached_results, cached_path = load_results_with_matching_params(experiment_name, results['params'])
        if cached_results is not None:
            missing_data = False
            for batch_size in batch_size_array:
                batch_data = cached_results.get(batch_size)
                if (batch_data is None or
                        'Gaussian_sigma' not in batch_data or
                        'RDP' not in batch_data or
                        not all(B in batch_data['RDP'] for B in B_array)):
                    missing_data = True
                    break

            if not missing_data:
                cache_label = f" ({cached_path.name})" if cached_path else ""
                print(f"Reusing Gaussian/RDP results from saved experiment{cache_label}")
                for batch_size in batch_size_array:
                    batch_data = cached_results[batch_size]
                    gaussian_rdp_results[batch_size] = {
                        'Gaussian_sigma': batch_data['Gaussian_sigma'],
                        'RDP_UPPER': batch_data['RDP'].copy()
                    }
                    results[batch_size] = {
                        'Gaussian_sigma': batch_data['Gaussian_sigma'],
                        'RDP': batch_data['RDP'].copy()
                    }
            else:
                print("Historical results are missing required batch-size entries; recomputing Gaussian/RDP.")
        else:
            print("No saved results with matching params found; recomputing Gaussian/RDP.")

    batch_sizes_to_compute = []
    for batch_size in batch_size_array:
        if batch_size not in gaussian_rdp_results:
            batch_sizes_to_compute.append(batch_size)
            print(f"Batch size {batch_size} not found in cached results, will recompute")

    # Compute missing batch sizes
    if batch_sizes_to_compute:
        print(f"Computing Gaussian and RDP results for {len(batch_sizes_to_compute)} batch size(s)...")
        rdp_tasks = []

        for batch_size in batch_sizes_to_compute:
            sampling_probability = batch_size / float(settings_dict['sample size'])
            num_rounds = int(np.ceil(settings_dict['sample size'] / batch_size) * settings_dict['SGD_num_epochs'])

            # Compute Gaussian sigma
            Gaussian_optimization_func = lambda sigma: epsilon_from_sigma_gaussian(
                Gaussian_config.discretization, sigma, sampling_probability, num_rounds, settings_dict['delta'])
            Gaussian_sigma = bin_search_function(Gaussian_optimization_func,
                                                 target_epsilon,
                                                 (0.5, 20.0),
                                                 epsilon_search_resolution,
                                                 BoundType.DOMINATES)

            gaussian_rdp_results[batch_size] = {
                'Gaussian_sigma': Gaussian_sigma,
                'RDP_UPPER': {}
            }
            results[batch_size] = {'Gaussian_sigma': Gaussian_sigma}

            # Prepare RDP tasks for parallel execution
            for B in B_array:
                num_selected = max(1, int(settings_dict['communication constant'] / B))
                num_steps = int(settings_dict['D'] / B)
                block_norm = settings_dict['clip scale'] * np.sqrt(settings_dict['D'] / B) / num_selected
                task_dict = {'batch_size': batch_size,
                             'B': B,
                             'is_rdp': True,
                             'num steps': num_steps,
                             'block norm': block_norm,
                             'num selected': num_selected,
                             'sampling probability': sampling_probability,
                             'num rounds': num_rounds,
                             'delta': settings_dict['delta'],
                             'target epsilon': settings_dict['target epsilon'],
                             'epsilon tolerance': epsilon_search_resolution,
                             'Gaussian sigma': Gaussian_sigma}
                rdp_tasks.append((task_dict, None))  # No config needed for RDP

        # Execute RDP tasks in parallel
        total_rdp_tasks = len(rdp_tasks)
        completed = 0
        start_time = time.time()
        print(f"Starting RDP computations at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        print("=" * 80)

        with Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(_worker_wrapper, rdp_tasks):
                batch_size = result['batch_size']
                B = result['B']
                gaussian_rdp_results[batch_size]['RDP_UPPER'][B] = result['sigma']

                if 'RDP' not in results[batch_size]:
                    results[batch_size]['RDP'] = {}
                results[batch_size]['RDP'][B] = result['sigma']

                completed += 1
                progress_pct = (completed / total_rdp_tasks) * 100
                print(f"[{completed}/{total_rdp_tasks}] ({progress_pct:.1f}%) RDP: batch_size={batch_size}, B={B} → {result['sigma']:.4f}, {result['time']:.1f}s", flush=True)
            pool.close()
            pool.join()

        end_time = time.time()
        print("=" * 80)
        print(f"RDP completed at {time.strftime('%H:%M:%S', time.localtime(end_time))} ({(end_time - start_time)/60:.1f} minutes)")
        print("=" * 80)

    # Step 2: Run numerical experiment with bounds from Gaussian and RDP
    print(f"\nRunning numerical experiment: {numerical_config.convolution_method} ({bound_type})")
    print("Using bounds: min_sigma = 0.9 * Gaussian_sigma, max_sigma = 1.2 * RDP_sigma")
    print("=" * 80)

    numerical_tasks = []
    for batch_size in settings_dict['batch size array']:
        sampling_probability = batch_size / float(settings_dict['sample size'])
        num_rounds = int(np.ceil(settings_dict['sample size'] / batch_size) * settings_dict['SGD_num_epochs'])
        Gaussian_sigma = gaussian_rdp_results[batch_size]['Gaussian_sigma']

        for B in B_array:
            RDP_sigma = gaussian_rdp_results[batch_size]['RDP_UPPER'][B]
            num_selected = max(1, int(settings_dict['communication constant'] / B))
            num_steps = int(settings_dict['D'] / B)
            block_norm = settings_dict['clip scale'] * np.sqrt(settings_dict['D'] / B) / num_selected
            task_dict = {'batch_size': batch_size,
                         'B': B,
                         'is_rdp': False,
                         'bound_type': bound_type,
                         'num steps': num_steps,
                         'block norm': block_norm,
                         'num selected': num_selected,
                         'sampling probability': sampling_probability,
                         'num rounds': num_rounds,
                         'delta': settings_dict['delta'],
                         'target epsilon': settings_dict['target epsilon'],
                         'epsilon tolerance': epsilon_search_resolution,
                         'Gaussian sigma': Gaussian_sigma,
                         'RDP sigma': RDP_sigma}
            numerical_tasks.append((task_dict, numerical_config))

    # Execute numerical tasks in parallel
    total_numerical_tasks = len(numerical_tasks)
    completed = 0
    start_time = time.time()
    print(f"Starting numerical computations at {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # Store results with key as tuple (bound_type, convolution_method)
    method_key = (bound_type, numerical_config.convolution_method)

    with Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(_worker_wrapper, numerical_tasks):
            batch_size = result['batch_size']
            B = result['B']

            if method_key not in results[batch_size]:
                results[batch_size][method_key] = {}
            results[batch_size][method_key][B] = result['sigma']

            completed += 1
            progress_pct = (completed / total_numerical_tasks) * 100
            method_label = numerical_config.convolution_method
            print(f"[{completed}/{total_numerical_tasks}] ({progress_pct:.1f}%) {method_label} ({bound_type}): batch_size={batch_size}, B={B} → {result['sigma']:.4f}, {result['time']:.1f}s", flush=True)
        pool.close()
        pool.join()

    end_time = time.time()
    print("=" * 80)
    print(f"Numerical experiment completed at {time.strftime('%H:%M:%S', time.localtime(end_time))} ({(end_time - start_time)/60:.1f} minutes)")
    print("=" * 80)

    # Transform results to arrays for plotting compatibility
    for batch_size in results:
        if batch_size == 'params':
            continue
        for exp in list(results[batch_size].keys()):
            if exp == 'Gaussian_sigma':
                continue
            # Convert B -> sigma mapping to array format expected by plotting function
            B_values = sorted(results[batch_size][exp].keys())
            sigma_values = [results[batch_size][exp][B] for B in B_values]
            results[batch_size][exp] = np.array(sigma_values)

    return results

def plot_PREAMBLE_experiment(results):
    """Visualize PREAMBLE experiment results: sigma ratios vs. block size B.

    Uses the same styling and layout as plot_PREAMBLE_subset in the combine notebook.
    """
    batch_size_arr = [key for key in results.keys() if key != 'params']
    n_batch_sizes = len(batch_size_arr)
    
    # Create subplots - arrange in a grid
    if n_batch_sizes == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]  # Make it iterable
    elif n_batch_sizes <= 2:
        fig, axes = plt.subplots(1, n_batch_sizes, figsize=(15, 6))
    elif n_batch_sizes <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
    else:
        # For more than 4 batch sizes, create a larger grid
        cols = 3
        rows = (n_batch_sizes + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        axes = axes.flatten()
    
    B_array = results['params']['B array']

    def _method_name_from_key(method_key) -> str:
        if method_key == 'RDP':
            return 'RDP upper remove'
        if isinstance(method_key, tuple) and len(method_key) == 2:
            bound_type, conv_method = method_key
            bound_label = 'upper' if bound_type == BoundType.DOMINATES else 'lower'
            if conv_method == ConvolutionMethod.FFT:
                return f'FFT {bound_label} both'
            if conv_method == ConvolutionMethod.GEOM:
                return f'Mult {bound_label} both'
            if conv_method == ConvolutionMethod.COMBINED:
                return f'Combined {bound_label} both'
        return str(method_key)

    def get_style(method_name: str) -> dict:
        method_lower = method_name.lower()
        if method_name == 'best':
            label = 'PREAMBLE PLD'
        elif method_name == 'RDP upper remove':
            label = 'PREAMBLE RDP (FS25)'
        elif 'fft' in method_lower and 'upper' in method_lower:
            label = 'PREAMBLE PLD (FFT)'
        elif ('mult' in method_lower or 'geometric' in method_lower) and 'upper' in method_lower:
            label = 'PREAMBLE PLD (Mult)'
        elif 'combined' in method_lower and 'upper' in method_lower:
            label = 'PREAMBLE PLD (Combined)'
        else:
            label = method_name

        if method_name == 'best':
            color = 'blue'
            linestyle = '--'
            marker = 'o'
        elif method_name == 'RDP upper remove':
            color = 'red'
            linestyle = ':'
            marker = 's'
        elif 'combined' in method_lower:
            color = 'green'
            linestyle = '--' if 'upper' in method_lower else ':'
            marker = '^'
        elif 'mult' in method_lower or 'geometric' in method_lower:
            color = 'blue'
            linestyle = '--' if 'upper' in method_lower else ':'
            marker = 'D'
        elif 'fft' in method_lower:
            color = 'magenta'
            linestyle = '--' if 'upper' in method_lower else ':'
            marker = 'v'
        elif 'rdp' in method_lower:
            color = 'green'
            linestyle = '-'
            marker = '*'
        else:
            color = 'gray'
            linestyle = '-'
            marker = 'x'
        return {'color': color, 'linestyle': linestyle, 'marker': marker, 'label': label}

    for i, batch_size in enumerate(batch_size_arr):
        ax = axes[i]
        batch_results = results[batch_size]
        gaussian_sigma = batch_results.get('Gaussian_sigma')
        if isinstance(gaussian_sigma, dict):
            gaussian_sigma = list(gaussian_sigma.values())[0] if gaussian_sigma else None
        elif isinstance(gaussian_sigma, set):
            gaussian_sigma = list(gaussian_sigma)[0] if gaussian_sigma else None
        elif isinstance(gaussian_sigma, (list, tuple)):
            gaussian_sigma = gaussian_sigma[0] if len(gaussian_sigma) > 0 else None

        if gaussian_sigma is None or not isinstance(gaussian_sigma, (int, float, np.number)):
            print(f"Warning: Invalid Gaussian_sigma value for batch_size={batch_size}: {gaussian_sigma}")
            continue

        available_methods = [key for key in batch_results.keys() if key != 'Gaussian_sigma']
        methods_to_plot = []
        for method in available_methods:
            method_name = _method_name_from_key(method)
            if 'lower' in method_name.lower():
                continue
            methods_to_plot.append((method, method_name))

        for method_key, method_name in methods_to_plot:
            method_data = batch_results[method_key]
            if isinstance(method_data, dict) and 'data' in method_data:
                sigma_arr = method_data['data']
            else:
                sigma_arr = method_data
            if not isinstance(sigma_arr, np.ndarray):
                sigma_arr = np.array(sigma_arr)
            if len(sigma_arr) != len(B_array):
                print(f"Warning: Length mismatch for {method_name} in batch_size={batch_size}: "
                      f"sigma_arr has {len(sigma_arr)} elements, B_array has {len(B_array)}")
                continue

            style = get_style(method_name)
            ax.plot(B_array, sigma_arr / gaussian_sigma,
                   color=style['color'], linestyle=style['linestyle'],
                   marker=style['marker'], markersize=10,
                   label=style['label'], linewidth=3)

        # Add horizontal line at y=1 for Gaussian baseline
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=3, label='Gaussian', zorder=1)

        ax.set_xlabel('B', fontsize=24, fontweight='bold')
        ax.set_ylabel('sigma ratio', fontsize=24, fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim(0.9, 2.4)
        ax.set_title(f'Batch Size: {batch_size}', fontsize=26, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=14)

    # Hide unused subplots
    for i in range(n_batch_sizes, len(axes)):
        axes[i].set_visible(False)

    # Add a single legend below all subplots with larger font
    handles, labels = axes[0].get_legend_handles_labels()
    if len(labels) > 0:
        fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 4), fontsize=22,
                   framealpha=0.9, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        plt.subplots_adjust(bottom=0.10)
    else:
        plt.tight_layout()
    plt.show()


def _build_sigma_map(settings_dict: dict) -> dict[int, float]:
    batch_sizes = settings_dict["batch size array"]
    if "sigma array" in settings_dict:
        sigma_array = settings_dict["sigma array"]
        if len(sigma_array) != len(batch_sizes):
            raise ValueError("sigma array must match batch size array length")
        return dict(zip(batch_sizes, sigma_array))
    if "sigma" in settings_dict:
        return {batch_size: float(settings_dict["sigma"]) for batch_size in batch_sizes}
    raise ValueError("settings_dict must include 'sigma' or 'sigma array'")


def run_preamble_epsilon_experiment(
    settings_dict: dict,
    numerical_config: AllocationSchemeConfig,
    Gaussian_config: SchemeConfig,
) -> dict:
    """Run PREAMBLE epsilon experiment with fixed sigma per batch size."""
    print("Starting PREAMBLE epsilon experiment...", flush=True)
    dp_bound_type = BoundType.DOMINATES
    total_times = {"original": 0.0}
    if "sigma array" in settings_dict:
        if len(settings_dict["sigma array"]) != len(settings_dict["batch size array"]):
            raise ValueError("sigma array must match batch size array length")
    sigma_map = _build_sigma_map(settings_dict)
    B_array = settings_dict["B array"]

    print("PREAMBLE epsilon experiment configuration:", flush=True)
    print(f"  Sample size: {settings_dict['sample size']}", flush=True)
    print(f"  D: {settings_dict['D']}", flush=True)
    print(f"  Communication constant: {settings_dict['communication constant']}", flush=True)
    print(f"  SGD_num_epochs: {settings_dict['SGD_num_epochs']}", flush=True)
    print(f"  Batch size array: {settings_dict['batch size array']}", flush=True)
    print(f"  Sigma array: {np.array([sigma_map[b] for b in settings_dict['batch size array']])}", flush=True)
    print(f"  B array: {B_array}", flush=True)
    print(f"  Delta: {settings_dict['delta']}", flush=True)
    print(f"  Clip scale: {settings_dict['clip scale']}", flush=True)
    print(
        "  Numerical config: "
        f"loss_discretization={numerical_config.loss_discretization}, "
        f"tail_truncation={numerical_config.tail_truncation}, "
        f"max_grid_mult={numerical_config.max_grid_mult}, "
        f"max_grid_FFT={numerical_config.max_grid_FFT}, "
        f"convolution_method={numerical_config.convolution_method}",
        flush=True,
    )
    print(f"  Gaussian discretization: {Gaussian_config.discretization}", flush=True)
    print(f"  Bound type: {dp_bound_type}", flush=True)
    print("=" * 80, flush=True)

    results = {
        "params": {
            "sample size": settings_dict["sample size"],
            "D": settings_dict["D"],
            "communication constant": settings_dict["communication constant"],
            "SGD_num_epochs": settings_dict["SGD_num_epochs"],
            "B array": B_array,
            "delta": settings_dict["delta"],
            "clip scale": settings_dict["clip scale"],
            "sigma array": np.array([sigma_map[b] for b in settings_dict["batch size array"]]),
            "numerical config": {
                "loss_discretization": numerical_config.loss_discretization,
                "tail_truncation": numerical_config.tail_truncation,
                "max_grid_mult": numerical_config.max_grid_mult,
                "max_grid_FFT": numerical_config.max_grid_FFT,
            },
            "Gaussian config": {"loss_discretization": Gaussian_config.discretization},
            "bound_type": dp_bound_type,
            "convolution_method": numerical_config.convolution_method,
        }
    }

    print(f"\nRunning numerical experiment: {numerical_config.convolution_method} ({dp_bound_type})")
    print("Using bounds: fixed sigma per batch size")
    print("=" * 80)

    for batch_size in settings_dict["batch size array"]:
        sampling_probability = batch_size / float(settings_dict["sample size"])
        num_rounds = int(
            np.ceil(settings_dict["sample size"] / batch_size)
            * settings_dict["SGD_num_epochs"]
        )
        sigma_value = sigma_map[batch_size]
        print(
            f"Batch size={batch_size} sigma={sigma_value:.6g} "
            f"sampling_prob={sampling_probability:.6g} num_rounds={num_rounds}",
            flush=True,
        )

        gaussian_eps = epsilon_from_sigma_gaussian(
            discretization=Gaussian_config.discretization,
            sigma=sigma_value,
            sampling_probability=sampling_probability,
            num_rounds=num_rounds,
            delta=settings_dict["delta"],
        )
        print(f"  Gaussian epsilon={gaussian_eps:.6g}", flush=True)

        results[batch_size] = {
            "Gaussian_epsilon": gaussian_eps,
            "RDP": {},
        }

        batch_times = {"original": 0.0}
        upper_key = (BoundType.DOMINATES, numerical_config.convolution_method)
        results[batch_size].setdefault(upper_key, {})

        for B in B_array:
            print(f"  B={B}...", flush=True)
            num_selected = max(1, int(settings_dict["communication constant"] / B))
            num_steps = int(settings_dict["D"] / B)
            block_norm = (
                settings_dict["clip scale"] * np.sqrt(settings_dict["D"] / B) / num_selected
            )
            params = PrivacyParams(
                sigma=sigma_value / block_norm,
                num_steps=num_steps,
                num_selected=num_selected,
                delta=settings_dict["delta"],
            )

            rdp_eps = epsilon_from_sigma_allocation_RDP(
                params=params,
                sampling_probability=sampling_probability,
                num_rounds=num_rounds,
            )
            results[batch_size]["RDP"][B] = rdp_eps
            print(f"    RDP epsilon={rdp_eps:.6g}", flush=True)

            config_copy = AllocationSchemeConfig(
                loss_discretization=numerical_config.loss_discretization,
                tail_truncation=numerical_config.tail_truncation,
                max_grid_FFT=numerical_config.max_grid_FFT,
                max_grid_mult=numerical_config.max_grid_mult,
                convolution_method=numerical_config.convolution_method,
            )
            orig_start = time.time()
            pld_eps_upper = epsilon_from_sigma_allocation(
                params=params,
                config=config_copy,
                sampling_probability=sampling_probability,
                num_rounds=num_rounds,
                bound_type=BoundType.DOMINATES,
            )
            results[batch_size][upper_key][B] = pld_eps_upper
            orig_elapsed = time.time() - orig_start
            batch_times["original"] += orig_elapsed
            total_times["original"] += orig_elapsed
            print(
                "    PLD (Upper) epsilon="
                f"{pld_eps_upper:.6g} time={orig_elapsed:.2f}s",
                flush=True,
            )
        print(
            f"  PLD (Upper/Lower) total time for batch_size={batch_size}: "
            f"{batch_times['original']:.2f}s",
            flush=True,
        )

    print(
        f"PLD (Upper/Lower) total time for experiment: "
        f"{total_times['original']:.2f}s",
        flush=True,
    )

    for batch_size in results:
        if batch_size == "params":
            continue
        for key in list(results[batch_size].keys()):
            if key == "Gaussian_epsilon":
                continue
            B_values = sorted(results[batch_size][key].keys())
            epsilon_values = [results[batch_size][key][B] for B in B_values]
            results[batch_size][key] = np.array(epsilon_values)

    return results


def plot_preamble_epsilon_experiment(results: dict) -> None:
    """Plot epsilon vs B for the PREAMBLE epsilon experiment."""
    batch_sizes = [key for key in results.keys() if key != "params"]
    B_array = results["params"]["B array"]

    n_batch_sizes = len(batch_sizes)
    if n_batch_sizes == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    elif n_batch_sizes <= 2:
        fig, axes = plt.subplots(1, n_batch_sizes, figsize=(15, 6))
    else:
        rows = (n_batch_sizes + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        axes = axes.flatten()

    for ax, batch_size in zip(axes, batch_sizes):
        gaussian_eps = results[batch_size]["Gaussian_epsilon"]
        ax.plot(B_array, [gaussian_eps] * len(B_array), "o-", label="Gaussian")

        for key, values in results[batch_size].items():
            if key == "Gaussian_epsilon":
                continue
            if key == "RDP":
                label = "RDP"
            elif isinstance(key, tuple) and key[0] == BoundType.DOMINATES:
                label = "PLD (Upper)"
            elif isinstance(key, tuple) and key[0] == BoundType.IS_DOMINATED:
                label = "PLD (Lower)"
            else:
                label = "PLD (Envelope)"
            ax.plot(B_array, values, marker="s", label=label)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Block size B")
        ax.set_ylabel("Epsilon")
        ax.set_title(f"Batch size = {batch_size}")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()

    for ax in axes[len(batch_sizes):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

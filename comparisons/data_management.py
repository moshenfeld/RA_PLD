"""
Centralized I/O utilities for experiment data.

Provides consistent save/load functionality for all experiment types,
supporting both pickle and JSON/CSV formats.
"""
import os
import datetime
import pickle
import json
import glob
import pathlib
import warnings
from typing import Dict, Any, Optional, Callable
import numpy as np
import pandas as pd


def _get_data_dir():
    """Get the absolute path to the data directory.

    Locates data directory relative to this file's location.
    Implementation navigates from comparisons/ into data/.
    """
    # Get the directory containing this file (comparisons/)
    current_file = pathlib.Path(__file__).resolve()
    # Get the comparisons directory
    comparisons_dir = current_file.parent
    # Return the data directory path within comparisons/
    return comparisons_dir / "data"


def save_results(results, experiment_name):
    """Save experiment results to pickle file with timestamp.

    Creates subdirectory for experiment and saves with timestamped filename.
    Implementation uses pickle for serialization.
    """
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_dir = _get_data_dir()
    full_dir_name = data_dir / experiment_name
    os.makedirs(full_dir_name, exist_ok=True)
    filename = full_dir_name / f"{experiment_name}_{date_time}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f)


def load_results(experiment_name):
    """Load most recent experiment results from pickle file.

    Searches for latest timestamped file matching experiment name.
    Implementation uses file creation time to identify most recent result.
    Returns None if no results exist.
    """
    data_dir = _get_data_dir()
    search_pattern = str(data_dir / experiment_name / f"{experiment_name}*.pkl")
    files = glob.glob(search_pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, "rb") as f:
        results = pickle.load(f)
    return results


def merge_with_historical_results(experiment_name, results):
    """Merge current results with historical results that have matching params.

    Searches through existing experiment files in decreasing order of time (newest to oldest)
    until it finds a file where all params match the input results. If found, copies any
    keys from the historical file that don't exist in the new results.

    Args:
        experiment_name: Name of the experiment
        results: Dictionary containing current results with a 'params' key

    Returns:
        Updated results dictionary with historical data merged in, or original results if no match found
    """
    data_dir = _get_data_dir()
    search_pattern = str(data_dir / experiment_name / f"{experiment_name}*.pkl")
    existing_files = glob.glob(search_pattern)

    if not existing_files:
        # No historical files found
        return results

    # Sort files by creation time in decreasing order (newest first)
    existing_files_sorted = sorted(existing_files, key=os.path.getctime, reverse=True)

    # Check if results has params key
    if 'params' not in results:
        # Can't compare without params
        return results

    current_params = results['params']

    # Search through historical files
    for historical_file in existing_files_sorted:
        try:
            with open(historical_file, "rb") as f:
                historical_results = pickle.load(f)

            # Check if historical results has params
            if 'params' not in historical_results:
                continue

            historical_params = historical_results['params']

            # Check if all params match
            if _params_match(current_params, historical_params):
                # Found a matching file! Merge results
                merged_results = results.copy()

                # Copy keys from historical results that don't exist in new results
                for key in historical_results.keys():
                    if key not in merged_results:
                        merged_results[key] = historical_results[key]

                print(f"Merged results with historical file: {os.path.basename(historical_file)}")
                return merged_results

        except Exception as e:
            # If we can't read a file, just skip it
            warnings.warn(f"Could not read {historical_file}: {e}")
            continue

    # No matching historical file found
    return results


def _params_match(params1, params2):
    """Check if two parameter dictionaries are identical.

    Compares all keys and values in both dictionaries, handling nested structures
    and numpy arrays appropriately.

    Args:
        params1: First parameter dictionary
        params2: Second parameter dictionary

    Returns:
        True if all parameters match, False otherwise
    """
    # Check if keys match
    if set(params1.keys()) != set(params2.keys()):
        return False

    # Check if all values match
    for key in params1.keys():
        val1 = params1[key]
        val2 = params2[key]

        # Handle numpy arrays
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False
        # Handle dictionaries (nested params)
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not _params_match(val1, val2):
                return False
        # Handle regular values
        else:
            if val1 != val2:
                return False

    return True


def load_results_with_matching_params(experiment_name: str, params: dict) -> tuple[Optional[dict], Optional[pathlib.Path]]:
    """Return the most recent results whose params match the provided dictionary.

    Scans existing experiment files (newest first) and returns the first result file
    whose `params` payload matches `params` exactly. If no match is found, returns
    `(None, None)`.
    """
    data_dir = _get_data_dir()
    search_pattern = str(data_dir / experiment_name / f"{experiment_name}*.pkl")
    existing_files = glob.glob(search_pattern)

    if not existing_files:
        return None, None

    existing_files_sorted = sorted(existing_files, key=os.path.getctime, reverse=True)

    for historical_file in existing_files_sorted:
        try:
            with open(historical_file, "rb") as f:
                historical_results = pickle.load(f)

            if 'params' not in historical_results:
                continue

            if _params_match(params, historical_results['params']):
                return historical_results, pathlib.Path(historical_file)

        except Exception as e:
            warnings.warn(f"Could not read {historical_file}: {e}")
            continue

    return None, None


def combine_PREAMBLE_experiment_files(experiment_name: str, datetime_suffix1: str, datetime_suffix2: str, print_keys) -> dict:
    """
    Combine two PREAMBLE experiment result files into a single results dictionary.

    The function properly merges the results by:
    1. Keeping the 'params' key from both files (they should match)
    2. Iterating over all other keys and combining them from both files
    3. If a key exists in both files, the second file's value takes precedence

    Args:
        experiment_name: Name of the experiment (e.g., 'PREAMBLE_experiment')
        datetime_suffix1: Date-time suffix for the first file (e.g., '2025-10-19_18-53-49')
        datetime_suffix2: Date-time suffix for the second file (e.g., '2025-10-20_00-04-35')

    Returns:
        Combined results dictionary

    Example:
        combined_results = combine_PREAMBLE_experiment_files('PREAMBLE_experiment', '2025-10-19_18-53-49', '2025-10-20_00-04-35')
    """
    # Construct full file paths
    data_dir = _get_data_dir()
    file1_path = data_dir / experiment_name / f"{experiment_name}_{datetime_suffix1}.pkl"
    file2_path = data_dir / experiment_name / f"{experiment_name}_{datetime_suffix2}.pkl"

    # Load both experiment files
    print(f"Loading {file1_path}...")
    if not file1_path.exists():
        raise FileNotFoundError(f"File not found: {file1_path}")
    with open(file1_path, "rb") as f:
        results1 = pickle.load(f)

    print(f"Loading {file2_path}...")
    if not file2_path.exists():
        raise FileNotFoundError(f"File not found: {file2_path}")
    with open(file2_path, "rb") as f:
        results2 = pickle.load(f)

    if results1 is None or results2 is None:
        raise ValueError("One or both experiment files could not be loaded")

    if print_keys:
        print(f'results1.keys: {results1.keys()}')
        for key in results1.keys():
            print(f'\t results1[{key}].keys: {results1[key].keys()}')
        print(f'results2.keys: {results2.keys()}')
        for key in results2.keys():
            print(f'\t results2[{key}].keys: {results2[key].keys()}')

    # Initialize combined results with params from first file
    combined_results = {}
    for key in results1.keys():
        combined_results[key] = {}
        for sub_key in results1[key].keys():
            combined_results[key][sub_key] = results1[key][sub_key]
        if key in results2.keys():
            for sub_key in results2[key].keys():
                if sub_key in combined_results:
                    print(f"Key '[{key}][{sub_key}]' exists in both files - using value from {file2_path}")
                combined_results[key][sub_key] = results2[key][sub_key]

    if print_keys:
        print(f'combined_results.keys: {combined_results.keys()}')
        for key in combined_results.keys():
            print(f'\t combined_results[{key}].keys: {combined_results[key].keys()}')

    print(f"\nSuccessfully combined results:")
    print(f"  {file1_path}: {len([k for k in results1.keys() if k != 'params'])} data keys")
    print(f"  {file2_path}: {len([k for k in results2.keys() if k != 'params'])} data keys")
    print(f"  Combined: {len([k for k in combined_results.keys() if k != 'params'])} data keys")

    return combined_results


def save_experiment_data_json(
    results: Dict[str, Any],
    experiment_name: str,
    *,
    data_converter: Optional[Callable] = None
):
    """Save experiment results to JSON/CSV for reuse outside pickles.

    Generic version that can be customized with a data_converter function.

    Args:
        results: Experiment results dictionary
        experiment_name: Base name for output files
        data_converter: Optional function to convert experiment data to CSV rows
    """
    data_dir = os.path.dirname(experiment_name)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    def _convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if hasattr(value, "__dict__"):
            return vars(value)
        return value

    json_ready = _convert(results)
    with open(f"{experiment_name}.json", "w") as f:
        json.dump(json_ready, f, indent=2)

    # Use custom data converter if provided
    if data_converter:
        data_converter(results, experiment_name)

    print(f"Saved experiment data to {experiment_name}.json")


def load_experiment_data_json(experiment_name: str) -> Optional[Dict[str, Any]]:
    """Load JSON-saved experiment results.

    Generic loader that restores numpy arrays from lists.
    """
    json_file = f"{experiment_name}.json"
    if not os.path.exists(json_file):
        print(f"No data found at {json_file}")
        return None

    with open(json_file, "r") as f:
        results = json.load(f)

    def _restore(value):
        if isinstance(value, list):
            if value and all(not isinstance(v, dict) for v in value):
                try:
                    return np.array(value)
                except Exception:
                    return [_restore(v) for v in value]
            return [_restore(v) for v in value]
        if isinstance(value, dict):
            return {k: _restore(v) for k, v in value.items()}
        return value

    return _restore(results)

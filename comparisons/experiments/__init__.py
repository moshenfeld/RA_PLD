# experiments package

# Import common utilities for easy access
from comparisons.data_management import (
    save_results,
    load_results,
    merge_with_historical_results,
    save_experiment_data_json,
    load_experiment_data_json,
)

from comparisons.plotting_utils import (
    save_figure,
    configure_matplotlib_for_publication,
    show_or_close_figure,
    setup_common_axis_formatting,
    add_shared_legend,
    finalize_plot,
)

__all__ = [
    # I/O functions
    'save_results',
    'load_results',
    'merge_with_historical_results',
    'save_experiment_data_json',
    'load_experiment_data_json',
    # Plotting functions
    'save_figure',
    'configure_matplotlib_for_publication',
    'show_or_close_figure',
    'setup_common_axis_formatting',
    'add_shared_legend',
    'finalize_plot',
]

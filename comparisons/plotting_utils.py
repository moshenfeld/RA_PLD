"""
Centralized plotting utilities for experiment visualizations.

Provides common plotting functions and utilities used across multiple experiments.
This module consolidates duplicated plotting code and provides high-level APIs
for creating consistent, publication-quality visualizations.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any, Union
from comparisons.visualization_definitions import (
    method_colors,
    method_linestyles,
    method_markers,
    method_drawing_order,
    SAVEFIG_DPI
)


# ============================================================================
# Low-Level Utilities (File I/O, Configuration)
# ============================================================================

def save_figure(
    fig,
    *,
    save_plots: bool = False,
    plots_dir: Optional[str] = None,
    filename: str,
    default_plots_subdir: str = "plots",
    dpi: int = 300
):
    """Save a matplotlib figure to file.

    Centralizes the common pattern of saving plots with consistent parameters.

    Args:
        fig: Matplotlib figure to save
        save_plots: Whether to save the plot
        plots_dir: Directory to save plots (if None, uses default_plots_subdir)
        filename: Filename for the saved plot
        default_plots_subdir: Subdirectory name to use if plots_dir is None
        dpi: Resolution for saved figure

    Returns:
        Path to saved file if saved, None otherwise
    """
    if not save_plots:
        return None

    if plots_dir is None:
        # Use default plots directory relative to caller
        plots_dir = os.path.join(os.path.dirname(__file__), default_plots_subdir)

    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, filename)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    return output_path


def configure_matplotlib_for_publication(dpi: int = 600, savefig_dpi: int = 300):
    """Configure matplotlib with high-resolution settings for publication.

    Centralizes the common pattern of setting matplotlib rcParams.

    Args:
        dpi: Display DPI
        savefig_dpi: DPI for saved figures
    """
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = savefig_dpi
    plt.rcParams['savefig.bbox'] = 'tight'


def show_or_close_figure(fig, show_plots: bool = True):
    """Show or close a matplotlib figure.

    Centralizes the common pattern of conditionally displaying plots.

    Args:
        fig: Matplotlib figure
        show_plots: Whether to display the plot
    """
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def finalize_plot(
    fig,
    *,
    save_plots: bool = False,
    show_plots: bool = True,
    plots_dir: Optional[str] = None,
    filename: str = "plot.png",
    dpi: int = 300
):
    """Complete plotting workflow: save and show/close figure.

    Combines save_figure and show_or_close_figure for the common pattern
    used across all plotting functions.

    Args:
        fig: Matplotlib figure
        save_plots: Whether to save the plot
        show_plots: Whether to display the plot
        plots_dir: Directory to save plots
        filename: Filename for saved plot
        dpi: Resolution for saved figure

    Returns:
        Path to saved file if saved, None otherwise
    """
    output_path = None
    if save_plots:
        if plots_dir is None:
            plots_dir = os.path.join(os.path.dirname(__file__), "plots")
        output_path = save_figure(
            fig,
            save_plots=True,
            plots_dir=plots_dir,
            filename=filename,
            dpi=dpi
        )

    show_or_close_figure(fig, show_plots=show_plots)
    return output_path


# ============================================================================
# Subplot Creation and Layout
# ============================================================================

def create_subplot_grid(
    n_plots: int,
    *,
    layout: str = 'auto',
    figsize_per_plot: Tuple[float, float] = (6, 5),
    **subplot_kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a subplot grid with automatic layout detection.

    Automatically determines the optimal grid layout based on number of plots.
    Handles single subplot edge case to ensure consistent return type.

    Args:
        n_plots: Number of subplots to create
        layout: Layout strategy ('auto', 'horizontal', 'vertical', 'square')
        figsize_per_plot: Size of each subplot (width, height)
        **subplot_kwargs: Additional arguments passed to plt.subplots

    Returns:
        Tuple of (fig, axes) where axes is always a 1D numpy array
    """
    if n_plots <= 0:
        raise ValueError(f"n_plots must be positive, got {n_plots}")

    # Determine grid layout
    if layout == 'auto':
        if n_plots == 1:
            nrows, ncols = 1, 1
        elif n_plots <= 2:
            nrows, ncols = 1, n_plots
        elif n_plots <= 4:
            nrows, ncols = 2, 2
        else:
            # For more plots, use roughly square grid
            ncols = int(np.ceil(np.sqrt(n_plots)))
            nrows = int(np.ceil(n_plots / ncols))
    elif layout == 'horizontal':
        nrows, ncols = 1, n_plots
    elif layout == 'vertical':
        nrows, ncols = n_plots, 1
    elif layout == 'square':
        ncols = int(np.ceil(np.sqrt(n_plots)))
        nrows = int(np.ceil(n_plots / ncols))
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Calculate figure size
    figwidth = ncols * figsize_per_plot[0]
    figheight = nrows * figsize_per_plot[1]

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(figwidth, figheight), **subplot_kwargs)

    # Ensure axes is always a 1D array
    if n_plots == 1:
        axes = np.array([axes])
    else:
        axes = np.atleast_1d(axes).flatten()

    return fig, axes


# ============================================================================
# Axis Formatting
# ============================================================================

def format_axis(
    ax,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    grid: bool = True,
    grid_alpha: float = 0.3,
    grid_linestyle: str = '--',
    grid_linewidth: float = 0.5,
    tick_labelsize: int = 12,
    label_fontsize: int = 14,
    title_fontsize: int = 16,
    title_fontweight: str = 'bold',
    title_pad: float = 10
):
    """Apply comprehensive formatting to a matplotlib axis.

    Centralizes all common axis formatting operations in one function.

    Args:
        ax: Matplotlib axis to format
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        xscale: X-axis scale ('log', 'linear', etc.)
        yscale: Y-axis scale ('log', 'linear', etc.)
        xlim: X-axis limits (min, max)
        ylim: Y-axis limits (min, max)
        grid: Whether to show grid
        grid_alpha: Grid transparency
        grid_linestyle: Grid line style
        grid_linewidth: Grid line width
        tick_labelsize: Font size for tick labels
        label_fontsize: Font size for axis labels
        title_fontsize: Font size for title
        title_fontweight: Font weight for title
        title_pad: Padding above title
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=title_pad)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle, linewidth=grid_linewidth)
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)


# Alias for backward compatibility
def setup_common_axis_formatting(ax, **kwargs):
    """Deprecated: Use format_axis instead."""
    format_axis(ax, **kwargs)


# ============================================================================
# Legend Management
# ============================================================================

def extract_legend_handles_labels(axes: Union[plt.Axes, np.ndarray]) -> Tuple[List, List]:
    """Extract unique legend handles and labels from axes.

    Args:
        axes: Single axis or array of axes

    Returns:
        Tuple of (handles, labels) with duplicates removed
    """
    if isinstance(axes, plt.Axes):
        return axes.get_legend_handles_labels()

    # Flatten axes if necessary
    axes_flat = np.atleast_1d(axes).flatten()

    # Collect all handles and labels
    all_handles = []
    all_labels = []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)

    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for handle, label in zip(all_handles, all_labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    return unique_handles, unique_labels


def add_shared_legend_below(
    fig,
    axes: Union[plt.Axes, np.ndarray],
    *,
    ncol: Union[int, str] = 'auto',
    fontsize: int = 12,
    bbox_to_anchor: Tuple[float, float] = (0.5, -0.05),
    loc: str = 'lower center',
    frameon: bool = True,
    framealpha: float = 0.9,
    fancybox: bool = True,
    shadow: bool = True,
    bottom_margin: float = 0.08
):
    """Add a shared legend below all subplots.

    Extracts unique legend entries from all axes and places a single
    legend below the figure.

    Args:
        fig: Matplotlib figure
        axes: Axis or array of axes to extract legend from
        ncol: Number of columns ('auto' or integer)
        fontsize: Font size for legend text
        bbox_to_anchor: Position of legend anchor
        loc: Location of legend
        frameon: Whether to draw frame around legend
        framealpha: Frame transparency
        fancybox: Whether to use fancy box
        shadow: Whether to add shadow
        bottom_margin: Bottom margin adjustment for tight_layout

    Returns:
        The legend object
    """
    handles, labels = extract_legend_handles_labels(axes)

    if ncol == 'auto':
        ncol = min(len(labels), 4)

    legend = fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        fontsize=fontsize,
        frameon=frameon,
        framealpha=framealpha,
        fancybox=fancybox,
        shadow=shadow
    )

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, bottom_margin, 1, 1])

    return legend


# Alias for backward compatibility
def add_shared_legend(fig, handles, labels, **kwargs):
    """Deprecated: Use add_shared_legend_below instead."""
    return fig.legend(handles, labels, **kwargs)


# ============================================================================
# Method Plotting with Automatic Styling
# ============================================================================

def get_method_style(method_name: str) -> Dict[str, Any]:
    """Get the plotting style for a method from visualization_definitions.

    Args:
        method_name: Name of the method

    Returns:
        Dictionary with 'color', 'linestyle', 'marker' keys
    """
    return {
        'color': method_colors.get(method_name, 'gray'),
        'linestyle': method_linestyles.get(method_name, '-'),
        'marker': method_markers.get(method_name, 'o')
    }


def sort_methods_by_drawing_order(method_names: List[str]) -> List[str]:
    """Sort method names according to drawing order from visualization_definitions.

    Methods not in the drawing order list are placed at the end.

    Args:
        method_names: List of method names to sort

    Returns:
        Sorted list of method names
    """
    def get_order_index(name):
        try:
            return method_drawing_order.index(name)
        except ValueError:
            return len(method_drawing_order)

    return sorted(method_names, key=get_order_index)


def plot_methods_on_axis(
    ax,
    x_data: np.ndarray,
    method_data: Dict[str, np.ndarray],
    *,
    sort_methods: bool = True,
    linewidth: float = 2.0,
    markersize: float = 5,
    alpha: float = 0.9,
    style_overrides: Optional[Dict[str, Dict[str, Any]]] = None
):
    """Plot multiple methods on a single axis with automatic styling.

    Automatically applies colors, linestyles, and markers from visualization_definitions.
    Optionally sorts methods by drawing order for consistent layering.

    Args:
        ax: Matplotlib axis to plot on
        x_data: X-axis data (shared across all methods)
        method_data: Dictionary mapping method names to y-axis data
        sort_methods: Whether to sort methods by drawing order
        linewidth: Width of plot lines
        markersize: Size of markers
        alpha: Transparency of lines
        style_overrides: Optional dictionary of style overrides per method
    """
    if style_overrides is None:
        style_overrides = {}

    # Sort methods if requested
    method_names = list(method_data.keys())
    if sort_methods:
        method_names = sort_methods_by_drawing_order(method_names)

    # Plot each method
    for method_name in method_names:
        y_data = method_data[method_name]

        # Get default style
        style = get_method_style(method_name)

        # Apply overrides
        if method_name in style_overrides:
            style.update(style_overrides[method_name])

        # Plot
        ax.plot(
            x_data,
            y_data,
            label=method_name,
            color=style['color'],
            linestyle=style['linestyle'],
            marker=style['marker'],
            linewidth=linewidth,
            markersize=markersize,
            alpha=alpha
        )


# ============================================================================
# High-Level Plotting Functions
# ============================================================================

def create_method_comparison_figure(
    data_by_config: Dict[Any, Dict[str, np.ndarray]],
    x_arrays_by_config: Dict[Any, np.ndarray],
    config_titles: Dict[Any, str],
    *,
    xlabel: str = 'x',
    ylabel: str = 'y',
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    layout: str = 'auto',
    figsize_per_plot: Tuple[float, float] = (6, 5),
    linewidth: float = 2.0,
    markersize: float = 5,
    legend_fontsize: int = 14,
    legend_ncol: Union[int, str] = 'auto',
    legend_bottom_margin: Optional[float] = None,
    save_plots: bool = False,
    show_plots: bool = True,
    plots_dir: Optional[str] = None,
    filename: str = 'method_comparison.png'
) -> plt.Figure:
    """Create a multi-subplot figure comparing methods across configurations.

    This is a high-level function that handles the common pattern of:
    - Multiple subplots (one per configuration)
    - Each subplot compares multiple methods
    - Automatic styling from visualization_definitions
    - Shared legend below all subplots

    Args:
        data_by_config: Nested dict {config_key: {method_name: y_values}}
        x_arrays_by_config: Dict {config_key: x_values}
        config_titles: Dict {config_key: subplot_title}
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        xscale: Scale for x-axis ('log', 'linear', None)
        yscale: Scale for y-axis ('log', 'linear', None)
        layout: Subplot layout strategy
        figsize_per_plot: Size of each subplot
        linewidth: Width of plot lines
        markersize: Size of markers
        legend_fontsize: Font size for legend text (default: 14)
        legend_ncol: Number of legend columns ('auto' or integer)
        legend_bottom_margin: Bottom margin for legend (default: 0.08, uses add_shared_legend_below default)
        save_plots: Whether to save the figure
        show_plots: Whether to display the figure
        plots_dir: Directory to save plots
        filename: Filename for saved plot

    Returns:
        The created matplotlib figure
    """
    # Configure matplotlib
    configure_matplotlib_for_publication(dpi=600, savefig_dpi=SAVEFIG_DPI)

    # Get configuration keys
    config_keys = list(data_by_config.keys())
    n_configs = len(config_keys)

    if n_configs == 0:
        raise ValueError("data_by_config must contain at least one configuration")

    # Create subplot grid
    fig, axes = create_subplot_grid(n_configs, layout=layout, figsize_per_plot=figsize_per_plot)

    # Plot each configuration
    for idx, config_key in enumerate(config_keys):
        ax = axes[idx]
        x_data = x_arrays_by_config[config_key]
        method_data = data_by_config[config_key]
        title = config_titles.get(config_key, str(config_key))

        # Plot methods on this axis
        plot_methods_on_axis(
            ax,
            x_data,
            method_data,
            linewidth=linewidth,
            markersize=markersize
        )

        # Format axis
        format_axis(
            ax,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            xscale=xscale,
            yscale=yscale
        )

    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].set_visible(False)

    # Add shared legend
    legend_kwargs = {'fontsize': legend_fontsize, 'ncol': legend_ncol}
    if legend_bottom_margin is not None:
        legend_kwargs['bottom_margin'] = legend_bottom_margin
    add_shared_legend_below(fig, axes, **legend_kwargs)

    # Finalize
    finalize_plot(
        fig,
        save_plots=save_plots,
        show_plots=show_plots,
        plots_dir=plots_dir,
        filename=filename,
        dpi=SAVEFIG_DPI
    )

    return fig


def create_ratio_comparison_figure(
    data_by_config: Dict[Any, Dict[str, np.ndarray]],
    x_arrays_by_config: Dict[Any, np.ndarray],
    baseline_method: str,
    config_titles: Dict[Any, str],
    *,
    xlabel: str = 'x',
    ylabel: str = 'ratio',
    add_baseline_line: bool = True,
    **kwargs
) -> plt.Figure:
    """Create a comparison figure showing ratios relative to a baseline method.

    Similar to create_method_comparison_figure but plots ratios instead of raw values.

    Args:
        data_by_config: Nested dict {config_key: {method_name: y_values}}
        x_arrays_by_config: Dict {config_key: x_values}
        baseline_method: Name of baseline method for ratio calculation
        config_titles: Dict {config_key: subplot_title}
        xlabel: Label for x-axis
        ylabel: Label for y-axis (e.g., 'σ ratio')
        add_baseline_line: Whether to add horizontal line at y=1
        **kwargs: Additional arguments passed to create_method_comparison_figure

    Returns:
        The created matplotlib figure
    """
    # Calculate ratios
    ratio_data = {}
    for config_key, method_dict in data_by_config.items():
        if baseline_method not in method_dict:
            raise ValueError(f"Baseline method '{baseline_method}' not found in config {config_key}")

        baseline_values = method_dict[baseline_method]
        ratio_data[config_key] = {}

        for method_name, values in method_dict.items():
            if method_name == baseline_method:
                continue  # Skip baseline itself
            ratio_data[config_key][method_name] = values / baseline_values

    # Create figure using standard comparison
    fig = create_method_comparison_figure(
        ratio_data,
        x_arrays_by_config,
        config_titles,
        xlabel=xlabel,
        ylabel=ylabel,
        **kwargs
    )

    # Add baseline line at y=1 if requested
    if add_baseline_line:
        for ax in fig.axes:
            if ax.get_visible():
                ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2,
                          label=baseline_method, zorder=1)

    return fig

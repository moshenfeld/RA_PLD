"""
Visualization definitions for experiment plots.

This module provides consistent styling, resolution settings, and method configurations
across all experiment plots.
"""

# ====================================================================
# Resolution settings
# ====================================================================

# Save resolution for plots (600 DPI for high-quality plots)
SAVEFIG_DPI = 600


# ====================================================================
# Method styling definitions
# ====================================================================

# Color scheme for methods
method_colors = {
    'Poisson (PLD)':              'r',
    'Allocation (DCO25)':         'c',
    'Allocation (FS25)':          'k',
    'Allocation (PLD)':           'b',
    'Allocation (PLD, upper)':    'b',
    'Allocation (PLD, lower)':    'y',
    'Allocation (MC, mean)':      'g',
    'Allocation (MC, high prob)': 'g',
    'Allocation (lower bound)':   'y',
}

# Line style scheme for methods
method_linestyles = {
    'Poisson (PLD)':              ':',
    'Allocation (DCO25)':         '--',
    'Allocation (FS25)':          '-.',
    'Allocation (PLD)':           '-',
    'Allocation (PLD, upper)':    '-',
    'Allocation (PLD, lower)':    '-',
    'Allocation (MC, mean)':      ':',
    'Allocation (MC, high prob)': ':',
    'Allocation (lower bound)':   '-.',
}

# Marker scheme for methods
# Each method gets a unique marker to avoid confusion when multiple methods appear in the same plot
method_markers = {
    'Poisson (PLD)':              'o',
    'Allocation (DCO25)':         's',
    'Allocation (FS25)':          'D',
    'Allocation (PLD)':           '^',
    'Allocation (PLD, upper)':    '^',
    'Allocation (PLD, lower)':    'v',
    'Allocation (MC, mean)':      '*',
    'Allocation (MC, high prob)': 'P',
    'Allocation (lower bound)':   'x',
}

# Drawing order for methods (lower index = drawn first = appears behind)
# Methods not in this list will be drawn last
method_drawing_order = [
    'Allocation (lower bound)',
    'Poisson (PLD)',
    'Allocation (DCO25)',
    'Allocation (FS25)',
    'Allocation (PLD)',
    'Allocation (PLD, lower)',
    'Allocation (PLD, upper)',
    'Allocation (MC, mean)',
    'Allocation (MC, high prob)',
]

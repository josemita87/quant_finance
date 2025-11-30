"""
Configuration module for corporate styling and global plot settings.
Defines color palettes, fonts, and consistent visual identity.
"""

import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class PlotConfig:
    """
    Centralized configuration for all visualization styling.
    Ensures consistency across all generated plots.
    """

    # Professional Orange-Yellowish Color Palette
    COLORS = {
        'brand_primary': '#D97706',    # Deep amber/orange - primary
        'brand_secondary': '#F59E0B',  # Bright amber - emphasis/arrows
        'accent_blue': '#F97316',      # Medium orange - accents
        'light_blue': '#FB923C',       # Light orange - highlights
        'sky_blue': '#FCD34D',         # Yellow - soft accents
        'dark_navy': '#78350F',        # Dark brown/amber - text/borders
        'steel_blue': '#92400E',       # Medium brown - secondary text
        'slate': '#78716C',            # Warm gray - neutral elements
        'light_slate': '#A8A29E',      # Light warm gray - subtle elements
        'white': '#FFFFFF',            # White - for contrast
        'transparent': 'none'          # Transparent backgrounds
    }

    # Extended orange-yellowish palette for multi-category plots
    PALETTE = [
        '#D97706',  # Deep amber
        '#F59E0B',  # Bright amber
        '#F97316',  # Medium orange
        '#FB923C',  # Light orange
        '#FCD34D',  # Yellow
        '#FDE047',  # Bright yellow
        '#EAB308',  # Golden yellow
        '#CA8A04',  # Dark amber
    ]

    # Chart-specific color schemes
    DIVERGING_PALETTE = ['#DC2626', '#95A5A6', '#16A34A']  # Negative, neutral, positive
    SEQUENTIAL_PALETTE = ['#FEF3C7', '#FCD34D', '#92400E']  # Light yellow to dark amber

    # Typography - Professional and Simplistic
    FONTS = {
        'title': {
            'family': 'sans-serif',
            'size': 16,
            'weight': 'bold',
            'color': COLORS['dark_navy']
        },
        'subtitle': {
            'family': 'sans-serif',
            'size': 13,
            'weight': 'normal',
            'color': COLORS['steel_blue']
        },
        'axis_label': {
            'family': 'sans-serif',
            'size': 11,
            'weight': '600',
            'color': COLORS['dark_navy']
        },
        'tick_label': {
            'family': 'sans-serif',
            'size': 9,
            'weight': 'normal',
            'color': COLORS['slate']
        },
        'annotation': {
            'family': 'sans-serif',
            'size': 9,
            'weight': 'normal',
            'color': COLORS['steel_blue']
        }
    }

    # Figure settings
    DPI = 300  # High resolution for publication
    FIGURE_SIZE = (12, 7)  # Default figure size (width, height) in inches

    @classmethod
    def apply_global_style(cls):
        """Apply professional, simplistic orange-yellowish styling with transparent backgrounds."""
        plt.style.use('seaborn-v0_8-whitegrid')

        plt.rcParams.update({
            # Figure - Transparent background
            'figure.figsize': cls.FIGURE_SIZE,
            'figure.dpi': cls.DPI,
            'figure.facecolor': 'none',  # Transparent
            'figure.edgecolor': 'none',

            # Font - Professional and clean
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 10,

            # Axes - Transparent with subtle borders
            'axes.facecolor': 'none',  # Transparent
            'axes.edgecolor': cls.COLORS['light_slate'],
            'axes.linewidth': 1.0,
            'axes.labelsize': cls.FONTS['axis_label']['size'],
            'axes.labelweight': cls.FONTS['axis_label']['weight'],
            'axes.labelcolor': cls.FONTS['axis_label']['color'],
            'axes.titlesize': cls.FONTS['title']['size'],
            'axes.titleweight': cls.FONTS['title']['weight'],
            'axes.titlecolor': cls.FONTS['title']['color'],
            'axes.titlepad': 15,
            'axes.grid': True,
            'axes.axisbelow': True,
            'axes.spines.top': False,     # Remove top spine
            'axes.spines.right': False,    # Remove right spine

            # Grid - Very subtle
            'grid.color': cls.COLORS['light_slate'],
            'grid.linestyle': ':',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.3,

            # Ticks - Clean and minimal
            'xtick.labelsize': cls.FONTS['tick_label']['size'],
            'ytick.labelsize': cls.FONTS['tick_label']['size'],
            'xtick.color': cls.COLORS['slate'],
            'ytick.color': cls.COLORS['slate'],
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,

            # Legend - Minimal and clean
            'legend.frameon': False,
            'legend.fontsize': 9,
            'legend.loc': 'best',
            'legend.fancybox': False,

            # Lines and markers - Professional weight
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            'lines.solid_capstyle': 'round',

            # Saving - Transparent background
            'savefig.dpi': cls.DPI,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'none',  # Transparent
            'savefig.edgecolor': 'none',
            'savefig.pad_inches': 0.1,
            'savefig.transparent': True    # Ensure transparency
        })

    @classmethod
    def get_color(cls, key: str) -> str:
        """Get a color from the palette by key."""
        return cls.COLORS.get(key, cls.COLORS['brand_primary'])

    @classmethod
    def get_palette(cls, n_colors: int = None) -> List[str]:
        """
        Get a color palette.

        Args:
            n_colors: Number of colors needed. If None, returns full palette.

        Returns:
            List of color hex codes.
        """
        if n_colors is None:
            return cls.PALETTE

        if n_colors <= len(cls.PALETTE):
            return cls.PALETTE[:n_colors]
        else:
            # Cycle through palette if more colors needed
            return [cls.PALETTE[i % len(cls.PALETTE)] for i in range(n_colors)]

    @classmethod
    def format_percentage(cls, value: float, decimals: int = 1) -> str:
        """Format a value as percentage."""
        return f"{value:.{decimals}f}%"

    @classmethod
    def format_currency(cls, value: float, currency: str = '€', decimals: int = 1) -> str:
        """Format a value as currency."""
        return f"{currency}{value:.{decimals}f}M"

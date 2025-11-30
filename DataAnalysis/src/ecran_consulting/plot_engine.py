"""
Reusable plotting engine for creating publication-ready strategic visualizations.
All functions follow consistent styling and return figure objects for flexibility.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from math import pi
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from .config import PlotConfig
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier, plot_tree



class PlotEngine:
    """
    High-level plotting engine for strategic consulting visualizations.
    Provides reusable methods for common chart types with corporate styling.
    """

    def __init__(self, output_dir: str = 'output'):
        """
        Initialize the plotting engine.

        Args:
            output_dir: Directory to save generated plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        PlotConfig.apply_global_style()

    def save_plot(self, filename: str, fig: Optional[plt.Figure] = None):
        """
        Save a plot to the output directory.

        Args:
            filename: Name of the file (with or without extension).
            fig: Figure object. If None, saves current figure.
        """
        if not filename.endswith('.png'):
            filename += '.png'

        filepath = self.output_dir / filename

        if fig is not None:
            fig.savefig(filepath, dpi=PlotConfig.DPI, bbox_inches='tight')
        else:
            plt.savefig(filepath, dpi=PlotConfig.DPI, bbox_inches='tight')

        print(f"✓ Saved: {filepath}")

    def bar_chart(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        xlabel: str = None,
        ylabel: str = None,
        colors: Optional[List[str]] = None,
        horizontal: bool = False,
        show_values: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a styled bar chart.

        Args:
            data: DataFrame containing the data.
            x_col: Column name for x-axis (categories).
            y_col: Column name for y-axis (values).
            title: Chart title.
            xlabel: X-axis label (defaults to x_col).
            ylabel: Y-axis label (defaults to y_col).
            colors: List of colors for bars. If None, uses default palette.
            horizontal: If True, creates horizontal bars.
            show_values: If True, displays values on bars.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=PlotConfig.FIGURE_SIZE)

        x_data = data[x_col]
        y_data = data[y_col]

        if colors is None:
            colors = PlotConfig.get_palette(len(data))

        if horizontal:
            bars = ax.barh(x_data, y_data, color=colors, edgecolor='none', alpha=0.85)
            ax.set_xlabel(ylabel or y_col)
            ax.set_ylabel(xlabel or x_col)
        else:
            bars = ax.bar(x_data, y_data, color=colors, edgecolor='none', alpha=0.85)
            ax.set_xlabel(xlabel or x_col)
            ax.set_ylabel(ylabel or y_col)
            plt.xticks(rotation=45, ha='right')

        ax.set_title(title, pad=20, **PlotConfig.FONTS['title'])

        if show_values:
            for bar in bars:
                if horizontal:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2,
                           f'{width:.1f}', ha='left', va='center',
                           fontsize=PlotConfig.FONTS['annotation']['size'],
                           fontweight='bold')
                else:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           f'{height:.1f}', ha='center', va='bottom',
                           fontsize=PlotConfig.FONTS['annotation']['size'],
                           fontweight='bold')

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def grouped_bar_chart(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str,
        xlabel: str = None,
        ylabel: str = None,
        labels: Optional[List[str]] = None,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a grouped bar chart for comparing multiple metrics.

        Args:
            data: DataFrame containing the data.
            x_col: Column name for x-axis (categories).
            y_cols: List of column names to compare.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            labels: Custom labels for legend. If None, uses y_cols.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=PlotConfig.FIGURE_SIZE)

        x_data = data[x_col]
        x_pos = np.arange(len(x_data))
        width = 0.8 / len(y_cols)

        colors = PlotConfig.get_palette(len(y_cols))
        labels = labels or y_cols

        for i, (col, label, color) in enumerate(zip(y_cols, labels, colors)):
            offset = width * (i - len(y_cols)/2 + 0.5)
            ax.bar(x_pos + offset, data[col], width, label=label,
                   color=color, edgecolor='none', alpha=0.85)

        ax.set_xlabel(xlabel or x_col, **PlotConfig.FONTS['axis_label'])
        ax.set_ylabel(ylabel or 'Value', **PlotConfig.FONTS['axis_label'])
        ax.set_title(title, pad=20, **PlotConfig.FONTS['title'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_data, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def line_chart(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str,
        xlabel: str = None,
        ylabel: str = None,
        labels: Optional[List[str]] = None,
        markers: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a line chart for trend visualization.

        Args:
            data: DataFrame containing the data.
            x_col: Column name for x-axis.
            y_cols: List of column names to plot.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            labels: Custom labels for legend.
            markers: If True, shows markers on data points.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=PlotConfig.FIGURE_SIZE)

        colors = PlotConfig.get_palette(len(y_cols))
        labels = labels or y_cols
        marker_style = 'o' if markers else ''

        for col, label, color in zip(y_cols, labels, colors):
            ax.plot(data[x_col], data[col], marker=marker_style,
                   label=label, color=color, linewidth=2.0, markersize=6, alpha=0.9)

        ax.set_xlabel(xlabel or x_col, **PlotConfig.FONTS['axis_label'])
        ax.set_ylabel(ylabel or 'Value', **PlotConfig.FONTS['axis_label'])
        ax.set_title(title, pad=20, **PlotConfig.FONTS['title'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def scatter_plot(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        size_col: Optional[str] = None,
        label_col: Optional[str] = None,
        title: str = '',
        xlabel: str = None,
        ylabel: str = None,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a scatter plot with optional sizing and labels.

        Args:
            data: DataFrame containing the data.
            x_col: Column name for x-axis.
            y_col: Column name for y-axis.
            size_col: Column name for bubble sizes (optional).
            label_col: Column name for point labels (optional).
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=PlotConfig.FIGURE_SIZE)

        sizes = data[size_col] * 10 if size_col else 100

        scatter = ax.scatter(
            data[x_col], data[y_col],
            s=sizes, alpha=0.7,
            c=PlotConfig.get_palette(len(data)),
            edgecolors=PlotConfig.COLORS['white'], linewidth=1.5
        )

        if label_col:
            for idx, row in data.iterrows():
                ax.annotate(
                    row[label_col],
                    (row[x_col], row[y_col]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=PlotConfig.FONTS['annotation']['size'],
                    fontweight='bold'
                )

        ax.set_xlabel(xlabel or x_col, **PlotConfig.FONTS['axis_label'])
        ax.set_ylabel(ylabel or y_col, **PlotConfig.FONTS['axis_label'])
        ax.set_title(title, pad=20, **PlotConfig.FONTS['title'])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def pie_chart(
        self,
        data: pd.DataFrame,
        values_col: str,
        labels_col: str,
        title: str,
        show_percentages: bool = True,
        explode_max: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a styled pie chart.

        Args:
            data: DataFrame containing the data.
            values_col: Column name for values.
            labels_col: Column name for labels.
            title: Chart title.
            show_percentages: If True, shows percentages on slices.
            explode_max: If True, explodes the largest slice.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        values = data[values_col]
        labels = data[labels_col]
        colors = PlotConfig.get_palette(len(data))

        explode = None
        if explode_max:
            explode = [0.1 if v == values.max() else 0 for v in values]

        autopct = '%1.1f%%' if show_percentages else None

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors,
            autopct=autopct, startangle=90,
            explode=explode,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'alpha': 0.85}
        )

        for text in texts:
            text.set_fontsize(PlotConfig.FONTS['tick_label']['size'])
            text.set_fontweight('bold')

        if autotexts:
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(PlotConfig.FONTS['annotation']['size'])
                autotext.set_fontweight('bold')

        ax.set_title(title, pad=20, **PlotConfig.FONTS['title'])

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def heatmap(
        self,
        data: pd.DataFrame,
        title: str,
        xlabel: str = '',
        ylabel: str = '',
        cmap: str = 'YlOrRd',
        annot: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a heatmap visualization.

        Args:
            data: DataFrame to visualize.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            cmap: Colormap name.
            annot: If True, annotates cells with values.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=PlotConfig.FIGURE_SIZE)

        im = ax.imshow(data.values, cmap=cmap, aspect='auto')

        # Set ticks
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.index)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=PlotConfig.FONTS['tick_label']['size'])

        # Annotate cells
        if annot:
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    text = ax.text(j, i, f'{data.iloc[i, j]:.1f}',
                                 ha='center', va='center',
                                 color='white' if data.iloc[i, j] > data.values.mean() else 'black',
                                 fontsize=PlotConfig.FONTS['annotation']['size'])

        ax.set_xlabel(xlabel, **PlotConfig.FONTS['axis_label'])
        ax.set_ylabel(ylabel, **PlotConfig.FONTS['axis_label'])
        ax.set_title(title, pad=20, **PlotConfig.FONTS['title'])

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    # ========== ECRAN-SPECIFIC STRATEGIC VISUALIZATIONS ==========

    def calendar_heatmap(
        self,
        monthly_data: List[float],
        months_labels: List[str],
        title: str = "THE 'BLUE OCEAN': 250 DAYS OF UNTAPPED OPPORTUNITY",
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a calendar-style heatmap showing seasonality patterns.
        Visualizes the winter opportunity with a granular weekly view.

        Args:
            monthly_data: List of 12 monthly sales values.
            months_labels: List of 12 month names.
            title: Chart title.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        # Interpolate 12 months to ~52 weeks for granular visualization
        x_months = np.linspace(0, 12, 12)
        x_weeks = np.linspace(0, 12, 52)
        weekly_intensity = np.interp(x_weeks, x_months, monthly_data)

        # Create grid (7 days x 52 weeks) with organic variation
        data_grid = np.zeros((7, 52))
        for i, intensity in enumerate(weekly_intensity):
            # Use flat intensity for the week (no synthetic variation)
            daily_val = np.full(7, intensity)
            data_grid[:, i] = daily_val

        fig, ax = plt.subplots(figsize=(14, 4))

        # Create heatmap with professional orange-yellowish colormap
        sns.heatmap(
            data_grid,
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor='white',
            cbar=False,
            ax=ax,
            alpha=0.9
        )

        ax.set_title(title, loc='left', fontweight='bold', fontsize=16, pad=15)
        ax.set_yticks([])
        ax.set_ylabel('')
        ax.set_xticks(np.linspace(2, 50, 12))
        ax.set_xticklabels(months_labels)
        ax.set_xlabel('Month', fontweight='bold')

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def comparison_radar(
        self,
        categories: List[str],
        profile_a: List[float],
        profile_b: List[float],
        label_a: str,
        label_b: str,
        title: str = "THE AGGRESSION SHIFT: WHY WINTER NEEDS PROTECTION",
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a radar chart comparing two profiles (e.g., summer vs. winter stressors).

        Args:
            categories: List of category names.
            profile_a: Values for first profile.
            profile_b: Values for second profile.
            label_a: Label for first profile.
            label_b: Label for second profile.
            title: Chart title.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        prof_a = profile_a + profile_a[:1]
        prof_b = profile_b + profile_b[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

        # Plot Profile A (e.g., Summer)
        ax.plot(
            angles, prof_a,
            linewidth=2,
            linestyle='--',
            color=PlotConfig.COLORS['light_blue'],
            label=label_a,
            alpha=0.8
        )
        ax.fill(angles, prof_a, color=PlotConfig.COLORS['light_blue'], alpha=0.15)

        # Plot Profile B (e.g., Winter)
        ax.plot(
            angles, prof_b,
            linewidth=2.5,
            linestyle='-',
            color=PlotConfig.COLORS['brand_primary'],
            label=label_b,
            alpha=0.9
        )
        ax.fill(angles, prof_b, color=PlotConfig.COLORS['brand_primary'], alpha=0.25)

        # Styling - Professional and clean
        plt.xticks(angles[:-1], categories, size=10, color=PlotConfig.COLORS['dark_navy'])
        ax.set_rlabel_position(0)
        plt.yticks([2, 5, 8], ["Low", "Med", "High"], color=PlotConfig.COLORS['slate'], size=8)
        plt.ylim(0, 10)
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=False, fontsize=9)
        plt.title(title, y=1.08, fontweight='bold', fontsize=14, color=PlotConfig.COLORS['dark_navy'])

        # Set transparent background
        ax.patch.set_alpha(0)
        fig.patch.set_alpha(0)

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def strategic_positioning_map(
        self,
        brand_data: Dict[str, Dict],
        title: str = "STRATEGIC POSITIONING: THE MIGRATION TO WINNING ZONE",
        xlabel: str = "COSMETIC APPEAL (Texture/Scent/Elegance)",
        ylabel: str = "CLINICAL AUTHORITY (Protection/Science)",
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a strategic positioning map with brand locations and migration arrow.

        Args:
            brand_data: Dict of {brand_name: {coords: (x, y), type: str}}
                       type can be 'competitor', 'current', or 'target'
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        current_pos = None
        target_pos = None

        # Plot brands by type
        for name, data in brand_data.items():
            x, y = data['coords']
            brand_type = data['type']

            if brand_type == 'competitor':
                ax.scatter(
                    x, y,
                    color=PlotConfig.COLORS['slate'],
                    s=120,
                    alpha=0.5,
                    zorder=2
                )
                ax.text(
                    x, y + 0.3, name,
                    ha='center',
                    color=PlotConfig.COLORS['slate'],
                    fontsize=9
                )

            elif brand_type == 'current':
                ax.scatter(
                    x, y,
                    color=PlotConfig.COLORS['accent_blue'],
                    s=250,
                    marker='D',
                    zorder=5,
                    edgecolors='white',
                    linewidth=1.5,
                    alpha=0.8
                )
                ax.text(
                    x, y - 0.5, name,
                    ha='center',
                    fontweight='bold',
                    color=PlotConfig.COLORS['brand_primary'],
                    fontsize=10
                )
                current_pos = (x, y)

            elif brand_type == 'target':
                ax.scatter(
                    x, y,
                    color=PlotConfig.COLORS['brand_primary'],
                    s=250,
                    marker='*',
                    zorder=5,
                    edgecolors='white',
                    linewidth=1.5,
                    alpha=0.9
                )
                ax.text(
                    x, y + 0.4, name,
                    ha='center',
                    fontweight='bold',
                    color=PlotConfig.COLORS['brand_primary'],
                    fontsize=10
                )
                target_pos = (x, y)

        # Draw strategic migration arrow
        if current_pos and target_pos:
            ax.annotate(
                '',
                xy=target_pos,
                xytext=current_pos,
                arrowprops=dict(
                    facecolor=PlotConfig.COLORS['brand_secondary'],
                    edgecolor=PlotConfig.COLORS['brand_secondary'],
                    shrink=0.05,
                    width=1.5,
                    headwidth=8,
                    headlength=8,
                    alpha=0.7
                ),
                zorder=4
            )

        # Add "Winning Zone" annotation
        ax.text(
            9, 9,
            "WINNING ZONE",
            ha='right',
            va='top',
            fontsize=10,
            fontweight='600',
            color=PlotConfig.COLORS['brand_primary'],
            alpha=0.4,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='none',
                     edgecolor=PlotConfig.COLORS['brand_primary'],
                     alpha=0.3, linewidth=1)
        )

        # Axis styling - Clean and professional
        ax.set_xlabel(xlabel, fontweight='600', fontsize=11, color=PlotConfig.COLORS['dark_navy'])
        ax.set_ylabel(ylabel, fontweight='600', fontsize=11, color=PlotConfig.COLORS['dark_navy'])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title(title, pad=15, fontweight='bold', fontsize=14, color=PlotConfig.COLORS['dark_navy'])
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

        # Add subtle quadrant lines
        ax.axhline(y=5, color=PlotConfig.COLORS['light_slate'], linestyle=':', linewidth=0.8, alpha=0.3)
        ax.axvline(x=5, color=PlotConfig.COLORS['light_slate'], linestyle=':', linewidth=0.8, alpha=0.3)

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    # ========== PHASE 2: ML & INSIGHTS VISUALIZATIONS ==========

    def plot_cluster_tribes(
        self,
        df: pd.DataFrame,
        smart_ager_cluster: int,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize customer clusters in 2D focusing on the Smart Ager segment.
        Uses texture_sensitivity and concern_uvb as the primary features.

        Args:
            df: DataFrame with cluster assignments.
            smart_ager_cluster: ID of the Smart Ager cluster.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        # Set up the figure with a larger size for better visibility
        fig, ax = plt.subplots(figsize=(14, 10))

        # Prepare data
        n_clusters = df['cluster_id'].nunique()
        colors = []
        for i in range(n_clusters):
            if i == smart_ager_cluster:
                colors.append(PlotConfig.COLORS['brand_primary'])  # Deep amber for target
            else:
                colors.append(PlotConfig.COLORS['slate'])  # Gray for others

        # Plot each cluster in 2D
        for i in range(n_clusters):
            cluster_data = df[df['cluster_id'] == i]

            # Adjust visual properties for better distinction
            alpha = 0.9 if i == smart_ager_cluster else 0.6
            size = 120 if i == smart_ager_cluster else 60
            edge_color = 'white' if i == smart_ager_cluster else 'none'
            line_width = 1.5 if i == smart_ager_cluster else 0.5

            ax.scatter(
                x=cluster_data['texture_sensitivity'],
                y=cluster_data['concern_uvb'],
                c=colors[i],
                alpha=alpha,
                s=size,
                edgecolors=edge_color,
                linewidth=line_width,
                label=f'Cluster {i}' + (' (Smart Ager)' if i == smart_ager_cluster else ''),
                zorder=3  # Ensure points are above grid lines
            )

        # Add annotations for Smart Ager cluster
        smart_ager_data = df[df['cluster_id'] == smart_ager_cluster]
        center_x = smart_ager_data['texture_sensitivity'].mean()
        center_y = smart_ager_data['concern_uvb'].mean()
        buy_rate = smart_ager_data['willingness_to_buy_winter'].mean()

        # Add text annotation
        ax.annotate(
            f'SMART AGERS\n{len(smart_ager_data)} consumers\n{buy_rate*100:.0f}% buy rate',
            xy=(center_x, center_y),
            xytext=(center_x + 1.5, center_y + 1.5),
            fontsize=10,
            fontweight='bold',
            color=PlotConfig.COLORS['brand_primary'],
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor=PlotConfig.COLORS['brand_primary'],
                alpha=0.9,
                linewidth=2
            ),
            arrowprops=dict(
                arrowstyle='->',
                color=PlotConfig.COLORS['brand_primary'],
                linewidth=1.5,
                connectionstyle='arc3,rad=0.2'
            )
        )

        # Styling
        ax.set_xlabel(
            'Texture Sensitivity\n(Cosmetic Elegance Demand)',
            fontweight='600',
            fontsize=12,
            color=PlotConfig.COLORS['dark_navy'],
            labelpad=10
        )
        ax.set_ylabel(
            'UVB Concern Score\n(Burning Risk)',
            fontweight='600',
            fontsize=12,
            color=PlotConfig.COLORS['dark_navy'],
            labelpad=10
        )

        # Set axis limits with some padding
        ax.set_xlim(
            df['texture_sensitivity'].min() - 0.5,
            df['texture_sensitivity'].max() + 0.5
        )
        ax.set_ylim(
            df['concern_uvb'].min() - 0.5,
            df['concern_uvb'].max() + 0.5
        )

        # Add grid for better readability
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=0)

        # Add title and legend
        ax.set_title(
            "IDENTIFYING THE 'SMART AGERS': 2D CLUSTER VISUALIZATION",
            pad=20,
            fontweight='bold',
            fontsize=14,
            color=PlotConfig.COLORS['dark_navy']
        )

        # Position legend outside the plot area
        legend = ax.legend(
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            frameon=True,
            framealpha=0.9,
            edgecolor='#f0f0f0',
            fontsize=10
        )

        # Add a subtle border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#dddddd')
            spine.set_linewidth(1)

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def plot_purchase_drivers(
        self,
        coef_df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize logistic regression coefficients as purchase drivers.

        Args:
            coef_df: DataFrame with features and coefficients.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Sort by coefficient value
        coef_df_sorted = coef_df.sort_values('Coefficient')

        # Create color gradient based on coefficient magnitude
        # Higher values get darker amber, lower values get lighter yellow
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # Normalize coefficients for color mapping
        norm = mcolors.Normalize(
            vmin=coef_df_sorted['Coefficient'].min(),
            vmax=coef_df_sorted['Coefficient'].max()
        )

        # Create color gradient from light yellow to dark amber
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'orange_gradient',
            [
                PlotConfig.COLORS['sky_blue'],      # Lightest (yellow)
                PlotConfig.COLORS['light_blue'],    # Light (light orange)
                PlotConfig.COLORS['accent_blue'],   # Medium (orange)
                PlotConfig.COLORS['brand_primary'], # Dark (amber)
                PlotConfig.COLORS['dark_navy']      # Darkest (dark brown)
            ]
        )

        colors = [cmap(norm(c)) for c in coef_df_sorted['Coefficient']]

        # Create horizontal bar chart
        bars = ax.barh(
            coef_df_sorted['Feature'],
            coef_df_sorted['Coefficient'],
            color=colors,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5
        )

        # Add value labels
        for i, (idx, row) in enumerate(coef_df_sorted.iterrows()):
            value = row['Coefficient']
            x_pos = value + (0.02 if value > 0 else -0.02)
            ha = 'left' if value > 0 else 'right'

            ax.text(
                x_pos,
                i,
                f'{value:.3f}',
                va='center',
                ha=ha,
                fontsize=9,
                fontweight='bold',
                color=PlotConfig.COLORS['dark_navy']
            )

        # Add zero line
        ax.axvline(
            x=0,
            color=PlotConfig.COLORS['dark_navy'],
            linestyle='-',
            linewidth=1.5,
            alpha=0.5
        )

        # Styling
        ax.set_xlabel(
            'Coefficient (Impact on Winter Purchase)',
            fontweight='600',
            fontsize=11,
            color=PlotConfig.COLORS['dark_navy']
        )
        ax.set_title(
            "WHAT DRIVES WINTER PURCHASE? (BLUE LIGHT & TEXTURE)",
            pad=15,
            fontweight='bold',
            fontsize=14,
            color=PlotConfig.COLORS['dark_navy']
        )

        ax.grid(True, axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def plot_decision_tree_rules(
        self,
        tree_model: DecisionTreeClassifier,
        feature_names: list,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize decision tree rules showing the golden path.

        Args:
            tree_model: Fitted DecisionTreeClassifier.
            feature_names: List of feature names.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot decision tree
        plot_tree(
            tree_model,
            feature_names=feature_names,
            class_names=['No Buy', 'Buy Winter'],
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax,
            impurity=False,
            proportion=True
        )

        # Customize colors - override default colors with orange-yellowish tones
        for artist in ax.get_children():
            if hasattr(artist, 'get_facecolor'):
                # Get current color
                try:
                    fc = artist.get_facecolor()
                    # If it's an array of RGBA values
                    if isinstance(fc, np.ndarray) and len(fc) == 4:
                        # Replace default colors with our orange-yellowish tones
                        if fc[0] < fc[2]:  # More blue than red (buy class)
                            artist.set_facecolor(PlotConfig.COLORS['brand_primary'])
                            artist.set_alpha(0.7)
                        elif fc[0] > fc[2]:  # More red than blue (no buy class)
                            artist.set_facecolor(PlotConfig.COLORS['light_blue'])
                            artist.set_alpha(0.3)
                except:
                    pass

        # Title
        ax.set_title(
            "THE GOLDEN PATH: TARGETING RULES FOR MAX CONVERSION",
            pad=20,
            fontweight='bold',
            fontsize=14,
            color=PlotConfig.COLORS['dark_navy']
        )

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance for Winter Purchase",
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize feature importance from tree model.

        Args:
            importance_df: DataFrame with features and importance scores.
            title: Chart title.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Reset index to ensure clean ordering, then filter out zero importance features
        importance_df_clean = importance_df.reset_index(drop=True).copy()
        # Filter out features with importance = 0 (like 'age')
        importance_df_filtered = importance_df_clean[importance_df_clean['Importance'] > 0].copy()
        # Sort by importance (ascending for horizontal bar)
        importance_df_sorted = importance_df_filtered.sort_values('Importance', ascending=True).copy()

        # Check if we have data to plot
        if len(importance_df_sorted) == 0:
            ax.text(0.5, 0.5, 'No features to display',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color=PlotConfig.COLORS['dark_navy'])
            ax.set_title(title, pad=15, fontweight='bold', fontsize=14,
                        color=PlotConfig.COLORS['dark_navy'])
            if filename:
                self.save_plot(filename, fig)
            return fig

        # All features now have importance > 0
        all_features = importance_df_sorted.copy()

        # Create colors for features (orange/red gradient)
        n_features = len(all_features)
        if n_features > 1:
            # Color gradient for features
            all_colors = plt.cm.YlOrRd(
                np.linspace(0.5, 0.95, n_features)
            )
        else:
            # Single feature - use primary color
            all_colors = [PlotConfig.COLORS['brand_primary']]

        # Horizontal bar chart
        bars = ax.barh(
            all_features['Feature'],
            all_features['Importance'],
            color=all_colors,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5
        )

        # Add value labels for all features
        for i, (idx, row) in enumerate(all_features.iterrows()):
            ax.text(
                row['Importance'] + 0.01,
                i,
                f"{row['Importance']:.3f}",
                va='center',
                fontsize=9,
                fontweight='bold',
                color=PlotConfig.COLORS['dark_navy']
            )

        # Styling
        ax.set_xlabel(
            'Importance Score',
            fontweight='600',
            fontsize=11,
            color=PlotConfig.COLORS['dark_navy']
        )
        ax.set_title(
            title,
            pad=15,
            fontweight='bold',
            fontsize=14,
            color=PlotConfig.COLORS['dark_navy']
        )

        ax.grid(True, axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def plot_regional_profiles(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize customer profiles by Spanish region.

        Args:
            df: DataFrame with regional data.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Filter to top 6 regions (exclude 'Other')
        top_regions = df[df['region'] != 'Other']['region'].value_counts().head(6).index.tolist()
        df_regions = df[df['region'].isin(top_regions)]

        # 1. Winter Purchase Intent by Region
        ax1 = axes[0, 0]
        regional_intent = df_regions.groupby('region')['willingness_to_buy_winter'].mean().sort_values(ascending=True)
        bars = ax1.barh(regional_intent.index, regional_intent.values * 100,
                       color=PlotConfig.COLORS['brand_primary'], alpha=0.8, edgecolor='none')
        ax1.set_xlabel('Winter Purchase Intent (%)', fontweight='600', fontsize=10)
        ax1.set_title('Winter Purchase Intent by Region', fontweight='bold', fontsize=12,
                     color=PlotConfig.COLORS['dark_navy'], pad=10)
        ax1.grid(True, axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

        # Add value labels
        for i, (idx, val) in enumerate(regional_intent.items()):
            ax1.text(val * 100 + 1, i, f'{val*100:.1f}%', va='center', fontsize=9,
                    fontweight='bold', color=PlotConfig.COLORS['dark_navy'])

        # 2. Average Income by Region
        ax2 = axes[0, 1]
        regional_income = df_regions.groupby('region')['income'].mean().sort_values(ascending=True) / 1000
        bars = ax2.barh(regional_income.index, regional_income.values,
                       color=PlotConfig.COLORS['accent_blue'], alpha=0.8, edgecolor='none')
        ax2.set_xlabel('Average Income (€K)', fontweight='600', fontsize=10)
        ax2.set_title('Average Income by Region', fontweight='bold', fontsize=12,
                     color=PlotConfig.COLORS['dark_navy'], pad=10)
        ax2.grid(True, axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

        # Add value labels
        for i, (idx, val) in enumerate(regional_income.items()):
            ax2.text(val + 1, i, f'€{val:.0f}K', va='center', fontsize=9,
                    fontweight='bold', color=PlotConfig.COLORS['dark_navy'])

        # 3. Concern Profiles by Region (Radar-style)
        ax3 = axes[1, 0]
        concerns = ['concern_uva', 'concern_blue_light', 'concern_pollution']
        concern_labels = ['UVA\nAging', 'Blue Light\nDigital', 'Pollution\nEnvironment']

        # Get top 3 regions by population
        top_3_regions = df_regions['region'].value_counts().head(3).index.tolist()

        x_pos = np.arange(len(concerns))
        width = 0.25
        colors_reg = [PlotConfig.COLORS['brand_primary'],
                     PlotConfig.COLORS['accent_blue'],
                     PlotConfig.COLORS['light_blue']]

        for i, region in enumerate(top_3_regions):
            region_data = df_regions[df_regions['region'] == region]
            concern_means = [region_data[c].mean() for c in concerns]
            ax3.bar(x_pos + i * width, concern_means, width,
                   label=region, color=colors_reg[i], alpha=0.8, edgecolor='none')

        ax3.set_ylabel('Concern Level (1-10)', fontweight='600', fontsize=10)
        ax3.set_title('Concern Profiles: Top 3 Regions', fontweight='bold', fontsize=12,
                     color=PlotConfig.COLORS['dark_navy'], pad=10)
        ax3.set_xticks(x_pos + width)
        ax3.set_xticklabels(concern_labels, fontsize=9)
        ax3.legend(loc='upper left', frameon=False, fontsize=9)
        ax3.grid(True, axis='y', alpha=0.2, linestyle=':', linewidth=0.5)
        ax3.set_ylim(0, 10)

        # 4. Willingness to Pay by Region
        ax4 = axes[1, 1]
        regional_wtp = df_regions.groupby('region')['willingness_to_pay'].mean().sort_values(ascending=True)
        bars = ax4.barh(regional_wtp.index, regional_wtp.values,
                       color=PlotConfig.COLORS['brand_secondary'], alpha=0.8, edgecolor='none')
        ax4.set_xlabel('Average Willingness to Pay (€)', fontweight='600', fontsize=10)
        ax4.set_title('Willingness to Pay by Region', fontweight='bold', fontsize=12,
                     color=PlotConfig.COLORS['dark_navy'], pad=10)
        ax4.grid(True, axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

        # Add value labels
        for i, (idx, val) in enumerate(regional_wtp.items()):
            ax4.text(val + 0.5, i, f'€{val:.2f}', va='center', fontsize=9,
                    fontweight='bold', color=PlotConfig.COLORS['dark_navy'])

        # Overall title
        fig.suptitle('REGIONAL CUSTOMER PROFILES: SPAIN',
                    fontsize=16, fontweight='bold',
                    color=PlotConfig.COLORS['dark_navy'], y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            self.save_plot(filename, fig)

        return fig

    def plot_regional_smart_ager_distribution(
        self,
        df: pd.DataFrame,
        smart_ager_cluster: int,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Show Smart Ager distribution across Spanish regions.

        Args:
            df: DataFrame with cluster and region data.
            smart_ager_cluster: ID of the Smart Ager cluster.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Filter top regions
        top_regions = df[df['region'] != 'Other']['region'].value_counts().head(6).index.tolist()
        df_regions = df[df['region'].isin(top_regions)]

        # 1. Smart Ager Penetration by Region
        ax1 = axes[0]
        smart_ager_pct = df_regions.groupby('region').apply(
            lambda x: (x['cluster_id'] == smart_ager_cluster).sum() / len(x) * 100
        ).sort_values(ascending=True)

        bars = ax1.barh(smart_ager_pct.index, smart_ager_pct.values,
                       color=PlotConfig.COLORS['brand_primary'], alpha=0.8, edgecolor='none')
        ax1.set_xlabel('Smart Ager Penetration (%)', fontweight='600', fontsize=11)
        ax1.set_title('Smart Ager Segment: Regional Penetration',
                     fontweight='bold', fontsize=13, color=PlotConfig.COLORS['dark_navy'], pad=15)
        ax1.grid(True, axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

        # Add value labels
        for i, (idx, val) in enumerate(smart_ager_pct.items()):
            ax1.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=10,
                    fontweight='bold', color=PlotConfig.COLORS['dark_navy'])

        # 2. Regional Market Size (Total vs Smart Agers)
        ax2 = axes[1]
        regional_counts = df_regions['region'].value_counts().sort_values()
        smart_ager_counts = df_regions[df_regions['cluster_id'] == smart_ager_cluster]['region'].value_counts()

        x_pos = np.arange(len(regional_counts))
        width = 0.35

        bars1 = ax2.barh(x_pos - width/2, regional_counts.values, width,
                        label='Total Customers', color=PlotConfig.COLORS['slate'], alpha=0.5, edgecolor='none')
        bars2 = ax2.barh(x_pos + width/2, smart_ager_counts.reindex(regional_counts.index, fill_value=0).values, width,
                        label='Smart Agers', color=PlotConfig.COLORS['brand_primary'], alpha=0.8, edgecolor='none')

        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(regional_counts.index)
        ax2.set_xlabel('Number of Customers', fontweight='600', fontsize=11)
        ax2.set_title('Regional Market Size: Total vs Smart Agers',
                     fontweight='bold', fontsize=13, color=PlotConfig.COLORS['dark_navy'], pad=15)
        ax2.legend(loc='lower right', frameon=False, fontsize=10)
        ax2.grid(True, axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def plot_spain_map_regional_profiles(
        self,
        df: pd.DataFrame,
        smart_ager_cluster: int,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a professional map of Spain using actual geographic data showing regional customer profiles.

        Args:
            df: DataFrame with regional data and clusters.
            smart_ager_cluster: ID of the Smart Ager cluster.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from src.spain_map_data import SPAIN_REGIONS

        fig = plt.figure(figsize=(18, 11))

        # Create grid for multiple visualizations
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                             left=0.05, right=0.95, top=0.92, bottom=0.08)

        # Main map with cartopy projection (large, left side)
        ax_map = fig.add_subplot(gs[:, :2], projection=ccrs.PlateCarree())

        # Side metrics (right side)
        ax_intent = fig.add_subplot(gs[0, 2])
        ax_income = fig.add_subplot(gs[1, 2])

        # Prepare regional data
        regions_to_plot = [r for r in SPAIN_REGIONS.keys() if r in df['region'].values]

        regional_data = {}
        for region in regions_to_plot:
            region_df = df[df['region'] == region]
            if len(region_df) > 0:
                regional_data[region] = {
                    'count': len(region_df),
                    'winter_intent': region_df['willingness_to_buy_winter'].mean() * 100,
                    'smart_ager_pct': (region_df['cluster_id'] == smart_ager_cluster).sum() / len(region_df) * 100,
                    'avg_income': region_df['income'].mean() / 1000,
                    'avg_wtp': region_df['willingness_to_pay'].mean(),
                }

        # === MAIN MAP WITH ACTUAL SPAIN GEOGRAPHY ===
        # Set map extent to focus on Spain (including Canary Islands roughly)
        ax_map.set_extent([-10, 5, 35.5, 44.5], crs=ccrs.PlateCarree())

        # Set transparent background
        ax_map.patch.set_visible(False)
        fig.patch.set_alpha(0.0)

        # Add geographic features - OUTLINES ONLY
        # Borders - Spain and neighboring countries
        ax_map.add_feature(cfeature.BORDERS, linewidth=1.8,
                          edgecolor=PlotConfig.COLORS['slate'],
                          alpha=0.6, zorder=2)

        # Coastlines - more prominent
        ax_map.add_feature(cfeature.COASTLINE, linewidth=2.0,
                          edgecolor=PlotConfig.COLORS['steel_blue'],
                          alpha=0.8, zorder=3)

        # Add Spain country outline specifically (no fill, just outline)
        import cartopy.io.shapereader as shpreader
        countries = shpreader.natural_earth(resolution='50m', category='cultural',
                                           name='admin_0_countries')

        for country in shpreader.Reader(countries).records():
            if country.attributes['NAME'] == 'Spain':
                ax_map.add_geometries([country.geometry], ccrs.PlateCarree(),
                                     facecolor='none',  # No fill
                                     edgecolor=PlotConfig.COLORS['dark_navy'],
                                     linewidth=3.0,
                                     zorder=4)

        # Plot each region as a bubble
        for region, coords in SPAIN_REGIONS.items():
            if region in regional_data:
                data = regional_data[region]

                # Bubble size based on sample count
                size = data['count'] * 250

                # Color based on Smart Ager penetration
                smart_ager_pct = data['smart_ager_pct']
                if smart_ager_pct > 25:
                    color = PlotConfig.COLORS['brand_primary']
                    alpha = 0.9
                elif smart_ager_pct > 15:
                    color = PlotConfig.COLORS['accent_blue']
                    alpha = 0.8
                else:
                    color = PlotConfig.COLORS['light_blue']
                    alpha = 0.6

                # Plot bubble
                ax_map.scatter(coords['lon'], coords['lat'], s=size,
                             color=color, alpha=alpha,
                             edgecolors='white', linewidth=3, zorder=10,
                             transform=ccrs.PlateCarree())

                # Add city label
                ax_map.text(coords['lon'], coords['lat'] + 0.45, region,
                          fontsize=11, fontweight='bold', ha='center',
                          color=PlotConfig.COLORS['dark_navy'],
                          bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                   edgecolor=PlotConfig.COLORS['slate'], linewidth=1, alpha=0.95),
                          transform=ccrs.PlateCarree(), zorder=11)

                # Add metrics label below
                metrics_text = f"{data['winter_intent']:.0f}% intent | {data['smart_ager_pct']:.0f}% SA"
                ax_map.text(coords['lon'], coords['lat'] - 0.5, metrics_text,
                          fontsize=8.5, ha='center', va='top',
                          color=PlotConfig.COLORS['steel_blue'], fontweight='600',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='none', alpha=0.85),
                          transform=ccrs.PlateCarree(), zorder=11)

        # Map styling
        ax_map.set_title('SPAIN: REGIONAL CUSTOMER INSIGHTS MAP',
                        fontsize=17, fontweight='bold', pad=20,
                        color=PlotConfig.COLORS['dark_navy'])

        # Add gridlines
        gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                             alpha=0.3, linestyle=':', zorder=3)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9, 'color': PlotConfig.COLORS['slate']}
        gl.ylabel_style = {'size': 9, 'color': PlotConfig.COLORS['slate']}

        # Add legend
        legend_elements = [
            plt.scatter([], [], s=200, c=PlotConfig.COLORS['brand_primary'],
                       alpha=0.9, edgecolors='white', linewidth=2,
                       label='High Smart Ager % (>25%)'),
            plt.scatter([], [], s=200, c=PlotConfig.COLORS['accent_blue'],
                       alpha=0.8, edgecolors='white', linewidth=2,
                       label='Medium Smart Ager % (15-25%)'),
            plt.scatter([], [], s=200, c=PlotConfig.COLORS['light_blue'],
                       alpha=0.6, edgecolors='white', linewidth=2,
                       label='Lower Smart Ager % (<15%)'),
        ]
        ax_map.legend(handles=legend_elements, loc='lower left',
                     frameon=True, fancybox=True, shadow=True,
                     fontsize=9, framealpha=0.9)

        # Add note
        ax_map.text(0.98, 0.02, 'Bubble size = sample count | SA = Smart Agers',
                   transform=ax_map.transAxes, fontsize=8,
                   ha='right', va='bottom', style='italic',
                   color=PlotConfig.COLORS['slate'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='none', alpha=0.7))

        # === WINTER PURCHASE INTENT BAR ===
        if regional_data:
            regions_sorted = sorted(regional_data.items(),
                                  key=lambda x: x[1]['winter_intent'])
            region_names = [r[0] for r in regions_sorted]
            intent_values = [r[1]['winter_intent'] for r in regions_sorted]

            bars = ax_intent.barh(region_names, intent_values,
                                 color=PlotConfig.COLORS['brand_primary'],
                                 alpha=0.8, edgecolor='none')
            ax_intent.set_xlabel('Winter Intent (%)', fontsize=9, fontweight='600')
            ax_intent.set_title('Purchase Intent by Region', fontsize=11,
                              fontweight='bold', color=PlotConfig.COLORS['dark_navy'])
            ax_intent.grid(True, axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

            # Add value labels
            for i, val in enumerate(intent_values):
                ax_intent.text(val + 1, i, f'{val:.0f}%', va='center',
                             fontsize=8, fontweight='bold',
                             color=PlotConfig.COLORS['dark_navy'])

        # === AVERAGE INCOME BAR ===
        if regional_data:
            regions_sorted = sorted(regional_data.items(),
                                  key=lambda x: x[1]['avg_income'])
            region_names = [r[0] for r in regions_sorted]
            income_values = [r[1]['avg_income'] for r in regions_sorted]

            bars = ax_income.barh(region_names, income_values,
                                 color=PlotConfig.COLORS['accent_blue'],
                                 alpha=0.8, edgecolor='none')
            ax_income.set_xlabel('Avg Income (€K)', fontsize=9, fontweight='600')
            ax_income.set_title('Average Income by Region', fontsize=11,
                               fontweight='bold', color=PlotConfig.COLORS['dark_navy'])
            ax_income.grid(True, axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

            # Add value labels
            for i, val in enumerate(income_values):
                ax_income.text(val + 1, i, f'€{val:.0f}K', va='center',
                             fontsize=8, fontweight='bold',
                             color=PlotConfig.COLORS['dark_navy'])

        # Overall title
        fig.suptitle('SPAIN REGIONAL MARKET ANALYSIS: ECRAN CUSTOMER PROFILES',
                    fontsize=18, fontweight='bold', y=0.98,
                    color=PlotConfig.COLORS['dark_navy'])

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            self.save_plot(filename, fig)

        return fig

    def plot_winter_barriers(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize barriers to winter sunscreen purchase.
        Answers: "Why do customers only buy sunscreen in summer?"

        Args:
            df: DataFrame with barrier columns.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

        # Define barrier columns and labels
        barrier_cols = [
            'barrier_no_sun',
            'barrier_too_expensive',
            'barrier_heavy_texture',
            'barrier_forget_apply',
            'barrier_not_beach',
            'barrier_unaware_damage',
            'barrier_time_consuming'
        ]

        barrier_labels = [
            "Don't see the need\n(Less sun in winter)",
            "Too expensive for\nyear-round use",
            "Texture too heavy/greasy\nfor daily wear",
            "Forget to apply\n(not in routine)",
            "Only use at beach/\noutdoor activities",
            "Unaware of winter\nUV damage",
            "Takes too long\nto apply daily"
        ]

        # === SUBPLOT 1: Overall Barriers (Top) ===
        ax1 = fig.add_subplot(gs[0, :])

        # Calculate % mentioning each barrier
        barrier_pcts = []
        for col in barrier_cols:
            pct = (df[col].sum() / len(df)) * 100
            barrier_pcts.append(pct)

        # Sort by percentage
        sorted_indices = np.argsort(barrier_pcts)[::-1]
        sorted_labels = [barrier_labels[i] for i in sorted_indices]
        sorted_pcts = [barrier_pcts[i] for i in sorted_indices]

        # Color gradient: darker for higher % (orange-yellowish palette)
        colors = plt.cm.YlOrRd(np.linspace(0.5, 0.95, len(sorted_pcts)))

        bars = ax1.barh(sorted_labels, sorted_pcts, color=colors,
                       alpha=0.85, edgecolor='white', linewidth=2)

        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, sorted_pcts)):
            ax1.text(pct + 1.5, i, f'{pct:.1f}%',
                    va='center', fontsize=11, fontweight='bold',
                    color=PlotConfig.COLORS['dark_navy'])

        ax1.set_xlabel('% of Respondents Citing Barrier', fontsize=12, fontweight='600')
        ax1.set_title('WHY CUSTOMERS ONLY BUY SUNSCREEN IN SUMMER: KEY BARRIERS',
                     fontsize=15, fontweight='bold', pad=15,
                     color=PlotConfig.COLORS['dark_navy'])
        ax1.set_xlim(0, max(sorted_pcts) * 1.15)
        ax1.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)

        # === SUBPLOT 2: Barriers by Segment (Bottom Left) ===
        ax2 = fig.add_subplot(gs[1, 0])

        # Top 4 barriers
        top_4_barriers = [barrier_cols[i] for i in sorted_indices[:4]]
        top_4_labels_short = [
            "No sun\nin winter",
            "Too\nexpensive",
            "Heavy\ntexture",
            "Forget to\napply"
        ]

        # Calculate by archetype
        archetypes = ['smart_ager', 'beach_traditionalist', 'budget_conscious', 'indifferent']
        archetype_names = ['Smart Ager', 'Beach\nTraditionalist', 'Budget\nConscious', 'Indifferent']

        x = np.arange(len(top_4_labels_short))
        width = 0.2

        segment_colors = [
            PlotConfig.COLORS['brand_primary'],      # Deep amber
            PlotConfig.COLORS['accent_blue'],        # Medium orange
            PlotConfig.COLORS['brand_secondary'],     # Bright amber
            PlotConfig.COLORS['light_blue']          # Light orange
        ]

        for i, (archetype, name, color) in enumerate(zip(archetypes, archetype_names, segment_colors)):
            segment_df = df[df['archetype'] == archetype]
            if len(segment_df) == 0:
                continue

            segment_pcts = []
            for barrier in top_4_barriers:
                pct = (segment_df[barrier].sum() / len(segment_df)) * 100
                segment_pcts.append(pct)

            ax2.bar(x + i * width, segment_pcts, width, label=name,
                   color=color, alpha=0.8, edgecolor='white', linewidth=1)

        ax2.set_ylabel('% Citing Barrier', fontsize=10, fontweight='600')
        ax2.set_title('Barriers by Customer Segment', fontsize=12,
                     fontweight='bold', color=PlotConfig.COLORS['dark_navy'])
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(top_4_labels_short, fontsize=9)
        ax2.legend(loc='upper right', frameon=True, fontsize=9, framealpha=0.9)
        ax2.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.set_ylim(0, 100)

        # === SUBPLOT 3: Strategic Insight Box (Bottom Right) ===
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')

        # Calculate key insights
        top_barrier_idx = sorted_indices[0]
        top_barrier_pct = sorted_pcts[0]
        top_barrier_name = barrier_labels[top_barrier_idx].replace('\n', ' ')

        non_winter_buyers = df[df['willingness_to_buy_winter'] == 0]
        winter_buyers = df[df['willingness_to_buy_winter'] == 1]

        if len(non_winter_buyers) > 0:
            top_barrier_non_winter = (non_winter_buyers[barrier_cols[top_barrier_idx]].sum() /
                                     len(non_winter_buyers)) * 100
        else:
            top_barrier_non_winter = 0

        # Create text box
        insight_text = f"""
KEY FINDINGS:

🎯 PRIMARY BARRIER
   {top_barrier_pct:.0f}% of respondents cite:
   "{top_barrier_name}"

📊 NON-WINTER BUYERS
   {top_barrier_non_winter:.0f}% mention top barrier
   vs. {100-top_barrier_non_winter:.0f}% who don't

💡 STRATEGIC IMPLICATION
   • Education gap on year-round UV damage
   • Need lighter, cosmetically elegant textures
   • Reposition as "daily facial skincare"
     not "sunscreen"

🚀 RECOMMENDATIONS
   1. Emphasize blue light + UVA aging
   2. Develop lightweight formulas
   3. Everyday packaging (pump, not tube)
   4. Position with moisturizers, not beach
        """

        ax3.text(0.05, 0.95, insight_text,
                transform=ax3.transAxes,
                fontsize=10, va='top', ha='left',
                fontfamily='monospace',
                color=PlotConfig.COLORS['dark_navy'],
                bbox=dict(boxstyle='round,pad=1',
                         facecolor=PlotConfig.COLORS['sky_blue'],  # Yellow background
                         edgecolor=PlotConfig.COLORS['steel_blue'],  # Medium brown edge
                         linewidth=2, alpha=0.3))

        # Overall title
        fig.suptitle('WINTER SUNSCREEN BARRIERS: ROOT CAUSE ANALYSIS',
                    fontsize=17, fontweight='bold', y=0.98,
                    color=PlotConfig.COLORS['dark_navy'])

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            self.save_plot(filename, fig)

        return fig

    def plot_competitive_table(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a styled competitive landscape table.

        Args:
            df: DataFrame containing competitive data.
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        # Calculate figure height based on rows
        row_height = 0.8
        header_height = 1.2
        n_rows = len(df)
        fig_height = header_height + (n_rows * row_height) + 1

        fig, ax = plt.subplots(figsize=(16, fig_height))

        # Hide axes
        ax.axis('off')

        # Define column widths (relative)
        # Columns: Product, Channel, Positioning, Tech, Blue Light, Texture, Heritage
        # Removed Price (was index 2)
        col_widths = [0.20, 0.12, 0.10, 0.18, 0.08, 0.14, 0.18]

        # Table data
        cell_text = []
        for row in df.values:
            cell_text.append(row)

        # Create table
        table = ax.table(
            cellText=cell_text,
            colLabels=df.columns,
            loc='center',
            cellLoc='left',
            colWidths=col_widths,
            bbox=[0, 0, 1, 1]
        )

        # Styling
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)  # Increase row height

        # Iterate through cells to style them
        for (row, col), cell in table.get_celld().items():
            # Header styling
            if row == 0:
                cell.set_text_props(weight='bold', color='white', ha='center')
                cell.set_facecolor(PlotConfig.COLORS['brand_primary'])
                cell.set_edgecolor('white')
                cell.set_linewidth(1)
                cell.set_height(0.1)
            else:
                # Body styling
                cell.set_edgecolor('#eeeeee')
                cell.set_linewidth(0.5)

                # Highlight Sunnique row
                # Assuming Sunnique is the first row or identified by name
                is_sunnique = 'Sunnique' in str(df.iloc[row-1, 0])

                if is_sunnique:
                    cell.set_facecolor(PlotConfig.COLORS['light_blue'])
                    cell.set_text_props(weight='bold', color=PlotConfig.COLORS['dark_navy'])
                    # Add left border highlight
                    if col == 0:
                        cell.set_linewidth(2)
                        cell.set_edgecolor(PlotConfig.COLORS['brand_primary'])
                else:
                    # Alternating row colors for others
                    if row % 2 == 0:
                        cell.set_facecolor('#f8f9fa')
                    else:
                        cell.set_facecolor('white')
                    cell.set_text_props(color='#333333')

                # Center align some columns
                # Positioning (2), Blue Light (4)
                if col in [2, 4]:
                    cell.set_text_props(ha='center')
                    if is_sunnique: # Re-apply bold/color if overwritten
                         cell.set_text_props(weight='bold', color=PlotConfig.COLORS['dark_navy'], ha='center')

        # Add title
        plt.title(
            "COMPETITIVE LANDSCAPE: FACIAL SUN CARE 2025",
            pad=20,
            fontweight='bold',
            fontsize=16,
            color=PlotConfig.COLORS['dark_navy'],
            loc='left'
        )

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

    def plot_bcg_matrix(
        self,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a BCG Growth-Share Matrix for ECRAN product portfolio.
        Uses orange-yellowish palette to match the overall design.

        Args:
            filename: If provided, saves the plot.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Define quadrant boundaries
        x_center = 0.5
        y_center = 0.5

        # Create quadrants with orange-yellowish colors
        # Top-Left: Stars (High Growth, High Share) - Bright amber
        ax.add_patch(plt.Rectangle((0, y_center), x_center, y_center,
                                   facecolor=PlotConfig.COLORS['brand_secondary'],
                                   alpha=0.3, edgecolor='none', zorder=0))

        # Top-Right: Question Marks (High Growth, Low Share) - Medium orange
        ax.add_patch(plt.Rectangle((x_center, y_center), x_center, y_center,
                                   facecolor=PlotConfig.COLORS['accent_blue'],
                                   alpha=0.3, edgecolor='none', zorder=0))

        # Bottom-Left: Cash Cows (Low Growth, High Share) - Light orange
        ax.add_patch(plt.Rectangle((0, 0), x_center, y_center,
                                   facecolor=PlotConfig.COLORS['light_blue'],
                                   alpha=0.3, edgecolor='none', zorder=0))

        # Bottom-Right: Dogs (Low Growth, Low Share) - Yellow
        ax.add_patch(plt.Rectangle((x_center, 0), x_center, y_center,
                                   facecolor=PlotConfig.COLORS['sky_blue'],
                                   alpha=0.3, edgecolor='none', zorder=0))

        # Draw dividing lines
        ax.axhline(y=y_center, color=PlotConfig.COLORS['dark_navy'],
                  linewidth=2, linestyle='-', zorder=1)
        ax.axvline(x=x_center, color=PlotConfig.COLORS['dark_navy'],
                  linewidth=2, linestyle='-', zorder=1)

        # Product positions and labels
        products = [
            {
                'name': 'SPF 50+ & Large Formats',
                'x': 0.25,  # High share
                'y': 0.75,  # High growth
                'icon': '*',
                'icon_label': 'STARS',
                'quadrant': 'Stars'
            },
            {
                'name': 'Facial Protection',
                'x': 0.75,  # Low share
                'y': 0.75,  # High growth
                'icon': '?',
                'icon_label': '?',
                'quadrant': 'Question Marks'
            },
            {
                'name': 'Sunmilk & Aftersun',
                'x': 0.25,  # High share
                'y': 0.25,  # Low growth
                'icon': 'C',
                'icon_label': 'COWS',
                'quadrant': 'Cash Cows'
            },
            {
                'name': 'SPF 15 & SPF 20',
                'x': 0.75,  # Low share
                'y': 0.25,  # Low growth
                'icon': 'D',
                'icon_label': 'DOGS',
                'quadrant': 'Dogs'
            }
        ]

        # Plot products
        for product in products:
            # Icon (using text symbol)
            ax.text(product['x'], product['y'] + 0.05, product['icon_label'],
                   fontsize=36, ha='center', va='center',
                   color='white', weight='bold', zorder=3,
                   family='sans-serif')

            # Product name
            ax.text(product['x'], product['y'] - 0.12, product['name'],
                   fontsize=11, ha='center', va='center',
                   color='white', weight='bold', zorder=3,
                   bbox=dict(boxstyle='round,pad=0.5',
                            facecolor=PlotConfig.COLORS['dark_navy'],
                            edgecolor='white', linewidth=1.5, alpha=0.9))

        # Axis labels
        ax.set_xlabel('Relative Market Share', fontsize=14, fontweight='bold',
                     color=PlotConfig.COLORS['dark_navy'], labelpad=15)
        ax.set_ylabel('Market Growth', fontsize=14, fontweight='bold',
                     color=PlotConfig.COLORS['dark_navy'], labelpad=15)

        # Axis ticks and labels
        ax.set_xticks([0.25, 0.75])
        ax.set_xticklabels(['High', 'Low'], fontsize=12, fontweight='600',
                          color=PlotConfig.COLORS['dark_navy'])
        ax.set_yticks([0.25, 0.75])
        ax.set_yticklabels(['Low', 'High'], fontsize=12, fontweight='600',
                          color=PlotConfig.COLORS['dark_navy'])

        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Title
        ax.set_title('ECRAN Product Portfolio Analysis (BCG Matrix)',
                    fontsize=16, fontweight='bold', pad=20,
                    color=PlotConfig.COLORS['dark_navy'])

        # Remove spines except for visual clarity
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add quadrant labels in corners
        ax.text(0.05, 0.95, 'STARS', fontsize=10, fontweight='bold',
               color=PlotConfig.COLORS['brand_secondary'], alpha=0.6,
               ha='left', va='top')
        ax.text(0.95, 0.95, 'QUESTION\nMARKS', fontsize=10, fontweight='bold',
               color=PlotConfig.COLORS['accent_blue'], alpha=0.6,
               ha='right', va='top')
        ax.text(0.05, 0.05, 'CASH\nCOWS', fontsize=10, fontweight='bold',
               color=PlotConfig.COLORS['light_blue'], alpha=0.6,
               ha='left', va='bottom')
        ax.text(0.95, 0.05, 'DOGS', fontsize=10, fontweight='bold',
               color=PlotConfig.COLORS['sky_blue'], alpha=0.6,
               ha='right', va='bottom')

        # Set background to transparent
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)

        plt.tight_layout()

        if filename:
            self.save_plot(filename, fig)

        return fig

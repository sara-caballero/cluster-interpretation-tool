# Report Theme System for Cluster Analysis PDF
# Provides consistent styling, colors, and formatting utilities

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
import warnings

class ReportTheme:
    """Centralized theme configuration for PDF reports"""
    
    # Color palette (colorblind-safe)
    PRIMARY = "#2F6DB3"
    SECONDARY = "#7CB490" 
    ACCENT = "#F4A340"
    DANGER = "#D9534F"
    NEUTRAL_DARK = "#334155"
    NEUTRAL_MID = "#94A3B8"
    NEUTRAL_LIGHT = "#E5E7EB"
    
    # Cluster colors - distinct palette for up to 12 clusters
    CLUSTER_COLORS = [
        "#2F6DB3",  # Primary blue
        "#7CB490",  # Secondary green
        "#F4A340",  # Accent orange
        "#D9534F",  # Danger red
        "#8E44AD",  # Purple
        "#16A085",  # Teal
        "#E67E22",  # Dark orange
        "#2ECC71",  # Emerald
        "#E74C3C",  # Light red
        "#9B59B6",  # Light purple
        "#1ABC9C",  # Turquoise
        "#F39C12"   # Yellow
    ]
    
    # Typography
    FONT_FAMILY = "DejaVu Sans"  # Fallback for ReportLab compatibility
    TITLE_SIZE = 16
    HEADING_SIZE = 14
    BODY_SIZE = 10
    CAPTION_SIZE = 8
    
    # Spacing (base unit = 8px)
    BASE_UNIT = 8
    MARGIN_LARGE = BASE_UNIT * 3  # 24px
    MARGIN_MEDIUM = BASE_UNIT * 2  # 16px
    MARGIN_SMALL = BASE_UNIT * 1  # 8px
    
    # Plot settings
    DPI = 300
    FIGURE_SIZE = (8, 6)
    POINT_SIZE = 25
    ALPHA_CLUSTER = 0.35
    ALPHA_HULL = 0.5
    
    @classmethod
    def get_cluster_color(cls, cluster_id: int) -> str:
        """Get color for cluster ID, cycling through palette"""
        return cls.CLUSTER_COLORS[cluster_id % len(cls.CLUSTER_COLORS)]
    
    @classmethod
    def get_cluster_color_rgba(cls, cluster_id: int, alpha: float = 1.0) -> Tuple[float, float, float, float]:
        """Get RGBA color for cluster ID"""
        color = cls.get_cluster_color(cluster_id)
        return to_rgba(color, alpha)

# Global theme instance
THEME = ReportTheme()

def prettify_name(name: str) -> str:
    """
    Convert feature names to human-readable format with consistent casing.
    
    Args:
        name: Raw feature name (e.g., "max_hr", "ST_Slope", "oldpeak")
    
    Returns:
        Prettified name (e.g., "MaxHR", "ST Slope", "Oldpeak")
    """
    # Special case overrides for common medical/technical terms
    special_cases = {
        "max_hr": "MaxHR",
        "maxhr": "MaxHR", 
        "st_slope": "ST Slope",
        "stslope": "ST Slope",
        "oldpeak": "Oldpeak",
        "old_peak": "Oldpeak",
        "restecg": "RestECG",
        "rest_ecg": "RestECG",
        "chol": "Cholesterol",
        "trestbps": "Resting BP",
        "trest_bps": "Resting BP",
        "fbs": "Fasting Blood Sugar",
        "exang": "Exercise Angina",
        "ca": "Major Vessels",
        "thal": "Thalassemia",
        "cp": "Chest Pain Type",
        "sex": "Sex",
        "age": "Age",
        "target": "Target"
    }
    
    # Check special cases first
    if name.lower() in special_cases:
        return special_cases[name.lower()]
    
    # Handle CamelCase and snake_case
    if '_' in name:
        # Snake case: replace underscores with spaces and title case
        parts = name.split('_')
        return ' '.join(part.title() for part in parts)
    elif any(c.isupper() for c in name[1:]):
        # CamelCase: add spaces before capitals
        import re
        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return spaced.title()
    else:
        # Simple title case
        return name.title()

def fmt_val(x: Union[int, float], decimals: Optional[int] = None) -> str:
    """
    Format numeric values with appropriate precision and separators.
    
    Args:
        x: Numeric value to format
        decimals: Number of decimal places (auto-detect if None)
    
    Returns:
        Formatted string
    """
    if pd.isna(x):
        return "N/A"
    
    # Auto-detect decimals for clean formatting
    if decimals is None:
        if abs(x) < 0.01:
            decimals = 3
        elif abs(x) < 1:
            decimals = 2
        elif abs(x) < 100:
            decimals = 1
        else:
            decimals = 0
    
    # Format with appropriate precision
    if decimals == 0:
        return f"{int(round(x)):,}"
    else:
        return f"{x:,.{decimals}f}"

def fmt_pct(p: float) -> str:
    """Format percentage values (0.83 -> "83%")"""
    if pd.isna(p):
        return "N/A"
    return f"{int(round(p * 100))}%"

def fmt_pair(a: float, b: float, pct_change: Optional[float] = None) -> str:
    """
    Format comparison pair with optional percentage change.
    
    Args:
        a: First value
        b: Second value  
        pct_change: Optional percentage change to highlight
    
    Returns:
        Formatted comparison string
    """
    a_str = fmt_val(a)
    b_str = fmt_val(b)
    
    if pct_change is not None:
        change_str = f" ({fmt_pct(abs(pct_change))} {'higher' if pct_change > 0 else 'lower'})"
        return f"{a_str} vs {b_str}{change_str}"
    else:
        return f"{a_str} vs {b_str}"

def legend_cluster_label(cluster_id: int, count: int, total: int) -> str:
    """Generate consistent cluster legend label"""
    pct = int(round(100 * count / total))
    return f"Cluster {cluster_id} — n={count:,} ({pct}%)"

def apply_theme(ax, title: Optional[str] = None, xlabel: Optional[str] = None, 
                ylabel: Optional[str] = None, grid: bool = False):
    """
    Apply consistent theme styling to matplotlib axes.
    
    Args:
        ax: Matplotlib axes object
        title: Optional title (will be prettified)
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
        grid: Whether to show grid
    """
    # Set font family
    plt.rcParams['font.family'] = THEME.FONT_FAMILY
    plt.rcParams['font.size'] = THEME.BODY_SIZE
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Style remaining spines
    ax.spines['left'].set_color(THEME.NEUTRAL_MID)
    ax.spines['bottom'].set_color(THEME.NEUTRAL_MID)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Set background
    ax.set_facecolor('white')
    
    # Configure ticks
    ax.tick_params(axis='both', which='major', 
                   colors=THEME.NEUTRAL_DARK, 
                   labelsize=THEME.BODY_SIZE)
    
    # Grid
    if grid:
        ax.grid(True, alpha=0.3, color=THEME.NEUTRAL_LIGHT, linewidth=0.5)
    else:
        ax.grid(False)
    
    # Labels and title
    if title:
        ax.set_title(prettify_name(title), 
                    fontsize=THEME.HEADING_SIZE, 
                    fontweight='600',
                    color=THEME.NEUTRAL_DARK,
                    pad=THEME.MARGIN_MEDIUM)
    
    if xlabel:
        ax.set_xlabel(prettify_name(xlabel), 
                     fontsize=THEME.BODY_SIZE,
                     color=THEME.NEUTRAL_DARK)
    
    if ylabel:
        ax.set_ylabel(prettify_name(ylabel), 
                     fontsize=THEME.BODY_SIZE,
                     color=THEME.NEUTRAL_DARK)

def setup_matplotlib_theme():
    """Configure global matplotlib settings for the theme"""
    plt.rcParams.update({
        'font.family': THEME.FONT_FAMILY,
        'font.size': THEME.BODY_SIZE,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 0.8,
        'axes.edgecolor': THEME.NEUTRAL_MID,
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.axisbelow': True,
        'xtick.color': THEME.NEUTRAL_DARK,
        'ytick.color': THEME.NEUTRAL_DARK,
        'xtick.labelsize': THEME.BODY_SIZE,
        'ytick.labelsize': THEME.BODY_SIZE,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': THEME.NEUTRAL_LIGHT,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'savefig.dpi': THEME.DPI,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'lines.antialiased': True,
        'patch.antialiased': True
    })

def get_direction_symbol(direction: str) -> str:
    """Get arrow symbol for direction"""
    if direction.lower() == 'higher':
        return "▲"
    elif direction.lower() == 'lower':
        return "▼"
    else:
        return "•"

def get_direction_color(direction: str) -> str:
    """Get color for direction"""
    if direction.lower() == 'higher':
        return THEME.SECONDARY
    elif direction.lower() == 'lower':
        return THEME.DANGER
    else:
        return THEME.NEUTRAL_DARK

# Initialize theme when module is imported
setup_matplotlib_theme()

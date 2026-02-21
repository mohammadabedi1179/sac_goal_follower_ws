"""
Training Analysis Plotter — GOLDEN HOUR EDITION
Warm sunset colors with golden highlights and cream tones

Reads episode_stats.json and generates the full analysis suite.

Usage:
  python golden_hour_plot_results.py --json episode_stats.json --out plots_out --rolling 20 --topk 12
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════
# GOLDEN HOUR THEME CONFIG
# ═══════════════════════════════════════════
BG        = "#fffdf7"      # Warm ivory
BG_AXES   = "#fffdf7"      # Warm ivory
GRID_CLR  = "#fde8cc"      # Peachy cream grid
TEXT_CLR   = "#8b4513"     # Rich brown
TICK_CLR   = "#a0522d"     # Sienna
EDGE_CLR   = "#ddbea9"     # Warm beige
PRIMARY   = "#ff8500"      # Sunset orange
SECONDARY = "#d2691e"      # Chocolate orange
ACCENT    = "#ffb347"      # Peach orange

# Sunset inspired palette
REASON_COLORS = {
    "goal":      "#228b22",   # Forest green
    "collision": "#dc143c",   # Crimson
    "timeout":   "#ff6347",   # Tomato orange
}

SHIELD_COLORS = {
    "none":     "#228b22",    # Forest green
    "caution":  "#ff6347",    # Tomato
    "danger":   "#dc143c",    # Crimson  
    "critical": "#8b0000",    # Dark red
}

LOSS_COLORS = {
    "actor":  "#ff8500",      # Sunset orange
    "critic": "#d2691e",      # Chocolate orange
}


def _apply_golden_hour_rc():
    """Set global rcParams for Golden Hour theme."""
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG_AXES,
        "axes.edgecolor":    EDGE_CLR,
        "axes.grid":         True,
        "grid.color":        GRID_CLR,
        "grid.alpha":        0.6,
        "grid.linewidth":    0.8,
        "grid.linestyle":    "-",
        "axes.labelcolor":   TEXT_CLR,
        "text.color":        TEXT_CLR,
        "xtick.color":       TICK_CLR,
        "ytick.color":       TICK_CLR,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.2,
        "font.size":         11,
        "font.family":       "sans-serif",
        "font.weight":       "normal",
        "legend.facecolor":  "#ffffff",
        "legend.edgecolor":  PRIMARY,
        "legend.labelcolor": TEXT_CLR,
        "legend.framealpha": 0.9,
        "savefig.facecolor": BG,
        "savefig.dpi":       150,
    })


def reason_color(r):
    return REASON_COLORS.get(r, PRIMARY)


def reason_palette(reasons):
    return {r: reason_color(r) for r in sorted(set(reasons))}


# ─────────────────────────────
# Golden hour effect helpers
# ─────────────────────────────
def sunset_line(ax, x, y, color, lw=2.5, label=None, alpha=0.9):
    """Draw a line with golden sunset warmth."""
    # Warm golden glow
    ax.plot(x, y, linewidth=lw + 2.0, color=ACCENT, alpha=0.2, solid_capstyle="round")
    # Main radiant line
    ax.plot(x, y, linewidth=lw, color=color, alpha=alpha, solid_capstyle="round", label=label)


def golden_scatter(ax, x, y, color, s=22, label=None, alpha=0.8):
    """Scatter with golden hour lighting."""
    # Warm golden aura
    ax.scatter(x, y, s=s * 2.0, alpha=0.15, color=ACCENT, edgecolors="none")
    # Main glowing points
    ax.scatter(x, y, s=s, alpha=alpha, color=color, edgecolors="#ffffff", 
               linewidths=1.2, label=label)


def horizon_hline(ax, y, color, label=None):
    """Horizontal line like sunset horizon."""
    ax.axhline(y=y, color=color, linewidth=2.8, alpha=0.3, linestyle="-")
    ax.axhline(y=y, color=color, linewidth=1.5, alpha=0.7, linestyle="-", label=label)


def golden_legend(ax, loc="best", fontsize=9):
    """Warm legend with golden hour styling."""
    legend = ax.legend(loc=loc, fontsize=fontsize, facecolor="#ffffff", 
                      edgecolor=PRIMARY, labelcolor=TEXT_CLR)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_linewidth(1.5)


def sunset_title(ax, title):
    """Warm title styling like golden hour light."""
    ax.set_title(title, color=PRIMARY, fontsize=14, fontweight="600", 
                pad=20, family="sans-serif")


def golden_boxplot(ax, data, tick_labels, colors):
    """Styled boxplot with sunset aesthetic."""
    bp = ax.boxplot(
        data, labels=tick_labels, patch_artist=True, showfliers=True,
        flierprops=dict(marker="o", markersize=4, markerfacecolor=ACCENT, 
                       markeredgecolor="#ffffff", markeredgewidth=1.0),
        medianprops=dict(color=PRIMARY, linewidth=3.0),
        whiskerprops=dict(color=TEXT_CLR, linewidth=1.5),
        capprops=dict(color=TEXT_CLR, linewidth=1.5),
        boxprops=dict(linewidth=1.5)
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.5)
    return bp


# ─────────────────────────────
# Helper functions
# ─────────────────────────────
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def rolling_mean(x, w):
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) == 0:
        return x
    out = np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(np.where(np.isfinite(x), x, 0.0))
    n = np.cumsum(np.isfinite(x).astype(float))
    for i in range(len(x)):
        j0 = max(0, i - w + 1)
        s = c[i] - (c[j0 - 1] if j0 > 0 else 0.0)
        k = n[i] - (n[j0 - 1] if j0 > 0 else 0.0)
        out[i] = (s / k) if k > 0 else np.nan
    return out

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def savefig(fname):
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ☀️ {fname}")


def plot_reward(episodes, outdir, w):
    ep = [e["episode"] for e in episodes]
    rew = [safe_float(e.get("reward")) for e in episodes]
    if all(np.isnan(r) for r in rew): return

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Individual rewards as golden particles
    golden_scatter(ax, ep, rew, PRIMARY, s=18, alpha=0.6)
    
    # Rolling mean as sunset ray
    sunset_line(ax, ep, rolling_mean(rew, w), PRIMARY, lw=2.8, 
               label=f"Rolling mean ({w})")
    
    # Reference lines like sunset horizon
    horizon_hline(ax, 0.0, EDGE_CLR, label="Horizon line")
    if any(r > 0 for r in rew if np.isfinite(r)):
        horizon_hline(ax, np.nanmax(rew), REASON_COLORS["goal"], label="Golden peak")
    
    ax.set_xlabel("Episode", fontweight="500")
    ax.set_ylabel("Reward", fontweight="500")
    sunset_title(ax, "Training Rewards — Golden Hour Journey")
    golden_legend(ax)
    savefig(os.path.join(outdir, "01_golden_hour_reward_progress.png"))


def main():
    parser = argparse.ArgumentParser(description="Plot training results — Golden Hour Edition")
    parser.add_argument("--json", required=True, help="Path to episode_stats.json")
    parser.add_argument("--out", default="plots_out", help="Output directory for figures.")
    parser.add_argument("--rolling", type=int, default=20, help="Rolling window size.")
    parser.add_argument("--topk", type=int, default=12, help="Representative episodes for overlays.")
    args = parser.parse_args()

    ensure_dir(args.out)
    _apply_golden_hour_rc()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("JSON must be a non-empty list of episode dicts.")

    data = sorted(data, key=lambda e: safe_float(e.get("episode"), np.inf))
    print(f"☀️ Loaded {len(data)} episodes. Generating Golden Hour plots...")

    plot_reward(data, args.out, args.rolling)
    # Additional plotting functions would follow...

    print(f"☀️ Done! Golden hour plots saved to: {os.path.abspath(args.out)}")
    print("✨ Your visualizations now glow with the warm radiance of sunset magic.")


if __name__ == "__main__":
    main()
"""
Created on September 29, 2025

@author: Jiale Lao

Plotting script to visualize the trade-offs between SF scaling, execution time,
money cost, and quality for LOTUS approximate vs exact methods.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set professional academic style with large fonts for paper (following temp_plot_reasoning.py)
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 6,
    'xtick.minor.size': 4,
    'ytick.major.size': 6,
    'ytick.minor.size': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})

# ACM two-column layout dimensions
latex_full_width = 7.031875
latex_single_column_width = 3.349263889


def load_lotus_data() -> pd.DataFrame:
    """Load and parse LOTUS experimental data from JSON files."""
    base_dir = Path("files/detective/metrics/lotus/")

    # Define the file mappings
    files = {
        ("SF200", "exact"): "lotus_SF200_exact.json",
        ("SF200", "approximate"): "lotus_SF200_approximate.json",
        ("SF600", "exact"): "lotus_SF600_exact.json",
        ("SF600", "approximate"): "lotus_SF600_approximaate.json"  # Note: keeping original typo in filename
    }

    data_records = []

    for (sf, method), filename in files.items():
        filepath = base_dir / filename

        if not filepath.exists():
            print(f"Warning: File {filepath} not found, skipping...")
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Extract data for Q1, Q2, Q3 (queries with f1_score)
        for query in ["Q1", "Q2", "Q3"]:
            if query in data:
                record = {
                    "SF": int(sf[2:]),  # Extract numeric part (200 or 600)
                    "Method": method,
                    "Query": query,
                    "Execution_Time": data[query]["execution_time"],
                    "Money_Cost": data[query]["money_cost"],
                    "F1_Score": data[query]["f1_score"]
                }
                data_records.append(record)

    return pd.DataFrame(data_records)


def create_sf_scaling_plot(df: pd.DataFrame) -> None:
    """Create compact bar charts for the 3 queries showing SF scaling effects and approximation benefits."""

    # Set up compact figure (following temp_plot_reasoning.py style)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Color scheme: blues for SF200, oranges for SF600 (following temp_plot_reasoning.py)
    colors = {
        "exact_200": "#2E5984",       # Dark blue - exact SF200
        "approximate_200": "#4A90E2", # Light blue - approx SF200
        "exact_600": "#D2691E",       # Dark orange - exact SF600
        "approximate_600": "#FF8C42"  # Light orange - approx SF600
    }

    metrics = ["Execution_Time", "Money_Cost", "F1_Score"]
    metric_labels = ["Execution Time (seconds)", "Monetary Cost (USD)", "F1 Score"]
    queries = ["Q1", "Q2", "Q3"]

    # Set up bar positions - keep SF200 methods close together, SF600 methods close together
    x_pos = np.arange(len(queries))
    bar_width = 0.2

    for idx, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # Bar positions: SF200 exact/approx close together, then gap, then SF600 exact/approx close together
        positions = {
            "exact_200": x_pos - 1.5 * bar_width,
            "approximate_200": x_pos - 0.5 * bar_width,
            "exact_600": x_pos + 0.5 * bar_width,
            "approximate_600": x_pos + 1.5 * bar_width
        }

        # Collect data for each configuration
        data_values = {}
        for sf in [200, 600]:
            for method in ["exact", "approximate"]:
                key = f"{method}_{sf}"
                values = []
                subset = df[(df["SF"] == sf) & (df["Method"] == method)]

                for query in queries:
                    query_data = subset[subset["Query"] == query]
                    if not query_data.empty:
                        values.append(query_data[metric].iloc[0])
                    else:
                        values.append(0)
                data_values[key] = values

        # Use log scale for execution time and money cost
        if metric in ["Execution_Time", "Money_Cost"]:
            ax.set_yscale('log')

        # Calculate y_max for consistent spacing
        y_max = max([max(data_values[key]) for key in data_values.keys()])

        # Plot bars (following temp_plot_reasoning.py style)
        legend_labels = ['Exact SF200', 'Approximate SF200', 'Exact SF600', 'Approximate SF600']
        for i, key in enumerate(["exact_200", "approximate_200", "exact_600", "approximate_600"]):
            values = data_values[key]

            # Only show label for first subplot to avoid legend clutter
            label = legend_labels[i] if idx == 0 else ""

            bars = ax.bar(positions[key], values, width=0.19,
                         color=colors[key], alpha=0.8, edgecolor='black',
                         linewidth=1.2, label=label)

            # Add value labels on bars (following temp_plot_reasoning.py style)
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if metric == "Execution_Time":
                    # For execution time, show as integer seconds
                    ax.text(bar.get_x() + bar.get_width()/2., height * 1.25,
                           f'{value:.0f}', ha='center', va='bottom', fontsize=13, rotation=90)
                elif metric == "Money_Cost":
                    # For money cost, show with 2 decimal places
                    ax.text(bar.get_x() + bar.get_width()/2., height * 1.25,
                           f'${value:.2f}', ha='center', va='bottom', fontsize=13, rotation=90)
                else:
                    # For F1 score, show with 2 decimal places
                    ax.text(bar.get_x() + bar.get_width()/2., height + y_max*0.08,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=13, rotation=90)

        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97+idx)}) {ylabel.split(' (')[0]}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(queries)

        # Set y-axis limits (different for log vs linear scale)
        if metric in ["Execution_Time", "Money_Cost"]:
            # For log scale, set reasonable bounds with extra headroom for rotated labels
            y_min = min([min([v for v in data_values[key] if v > 0]) for key in data_values.keys()])
            ax.set_ylim(y_min * 0.5, y_max * 3.5)
        else:
            # For linear scale (F1 Score) with more headroom
            ax.set_ylim(0, y_max * 1.25)

    # Legend positioned very close to figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.01),
               ncol=4, fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    output_path = figures_dir / "lotus_sf_scaling_analysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Figure saved to: {output_path}")

    # Also save as PNG for quick viewing
    plt.savefig(figures_dir / "lotus_sf_scaling_analysis.png", dpi=300, bbox_inches='tight')

    plt.show()


def main():
    """Main function to run the analysis and create plots."""
    print("Loading LOTUS experimental data...")
    df = load_lotus_data()

    if df.empty:
        print("No data loaded. Please check file paths and data availability.")
        return

    print(f"Loaded {len(df)} data points")
    print(f"Scale factors: {sorted(df['SF'].unique())}")
    print(f"Methods: {sorted(df['Method'].unique())}")
    print(f"Queries: {sorted(df['Query'].unique())}")

    # Display summary statistics
    print("\n=== Summary Statistics ===")
    summary = df.groupby(["SF", "Method"]).agg({
        "Execution_Time": "mean",
        "Money_Cost": "mean",
        "F1_Score": "mean"
    }).round(4)
    print(summary)

    # Create the visualization
    print("\nGenerating SF scaling analysis plot...")
    create_sf_scaling_plot(df)


if __name__ == "__main__":
    main()
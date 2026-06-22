"""
Combined Scalability and Memory Plotting for VLDB Conference Paper
Creates a single figure with all scenarios and metrics
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# Configure matplotlib for publication quality
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts for papers
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


class CombinedPlotter:
    """Generates a combined publication-ready figure with scalability and memory plots."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.files_dir = self.base_dir / "files"
        self.figures_dir = self.base_dir / "figures"

        # System colors (consistent across plots)
        self.system_colors = {
            "lotus": "#FF730F",
            "bigquery": "#d62728",
            "palimpzest": "#9467bd",
            "flockmtl": "#8c564b",
            "caesura": "#e377c2",
            "thalamusdb": "#7f7f7f",
        }

        # System markers for distinguishing overlapping lines
        self.system_markers = {
            "lotus": "o",        # circle
            "bigquery": "s",     # square
            "palimpzest": "^",   # triangle up
            "flockmtl": "D",     # diamond
            "caesura": "v",      # triangle down
            "thalamusdb": "P",   # plus (filled)
        }

        # Ordered scenarios (all 5)
        self.scenario_order = ["movie", "ecomm", "animals", "mmqa", "cars"]

    def format_system_name(self, system: str) -> str:
        """Format system name for display in legend."""
        name_map = {
            "bigquery": "BigQuery",
            "lotus": "LOTUS",
            "palimpzest": "Palimpzest",
            "thalamusdb": "ThalamusDB",
            "flockmtl": "FlockMTL",
            "caesura": "Caesura"
        }
        return name_map.get(system.lower(), system.capitalize())

    def format_scenario_name(self, scenario: str) -> str:
        """Format scenario name for display."""
        name_map = {
            "movie": "Movie",
            "ecomm": "E-Commerce",
            "animals": "Wildlife",
            "mmqa": "MMQA",
            "cars": "Cars"
        }
        return name_map.get(scenario.lower(), scenario.upper())

    # ================== Scalability Data Loading ==================
    def load_scalability_data(
        self, scenario: str, model_tag: str = "2.5flash"
    ) -> Tuple[Dict[int, Dict[int, Dict[str, Dict[str, dict]]]], Dict[int, Dict[int, set]]]:
        """Load scalability experiment data including all repeated runs."""
        metrics_dir = self.files_dir / scenario / "metrics"
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        skipped_systems = defaultdict(lambda: defaultdict(set))

        pattern = f"across_system_{model_tag}_sf*"
        sf_dirs = sorted(metrics_dir.glob(pattern))

        all_systems = set()
        for sf_dir in sf_dirs:
            for system_file in sf_dir.glob("*.json"):
                all_systems.add(system_file.stem)

        for sf_dir in sf_dirs:
            try:
                dir_name = sf_dir.name
                if "_repeat" not in dir_name:
                    continue

                parts = dir_name.split("_sf")[-1].split("_repeat")
                scale_factor = int(parts[0])
                repeat_num = int(parts[1])
            except (IndexError, ValueError):
                continue

            systems_present = set()
            for system_file in sf_dir.glob("*.json"):
                system_name = system_file.stem
                systems_present.add(system_name)
                try:
                    with open(system_file, "r") as f:
                        system_data = json.load(f)
                        data[scale_factor][repeat_num][system_name] = system_data
                except Exception:
                    pass

            systems_missing = all_systems - systems_present
            skipped_systems[scale_factor][repeat_num] = systems_missing

        return dict(data), dict(skipped_systems)

    def unify_accuracy_metric(self, metric: Dict) -> Tuple[str, float]:
        """Unify different accuracy metrics into a single format."""
        if "metric_type" in metric and "accuracy" in metric:
            return metric["metric_type"], metric["accuracy"]
        elif "f1_score" in metric:
            return "f1_score", metric["f1_score"]
        elif "relative_error" in metric:
            return "relative_error", (
                1 / (1 + metric["relative_error"])
                if metric["relative_error"] is not None
                else None
            )
        elif "spearman_correlation" in metric:
            return "spearman_correlation", (
                max(0, metric["spearman_correlation"])
                if metric["spearman_correlation"] is not None
                else None
            )
        else:
            return None, None

    def extract_metric(
        self, data: Dict, metric_name: str, default_value: float = None
    ) -> float:
        """Extract a metric value from query data."""
        if metric_name == "quality":
            metric_type, value = self.unify_accuracy_metric(data)
            if value is not None:
                return value
            return default_value
        return data.get(metric_name, default_value)

    def aggregate_across_repeats(
        self, data: Dict[int, Dict[int, Dict[str, Dict[str, dict]]]],
        skipped_systems: Dict[int, Dict[int, set]],
        timeout_threshold: float = 3600.0
    ) -> Tuple[Dict[int, Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]], Dict[int, set]]:
        """Aggregate metrics across repeated runs."""
        aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        systems_exceeded_timeout = defaultdict(set)

        for scale_factor, repeats_data in data.items():
            all_systems = set()
            all_queries = defaultdict(set)

            for repeat_num, systems_data in repeats_data.items():
                for system, queries in systems_data.items():
                    all_systems.add(system)
                    for query_id in queries.keys():
                        all_queries[system].add(query_id)

            for system in all_systems:
                for query_id in all_queries[system]:
                    metric_values = defaultdict(list)

                    for repeat_num, systems_data in repeats_data.items():
                        if system in systems_data and query_id in systems_data[system]:
                            query_data = systems_data[system][query_id]

                            if query_data.get("status") == "success":
                                exec_time = query_data.get("execution_time", 0)
                                money = query_data.get("money_cost", 0)

                                if exec_time <= 0 or money <= 0:
                                    continue

                                for metric_name in ["execution_time", "money_cost", "quality"]:
                                    value = self.extract_metric(query_data, metric_name)
                                    if value is not None and value > 0:
                                        metric_values[metric_name].append(value)

                    for metric_name, values in metric_values.items():
                        if values:
                            mean_val = np.mean(values)
                            std_val = np.std(values, ddof=0) if len(values) > 1 else 0.0
                            aggregated[scale_factor][system][query_id][metric_name] = (mean_val, std_val)

            for system in all_systems:
                for query_id in all_queries[system]:
                    if query_id in aggregated[scale_factor][system]:
                        if "execution_time" in aggregated[scale_factor][system][query_id]:
                            avg_exec_time, _ = aggregated[scale_factor][system][query_id]["execution_time"]
                            if avg_exec_time > timeout_threshold:
                                systems_exceeded_timeout[scale_factor].add(system)
                                break

        return dict(aggregated), dict(systems_exceeded_timeout)

    # ================== Memory Data Loading ==================
    def load_memory_data(
        self, scenario: str, model_tag: str = "2.5flash"
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Load memory consumption data for a scenario."""
        metrics_dir = self.files_dir / scenario / "metrics"
        data = defaultdict(lambda: defaultdict(dict))

        pattern = f"across_system_{model_tag}_sf*"
        sf_dirs = sorted(metrics_dir.glob(pattern))

        for sf_dir in sf_dirs:
            if "_repeat" in sf_dir.name:
                continue

            try:
                sf_str = sf_dir.name.split("_sf")[-1]
                scale_factor = int(sf_str)
            except (IndexError, ValueError):
                continue

            for memory_file in sf_dir.glob("*_memory.json"):
                system_name = memory_file.stem.replace("_memory", "")

                if system_name.lower() == "bigquery":
                    continue

                try:
                    with open(memory_file, "r") as f:
                        memory_data = json.load(f)

                        for query_id, mem_info in memory_data.items():
                            qid = query_id if query_id.startswith('Q') else f'Q{query_id}'
                            peak_memory_mb = mem_info.get('peak_memory_mb')
                            if peak_memory_mb is not None and peak_memory_mb > 0:
                                data[scale_factor][system_name][qid] = peak_memory_mb / 1024.0
                except Exception as e:
                    print(f"Warning: Error loading {memory_file}: {e}")

        return dict(data)

    def find_common_queries_memory(
        self, data: Dict[int, Dict[str, Dict[str, float]]]
    ) -> set:
        """Find queries common across all systems and scale factors for memory data."""
        if not data:
            return set()

        sf_common_queries = []
        for sf in sorted(data.keys()):
            systems_in_sf = list(data[sf].keys())
            if not systems_in_sf:
                continue

            queries_per_system = [set(data[sf][system].keys()) for system in systems_in_sf]
            if queries_per_system:
                common_in_sf = set.intersection(*queries_per_system)
                sf_common_queries.append(common_in_sf)

        if sf_common_queries:
            return set.intersection(*sf_common_queries)
        return set()

    def find_common_queries_scalability(
        self, data: Dict[int, Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]]
    ) -> set:
        """Find common queries across all systems and scale factors for scalability data."""
        all_scale_factors = sorted(data.keys())
        sf_common_queries = []

        for sf in all_scale_factors:
            systems_in_sf = list(data[sf].keys())
            if not systems_in_sf:
                continue

            queries_per_system = [set(data[sf][system].keys()) for system in systems_in_sf]
            if queries_per_system:
                common_in_sf = set.intersection(*queries_per_system)
                sf_common_queries.append(common_in_sf)

        return set.intersection(*sf_common_queries) if sf_common_queries else set()

    # ================== Combined Plotting ==================
    def plot_combined_figure(
        self,
        scalability_data: Dict[str, Tuple[Dict, Dict]],
        memory_data: Dict[str, Dict],
        output_dir: Path,
        model_tag: str = "2.5flash"
    ):
        """
        Create a combined figure with 4 rows (Time, Cost, Quality, Memory) x 4 columns (scenarios).
        """
        scenarios = [s for s in self.scenario_order if s in scalability_data or s in memory_data]
        n_scenarios = len(scenarios)

        if n_scenarios == 0:
            print("No scenarios to plot")
            return

        # Figure parameters (larger fonts)
        FONTSIZE_LABEL = 18
        FONTSIZE_TICK = 16
        FONTSIZE_LEGEND = 16
        FONTSIZE_TITLE = 18
        LINEWIDTH = 3.5
        MARKERSIZE = 12
        ERROR_ALPHA = 0.2

        # Figure size
        fig_width = 3.0 * n_scenarios  # Width per column
        fig_height = 1.6 * 4  # Row height

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create GridSpec - no vertical space, moderate horizontal space
        gs = GridSpec(4, n_scenarios, figure=fig,
                      hspace=0.0, wspace=0.30,
                      left=0.05, right=0.995, bottom=0.07, top=0.85)

        # Metric labels for y-axis
        metric_labels = ["Time (s)", "Cost ($)", "Quality", "Memory (GB)"]
        metric_keys = ["execution_time", "money_cost", "quality", "memory"]

        # Collect all systems across all scenarios
        all_systems = set()
        for scenario in scenarios:
            if scenario in scalability_data:
                agg_data, _ = scalability_data[scenario]
                for sf_data in agg_data.values():
                    all_systems.update(sf_data.keys())
            if scenario in memory_data:
                for sf_data in memory_data[scenario].values():
                    all_systems.update(sf_data.keys())

        # Track which systems timed out (for scalability)
        scenario_timeout_info = {}
        for scenario in scenarios:
            if scenario in scalability_data:
                _, systems_exceeded_timeout = scalability_data[scenario]
                scenario_timeout_info[scenario] = systems_exceeded_timeout

        # Create axes array
        axes = []
        for row in range(4):
            row_axes = []
            for col in range(n_scenarios):
                ax = fig.add_subplot(gs[row, col])
                row_axes.append(ax)
            axes.append(row_axes)

        # Plot each cell
        for col, scenario in enumerate(scenarios):
            # Get scale factors for this scenario
            if scenario in scalability_data:
                agg_data, systems_exceeded_timeout = scalability_data[scenario]
                all_scale_factors = sorted(agg_data.keys())
                common_queries_scal = self.find_common_queries_scalability(agg_data)
            else:
                agg_data = {}
                systems_exceeded_timeout = {}
                all_scale_factors = []
                common_queries_scal = set()

            if scenario in memory_data:
                mem_data = memory_data[scenario]
                mem_scale_factors = sorted(mem_data.keys())
                common_queries_mem = self.find_common_queries_memory(mem_data)
            else:
                mem_data = {}
                mem_scale_factors = []
                common_queries_mem = set()

            # Track timeout scale factors per system
            system_timeout_sf = {}
            for sf in all_scale_factors:
                if sf in systems_exceeded_timeout:
                    for system in systems_exceeded_timeout[sf]:
                        if system not in system_timeout_sf:
                            system_timeout_sf[system] = sf

            # Plot each row (metric)
            for row, (metric_key, metric_label) in enumerate(zip(metric_keys, metric_labels)):
                ax = axes[row][col]

                if metric_key == "memory":
                    # Memory plot
                    self._plot_memory_cell(
                        ax, scenario, mem_data, mem_scale_factors, common_queries_mem,
                        LINEWIDTH, MARKERSIZE
                    )
                else:
                    # Scalability plot
                    self._plot_scalability_cell(
                        ax, scenario, metric_key, agg_data, all_scale_factors,
                        common_queries_scal, systems_exceeded_timeout, system_timeout_sf,
                        LINEWIDTH, MARKERSIZE, ERROR_ALPHA
                    )

                # Styling
                ax.grid(True, alpha=0.25, linewidth=0.8, linestyle='-', color='gray')
                ax.set_axisbelow(True)
                ax.tick_params(labelsize=FONTSIZE_TICK, width=1.5, length=4, pad=2)

                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

                # Log scale for execution time and memory
                if metric_key == "execution_time" or metric_key == "memory":
                    ax.set_yscale("log")
                    # For memory, only show powers of 10 (not 2x10^0, etc.)
                    if metric_key == "memory":
                        from matplotlib.ticker import LogLocator, NullLocator, FuncFormatter
                        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
                        ax.yaxis.set_minor_locator(NullLocator())
                        # Format as 10^n
                        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else ''))

                # For cost row, don't show 0 as a tick label but keep axis starting from 0
                if metric_key == "money_cost":
                    from matplotlib.ticker import MaxNLocator, FuncFormatter
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    # Hide 0 tick label
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '' if x == 0 else f'{x:.1f}' if x < 1 else f'{x:.0f}'))

                # X-axis: set ticks for all rows but labels only on bottom row
                scale_factors_to_use = mem_scale_factors if metric_key == "memory" else all_scale_factors
                if len(scale_factors_to_use) >= 3:
                    ticks = [scale_factors_to_use[0], scale_factors_to_use[-2], scale_factors_to_use[-1]]
                else:
                    ticks = scale_factors_to_use
                ax.set_xticks(ticks)

                if row == 3:
                    ax.set_xlabel("Scale Factor", fontsize=FONTSIZE_LABEL, fontweight='bold')
                    if scenario in ["movie", "cars"]:
                        labels = [f'{int(sf/1000)}k' if sf >= 1000 else str(sf) for sf in ticks]
                    else:
                        labels = [str(sf) for sf in ticks]
                    ax.set_xticklabels(labels)
                else:
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', length=0)

                # Y-axis label only on leftmost column, but show tick labels on all
                if col == 0:
                    ax.set_ylabel(metric_label, fontsize=FONTSIZE_LABEL, fontweight='bold')
                    # Align all Y labels at the same x position (more negative = more space from numbers)
                    ax.yaxis.set_label_coords(-0.22, 0.5)
                    # Move Quality down slightly
                    if metric_key == "quality":
                        ax.yaxis.set_label_coords(-0.22, 0.55)
                    # Move Memory label down to avoid overlap with Quality
                    if metric_key == "memory":
                        ax.yaxis.set_label_coords(-0.22, 0.35)
                # Show Y-axis tick labels for all subfigures (different scales per scenario)

                # Scenario names on top row only
                if row == 0:
                    ax.set_title(self.format_scenario_name(scenario),
                                fontsize=FONTSIZE_TITLE, fontweight='bold', pad=4)

        # Create shared legend at top
        legend_handles = []
        legend_labels = []

        for system in sorted(all_systems):
            color = self.system_colors.get(system, None)
            marker = self.system_markers.get(system, "o")
            legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=LINEWIDTH,
                                            marker=marker, markersize=MARKERSIZE-2,
                                            markeredgewidth=0.8, markeredgecolor='white'))
            legend_labels.append(self.format_system_name(system))

        # Add timeout marker explanation
        timeout_marker = plt.Line2D([0], [0], marker='x', color='gray', linewidth=0,
                                    markersize=MARKERSIZE-1, markeredgewidth=3.0)
        legend_handles.append(timeout_marker)
        legend_labels.append('1 Hour Timeout')

        fig.legend(legend_handles, legend_labels,
                  loc='upper center',
                  bbox_to_anchor=(0.5, 0.98),
                  ncol=len(legend_handles),
                  fontsize=FONTSIZE_LEGEND,
                  frameon=False,
                  columnspacing=1.0,
                  handlelength=1.5,
                  handletextpad=0.4)

        # Save figure
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "scalability_memory_combined.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved combined figure: {output_file}")
        plt.close()

    def _plot_scalability_cell(
        self, ax, scenario, metric_key, data, all_scale_factors, common_queries,
        systems_exceeded_timeout, system_timeout_sf, linewidth, markersize, error_alpha
    ):
        """Plot a single scalability cell (Time, Cost, or Quality)."""
        system_data = defaultdict(lambda: {"sfs": [], "means": [], "stds": []})

        for sf in all_scale_factors:
            if sf not in data:
                continue
            for system, queries in data[sf].items():
                # Skip if system timed out at previous scale factor
                if system in system_timeout_sf and sf > system_timeout_sf[system]:
                    continue

                mean_values = []
                std_values = []

                for query_id, metrics_dict in queries.items():
                    if query_id not in common_queries:
                        continue

                    if metric_key in metrics_dict:
                        mean_val, std_val = metrics_dict[metric_key]
                        mean_values.append(mean_val)
                        std_values.append(std_val)

                if mean_values:
                    system_data[system]["sfs"].append(sf)
                    system_data[system]["means"].append(np.mean(mean_values))
                    combined_std = np.mean(std_values) if std_values else 0
                    system_data[system]["stds"].append(combined_std)

        # Plot each system
        for system in sorted(system_data.keys()):
            color = self.system_colors.get(system, None)
            marker = self.system_markers.get(system, "o")
            sfs = np.array(system_data[system]["sfs"])
            means = np.array(system_data[system]["means"])
            stds = np.array(system_data[system]["stds"])

            if len(sfs) == 0:
                continue

            # Identify timeout points
            timeout_mask = np.array([
                sf in systems_exceeded_timeout and system in systems_exceeded_timeout[sf]
                for sf in sfs
            ])
            normal_mask = ~timeout_mask

            normal_sfs = sfs[normal_mask]
            normal_means = means[normal_mask]
            timeout_sfs = sfs[timeout_mask]
            timeout_means = means[timeout_mask]

            # Plot line
            ax.plot(sfs, means, linestyle="-", linewidth=linewidth, color=color, alpha=0.95)

            # Plot normal points
            if len(normal_sfs) > 0:
                ax.scatter(normal_sfs, normal_means, marker=marker, s=markersize**2,
                          color=color, alpha=0.95, edgecolors='white', linewidths=0.8, zorder=4)

            # Plot timeout points
            if len(timeout_sfs) > 0:
                ax.scatter(timeout_sfs, timeout_means, marker="x", s=150,
                          linewidths=3.5, color=color, alpha=0.95, zorder=5)

            # Error region
            ax.fill_between(sfs, means - stds, means + stds, color=color, alpha=error_alpha)

    def _plot_memory_cell(
        self, ax, scenario, data, scale_factors, common_queries, linewidth, markersize
    ):
        """Plot a single memory cell with min-max error bars."""
        system_data = defaultdict(lambda: {"sfs": [], "memories": [], "mem_min": [], "mem_max": []})

        for sf in scale_factors:
            if sf not in data:
                continue
            for system, queries in data[sf].items():
                memories = []
                for query_id, memory_gb in queries.items():
                    if query_id in common_queries:
                        memories.append(memory_gb)

                if memories:
                    system_data[system]["sfs"].append(sf)
                    system_data[system]["memories"].append(np.mean(memories))
                    system_data[system]["mem_min"].append(np.min(memories))
                    system_data[system]["mem_max"].append(np.max(memories))

        # Plot each system with error bars
        for system in sorted(system_data.keys()):
            color = self.system_colors.get(system, None)
            marker = self.system_markers.get(system, "o")
            sfs = np.array(system_data[system]["sfs"])
            memories = np.array(system_data[system]["memories"])
            mem_min = np.array(system_data[system]["mem_min"])
            mem_max = np.array(system_data[system]["mem_max"])

            if len(sfs) == 0:
                continue

            # Calculate error bars (asymmetric: lower = mean - min, upper = max - mean)
            yerr_lower = memories - mem_min
            yerr_upper = mem_max - memories

            ax.errorbar(sfs, memories, yerr=[yerr_lower, yerr_upper],
                       marker=marker, linestyle="-", linewidth=linewidth,
                       markersize=markersize, color=color, alpha=0.95,
                       markeredgewidth=0.8, markeredgecolor='white',
                       capsize=3, capthick=1.5, elinewidth=1.5)

    def plot_all(self, model_tag: str = "2.5flash"):
        """Load all data and generate the combined figure."""
        print("Loading scalability data...")
        scalability_data = {}
        for scenario in self.scenario_order:
            raw_data, skipped = self.load_scalability_data(scenario, model_tag)
            if raw_data:
                agg_data, timeout_info = self.aggregate_across_repeats(raw_data, skipped)
                if agg_data:
                    scalability_data[scenario] = (agg_data, timeout_info)
                    print(f"  {scenario}: {len(agg_data)} scale factors")

        print("\nLoading memory data...")
        memory_data = {}
        for scenario in self.scenario_order:
            mem_data = self.load_memory_data(scenario, model_tag)
            if mem_data:
                memory_data[scenario] = mem_data
                print(f"  {scenario}: {len(mem_data)} scale factors")

        print("\nGenerating combined figure...")
        self.plot_combined_figure(scalability_data, memory_data, self.figures_dir, model_tag)

        print("\nDone!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate combined scalability and memory plot for VLDB paper"
    )
    parser.add_argument(
        "--model-tag",
        default="2.5flash",
        help="Model tag (default: 2.5flash)",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory (default: .)"
    )

    args = parser.parse_args()

    plotter = CombinedPlotter(base_dir=args.base_dir)
    plotter.plot_all(model_tag=args.model_tag)


if __name__ == "__main__":
    main()

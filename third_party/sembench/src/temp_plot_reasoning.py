import json
import matplotlib.pyplot as plt
import numpy as np

# Set professional academic style with large fonts for paper
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

def load_q7_data():
    """Load Q7 metrics from all lotus experiment files."""
    data_files = {
        'Gemini-2.5-Flash\n(Disable)': '/home/jiale/MMBench-System/files/movie/metrics/lotus/lotus_2.5flash_disable.json',
        'Gemini-2.5-Flash\n(Low)': '/home/jiale/MMBench-System/files/movie/metrics/lotus/lotus_2.5flash_low.json',
        'GPT-5-Mini\n(Minimal)': '/home/jiale/MMBench-System/files/movie/metrics/lotus/lotus_5mini_minimal.json',
        'GPT-5-Mini\n(Medium)': '/home/jiale/MMBench-System/files/movie/metrics/lotus/lotus_5mini_medium.json',
    }

    q7_metrics = {}

    for config, filepath in data_files.items():
        with open(filepath, 'r') as f:
            data = json.load(f)
            q7_data = data['Q7']
            q7_metrics[config] = {
                'latency': q7_data['execution_time'],
                'cost': q7_data['money_cost'],
                'f1_score': q7_data['f1_score'],
                'precision': q7_data['precision'],
                'recall': q7_data['recall']
            }

    return q7_metrics

def create_reasoning_comparison_figure():
    """Create a professional figure comparing reasoning vs non-reasoning models."""

    # Load data
    q7_data = load_q7_data()

    # No x-axis labels needed since legend provides all information
    configs = ['', '', '', '']
    latencies = [q7_data[k]['latency'] for k in q7_data.keys()]
    costs = [q7_data[k]['cost'] for k in q7_data.keys()]
    f1_scores = [q7_data[k]['f1_score'] for k in q7_data.keys()]

    # Color scheme: similar colors for same model, different shades for reasoning levels
    # gemini-2.5-flash: blues, gpt-5-mini: oranges
    colors = ['#2E5984', '#4A90E2', '#D2691E', '#FF8C42']  # Dark blue, light blue, dark orange, light orange
    legend_labels = ['gemini-2.5-flash (disable)', 'gemini-2.5-flash (low)', 'gpt-5-mini (minimal)', 'gpt-5-mini (medium)']

    # Create figure with 3 subplots - compact size with large fonts for paper
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Execution Time
    bars1 = ax1.bar(range(len(configs)), latencies, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.2, width=0.7, label=legend_labels)
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('(a) Execution Time')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=0, ha='center')

    # Use linear scale but format y-axis nicely
    ax1.set_ylim(0, max(latencies) * 1.15)

    # Add value labels on bars - larger font
    for i, (bar, latency) in enumerate(zip(bars1, latencies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(latencies)*0.02,
                f'{latency:.0f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # 2. Monetary Cost
    bars2 = ax2.bar(range(len(configs)), costs, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.2, width=0.7)
    ax2.set_ylabel('Monetary Cost (USD)')
    ax2.set_title('(b) Monetary Cost')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=0, ha='center')
    ax2.set_ylim(0, max(costs) * 1.15)

    # Add value labels on bars - larger font
    for i, (bar, cost) in enumerate(zip(bars2, costs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.02,
                f'{cost:.1f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # 3. F1 Score
    bars3 = ax3.bar(range(len(configs)), f1_scores, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.2, width=0.7)
    ax3.set_ylabel('F1 Score')
    ax3.set_title('(c) F1 Score')
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(configs, rotation=0, ha='center')
    ax3.set_ylim(0, max(f1_scores) * 1.2)

    # Add value labels on bars - larger font
    for i, (bar, f1) in enumerate(zip(bars3, f1_scores)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(f1_scores)*0.02,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Add baseline comparison lines (dashed horizontal lines for baseline values)
    baseline_latency = latencies[0]
    baseline_cost = costs[0]
    baseline_f1 = f1_scores[0]

    ax1.axhline(y=baseline_latency, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax2.axhline(y=baseline_cost, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax3.axhline(y=baseline_f1, color='gray', linestyle='--', alpha=0.7, linewidth=2)

    # Add shared legend positioned at the bottom - single row, same order as bars
    fig.legend(bars1, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=4, fontsize=18, frameon=False)

    # Remove x-axis tick marks (vertical lines under bars)
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', length=0)

    # Adjust layout with minimal space for legend and compact design
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.25)

    return fig

def save_figure():
    """Generate and save the reasoning comparison figure."""
    fig = create_reasoning_comparison_figure()

    # Save in multiple formats for academic paper
    output_path = '/home/jiale/MMBench-System/figures/lotus_reasoning_comparison'
    fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_path}.eps', dpi=300, bbox_inches='tight', facecolor='white')

    print(f"Figure saved as:")
    print(f"  - {output_path}.png")
    print(f"  - {output_path}.pdf")
    print(f"  - {output_path}.eps")

    # Display the figure
    plt.show()

    # Print summary statistics
    q7_data = load_q7_data()
    print("\nQ7 Performance Summary:")
    print("=" * 60)
    baseline = q7_data['Gemini-2.5-Flash\n(Disable)']

    for config, metrics in q7_data.items():
        latency_ratio = metrics['latency'] / baseline['latency']
        cost_ratio = metrics['cost'] / baseline['cost']
        f1_improvement = metrics['f1_score'] - baseline['f1_score']

        print(f"{config.replace(chr(10), ' '):<25}")
        print(f"  Latency: {metrics['latency']:>8.1f}s ({latency_ratio:>5.1f}× baseline)")
        print(f"  Cost:    ${metrics['cost']:>8.2f} ({cost_ratio:>5.1f}× baseline)")
        print(f"  F1:      {metrics['f1_score']:>8.3f} ({f1_improvement:>+6.3f} vs baseline)")
        print()

if __name__ == "__main__":
    save_figure()
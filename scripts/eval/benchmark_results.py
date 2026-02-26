"""
Benchmark Results Analysis for MARLISE.

Reads benchmark metrics CSVs and produces statistical analysis.
Usage: python scripts/eval/benchmark_results.py [--results_dir results/benchmark/small]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def load_all_metrics(results_dir: str) -> dict[str, list[pd.DataFrame]]:
    """Load all metric CSVs grouped by algorithm name."""
    algorithm_data = {}
    pattern = os.path.join(results_dir, "*_iter*_metrics.csv")

    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        # Parse algorithm name from filename: {alg_name}_iter{N}_metrics.csv
        parts = filename.rsplit('_iter', 1)
        alg_name = parts[0]

        if alg_name not in algorithm_data:
            algorithm_data[alg_name] = []

        df = pd.read_csv(filepath)
        algorithm_data[alg_name].append(df)

    return algorithm_data


def compute_statistics(algorithm_data: dict[str, list[pd.DataFrame]], sla_threshold_ms: float = 0.250) -> pd.DataFrame:
    """Compute per-algorithm summary statistics across iterations."""
    results = []

    for alg_name, dfs in algorithm_data.items():
        # Aggregate per-iteration metrics
        iter_metrics = []
        for df in dfs:
            rt = df['response_time']
            cpu_pct = df['cpu_percentage']
            reward = df['reward']

            iter_metrics.append({
                'mean_rt': rt.mean(),
                'p95_rt': rt.quantile(0.95),
                'p99_rt': rt.quantile(0.99),
                'sla_violations': (rt > sla_threshold_ms).sum() / len(rt) * 100,
                'mean_cpu_util': cpu_pct.mean(),
                'std_cpu_util': cpu_pct.std(),
                'mean_reward': reward.mean(),
                'total_reward': reward.sum(),
                'mean_decision_latency': df['decision_latency_ms'].mean(),
                'p95_decision_latency': df['decision_latency_ms'].quantile(0.95),
                'total_hpa_events': _count_hpa_events(df),
                'total_vpa_deltas': _count_vpa_changes(df),
            })

        iter_df = pd.DataFrame(iter_metrics)

        # Compute mean and 95% CI across iterations
        n = len(iter_df)
        for metric in iter_df.columns:
            values = iter_df[metric]
            mean = values.mean()
            std = values.std()
            ci_95 = 1.96 * std / np.sqrt(n) if n > 1 else 0

            results.append({
                'algorithm': alg_name,
                'metric': metric,
                'mean': mean,
                'std': std,
                'ci_95_lower': mean - ci_95,
                'ci_95_upper': mean + ci_95,
                'n_iterations': n,
            })

    return pd.DataFrame(results)


def _count_hpa_events(df: pd.DataFrame) -> int:
    """Count replica count changes."""
    if 'replica_count' not in df.columns:
        return 0
    replicas = df.groupby('step')['replica_count'].first()
    return (replicas.diff().abs() > 0).sum()


def _count_vpa_changes(df: pd.DataFrame) -> int:
    """Count CPU limit changes."""
    changes = 0
    for agent_id in df['agent_id'].unique():
        agent_df = df[df['agent_id'] == agent_id]
        cpu_limits = agent_df['cpu_limit']
        changes += (cpu_limits.diff().abs() > 0).sum()
    return changes


def per_phase_analysis(algorithm_data: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """Analyze metrics broken down by load phase."""
    results = []
    for alg_name, dfs in algorithm_data.items():
        combined = pd.concat(dfs, ignore_index=True)
        for phase in combined['phase'].unique():
            phase_data = combined[combined['phase'] == phase]
            results.append({
                'algorithm': alg_name,
                'phase': phase,
                'mean_rt': phase_data['response_time'].mean(),
                'p95_rt': phase_data['response_time'].quantile(0.95),
                'mean_cpu_util': phase_data['cpu_percentage'].mean(),
                'mean_reward': phase_data['reward'].mean(),
                'sla_violation_pct': (phase_data['response_time'] > 0.250).sum() / len(phase_data) * 100,
            })
    return pd.DataFrame(results)


def print_summary(stats_df: pd.DataFrame, phase_df: pd.DataFrame):
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Pivot to show algorithm comparison
    pivot = stats_df.pivot(index='metric', columns='algorithm', values='mean')
    print("\nPer-Algorithm Metrics (mean across iterations):")
    print(pivot.to_string(float_format='{:.4f}'.format))

    print("\n\nPer-Phase Analysis:")
    print(phase_df.to_string(index=False, float_format='{:.4f}'.format))

    # Highlight key comparisons
    print("\n\nKey Metrics Comparison:")
    key_metrics = ['mean_rt', 'p95_rt', 'sla_violations', 'mean_cpu_util', 'mean_decision_latency']
    for metric in key_metrics:
        metric_data = stats_df[stats_df['metric'] == metric][['algorithm', 'mean', 'ci_95_lower', 'ci_95_upper']]
        if not metric_data.empty:
            print(f"\n  {metric}:")
            for _, row in metric_data.iterrows():
                print(f"    {row['algorithm']:20s}: {row['mean']:.4f} [{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}]")


def save_results(stats_df: pd.DataFrame, phase_df: pd.DataFrame, output_dir: str):
    """Save analysis results to CSV."""
    stats_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
    phase_df.to_csv(os.path.join(output_dir, 'per_phase_analysis.csv'), index=False)
    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="MARLISE Benchmark Results Analysis")
    parser.add_argument('--results_dir', type=str, default='results/benchmark/small')
    parser.add_argument('--sla_threshold', type=float, default=0.250,
                        help="SLA threshold in seconds (default: 0.250)")
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist")
        return

    algorithm_data = load_all_metrics(args.results_dir)
    if not algorithm_data:
        print(f"No metrics files found in {args.results_dir}")
        return

    print(f"Loaded data for algorithms: {list(algorithm_data.keys())}")

    stats_df = compute_statistics(algorithm_data, sla_threshold_ms=args.sla_threshold)
    phase_df = per_phase_analysis(algorithm_data)

    print_summary(stats_df, phase_df)
    save_results(stats_df, phase_df, args.results_dir)


if __name__ == '__main__':
    main()

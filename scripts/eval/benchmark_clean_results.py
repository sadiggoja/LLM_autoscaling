"""
Benchmark Results Analysis (CLEAN) for MARLISE.

Same as benchmark_results.py but with two key differences:
  1. Failed-probe rows (response_time == 2.0) are EXCLUDED before computing any
     metric — so mean_rt, p95_rt, CPU utilisation, reward, etc. reflect only
     successful probe steps.
  2. Probe failure rate is reported separately, both overall and per-phase, so
     infrastructure noise stays visible without contaminating the algorithm
     metrics.

Usage:
    python scripts/eval/benchmark_clean_results.py [--results_dir results/benchmark/small]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


FAIL_VALUE = 2.0  # response_time fallback when make_request returns None


def load_all_metrics(results_dir: str) -> dict[str, list[pd.DataFrame]]:
    """Load all metric CSVs grouped by algorithm name."""
    algorithm_data = {}
    pattern = os.path.join(results_dir, "*_iter*_metrics.csv")

    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        parts = filename.rsplit('_iter', 1)
        alg_name = parts[0]
        algorithm_data.setdefault(alg_name, []).append(pd.read_csv(filepath))

    return algorithm_data


def split_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Return (clean_df, n_total, n_failed). clean_df excludes rt == FAIL_VALUE rows."""
    n_total = len(df)
    failed_mask = df['response_time'] == FAIL_VALUE
    n_failed = int(failed_mask.sum())
    clean_df = df.loc[~failed_mask].copy()
    return clean_df, n_total, n_failed


def compute_statistics(algorithm_data: dict[str, list[pd.DataFrame]],
                       sla_threshold_ms: float = 0.250) -> pd.DataFrame:
    """Per-algorithm summary stats, computed on clean rows only.

    fail_rate is reported as the fraction of probe failures (rt == 2.0) over the
    full unfiltered iteration; everything else is computed on the clean subset.
    """
    results = []

    for alg_name, dfs in algorithm_data.items():
        iter_metrics = []
        for df in dfs:
            clean, n_total, n_failed = split_clean(df)
            fail_rate_pct = 100.0 * n_failed / n_total if n_total else 0.0

            if clean.empty:
                continue

            rt = clean['response_time']
            cpu_pct = clean['cpu_percentage']
            reward = clean['reward']

            iter_metrics.append({
                'fail_rate_pct': fail_rate_pct,
                'mean_rt': rt.mean(),
                'p95_rt': rt.quantile(0.95),
                'p99_rt': rt.quantile(0.99),
                'sla_violations_clean': (rt > sla_threshold_ms).sum() / len(rt) * 100,
                'mean_cpu_util': cpu_pct.mean(),
                'std_cpu_util': cpu_pct.std(),
                'mean_reward': reward.mean(),
                'total_reward': reward.sum(),
                'mean_decision_latency': clean['decision_latency_ms'].mean(),
                'p95_decision_latency': clean['decision_latency_ms'].quantile(0.95),
                'total_hpa_events': _count_hpa_events(clean),
                'total_vpa_deltas': _count_vpa_changes(clean),
            })

        if not iter_metrics:
            continue

        iter_df = pd.DataFrame(iter_metrics)
        n = len(iter_df)
        for metric in iter_df.columns:
            values = iter_df[metric]
            mean = values.mean()
            std = values.std() if n > 1 else 0.0
            ci_95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            results.append({
                'algorithm': alg_name,
                'metric': metric,
                'mean': mean,
                'median': values.median(),
                'std': std,
                'ci_95_lower': mean - ci_95,
                'ci_95_upper': mean + ci_95,
                'n_iterations': n,
            })

    return pd.DataFrame(results)


def _count_hpa_events(df: pd.DataFrame) -> int:
    if 'replica_count' not in df.columns or df.empty:
        return 0
    replicas = df.groupby('step')['replica_count'].first()
    return int((replicas.diff().abs() > 0).sum())


def _count_vpa_changes(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    changes = 0
    for agent_id in df['agent_id'].unique():
        agent_df = df[df['agent_id'] == agent_id]
        cpu_limits = agent_df['cpu_limit']
        changes += int((cpu_limits.diff().abs() > 0).sum())
    return changes


def per_phase_analysis(algorithm_data: dict[str, list[pd.DataFrame]],
                       sla_threshold_ms: float = 0.250) -> pd.DataFrame:
    """Per-(algorithm, phase) metrics aggregated as mean ± std across iterations.

    For each iteration × phase, fail_rate is computed on the FULL phase rows
    (failed_count / total_count), and all other metrics are computed on the
    clean subset of that phase.
    """
    per_iter_rows = []
    for alg_name, dfs in algorithm_data.items():
        for df in dfs:
            for phase in df['phase'].unique():
                pdf_full = df[df['phase'] == phase]
                if pdf_full.empty:
                    continue

                n_total = len(pdf_full)
                failed_mask = pdf_full['response_time'] == FAIL_VALUE
                n_failed = int(failed_mask.sum())
                fail_rate = 100.0 * n_failed / n_total

                pdf = pdf_full.loc[~failed_mask]
                if pdf.empty:
                    per_iter_rows.append({
                        'algorithm': alg_name, 'phase': phase,
                        'fail_rate_pct': fail_rate,
                        'mean_rt': float('nan'), 'p95_rt': float('nan'),
                        'mean_cpu_util': float('nan'), 'mean_reward': float('nan'),
                        'sla_violation_pct': float('nan'),
                    })
                    continue

                per_iter_rows.append({
                    'algorithm': alg_name,
                    'phase': phase,
                    'fail_rate_pct': fail_rate,
                    'mean_rt': pdf['response_time'].mean(),
                    'p95_rt': pdf['response_time'].quantile(0.95),
                    'mean_cpu_util': pdf['cpu_percentage'].mean(),
                    'mean_reward': pdf['reward'].mean(),
                    'sla_violation_pct': (pdf['response_time'] > sla_threshold_ms).sum() / len(pdf) * 100,
                })

    iter_df = pd.DataFrame(per_iter_rows)
    if iter_df.empty:
        return iter_df

    metric_cols = ['fail_rate_pct', 'mean_rt', 'p95_rt', 'mean_cpu_util',
                   'mean_reward', 'sla_violation_pct']
    agg = (iter_df.groupby(['algorithm', 'phase'])[metric_cols]
                  .agg(['mean', 'median', 'std', 'count'])
                  .reset_index())
    agg.columns = [f"{a}_{b}" if b else a for a, b in agg.columns]
    return agg


def _fmt_mean_std(mean, std) -> str:
    if pd.isna(mean):
        return "NaN"
    s = 0.0 if pd.isna(std) else std
    return f"{mean:.4f} ± {s:.4f}"


def print_summary(stats_df: pd.DataFrame, phase_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS — CLEAN (probe failures excluded from metrics)")
    print("=" * 80)

    if stats_df.empty:
        print("\n(no data)")
        return

    display_df = stats_df.copy()
    display_df['display'] = [_fmt_mean_std(m, s) for m, s in zip(display_df['mean'], display_df['std'])]
    pivot = display_df.pivot(index='metric', columns='algorithm', values='display')
    n_iter = int(stats_df['n_iterations'].iloc[0]) if not stats_df.empty else 0
    print(f"\nPer-Algorithm Metrics (mean ± std across {n_iter} iteration(s)):")
    print(pivot.to_string())

    # Median listing — useful when iterations have heavy tails from infra noise
    print(f"\nPer-Algorithm Medians (across {n_iter} iteration(s)):")
    median_pivot = stats_df.pivot(index='metric', columns='algorithm', values='median')
    print(median_pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n\nPer-Phase Analysis (clean metrics + per-phase fail_rate, mean ± std across iters):")
    if phase_df.empty:
        print("  (no data)")
        return

    display_phase = phase_df[['algorithm', 'phase']].copy()
    metric_cols = [c[:-5] for c in phase_df.columns if c.endswith('_mean')]
    for m in metric_cols:
        display_phase[m] = [_fmt_mean_std(mean, std)
                            for mean, std in zip(phase_df[f'{m}_mean'], phase_df[f'{m}_std'])]
    print(display_phase.to_string(index=False))

    # Compact per-phase fail-rate table on its own
    print("\n\nPer-Phase Probe Failure Rate (mean ± std across iters):")
    fr = phase_df[['algorithm', 'phase', 'fail_rate_pct_mean', 'fail_rate_pct_std',
                   'fail_rate_pct_median']].copy()
    fr['fail_rate_pct'] = [_fmt_mean_std(m, s) for m, s in zip(fr['fail_rate_pct_mean'],
                                                                fr['fail_rate_pct_std'])]
    fr = fr[['algorithm', 'phase', 'fail_rate_pct', 'fail_rate_pct_median']]
    fr.columns = ['algorithm', 'phase', 'fail_rate (mean ± std)', 'fail_rate (median)']
    fr['fail_rate (median)'] = fr['fail_rate (median)'].apply(lambda x: f"{x:.2f}%")
    print(fr.to_string(index=False))

    # Headline comparison table
    print("\n\nKey Metrics (clean) — mean ± std across iterations:")
    key_metrics = ['fail_rate_pct', 'mean_rt', 'p95_rt', 'sla_violations_clean',
                   'mean_cpu_util', 'mean_reward']
    for metric in key_metrics:
        metric_data = stats_df[stats_df['metric'] == metric][['algorithm', 'mean', 'median', 'std']]
        if metric_data.empty:
            continue
        print(f"\n  {metric}:")
        for _, row in metric_data.iterrows():
            print(f"    {row['algorithm']:30s}: {_fmt_mean_std(row['mean'], row['std'])}  "
                  f"(median {row['median']:.4f})")


def save_results(stats_df: pd.DataFrame, phase_df: pd.DataFrame, output_dir: str):
    stats_df.to_csv(os.path.join(output_dir, 'summary_statistics_clean.csv'), index=False)
    phase_df.to_csv(os.path.join(output_dir, 'per_phase_analysis_clean.csv'), index=False)
    print(f"\nResults saved to {output_dir}/")
    print(f"  - summary_statistics_clean.csv")
    print(f"  - per_phase_analysis_clean.csv")


def main():
    parser = argparse.ArgumentParser(
        description="MARLISE Benchmark Analysis (clean: excludes failed probes)")
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
    for alg, dfs in algorithm_data.items():
        total_rows = sum(len(d) for d in dfs)
        total_fails = sum(int((d['response_time'] == FAIL_VALUE).sum()) for d in dfs)
        print(f"  {alg:<30s}  {len(dfs)} iters  {total_rows} rows  "
              f"{total_fails} failed probes ({100*total_fails/total_rows:.2f}%)")

    stats_df = compute_statistics(algorithm_data, sla_threshold_ms=args.sla_threshold)
    phase_df = per_phase_analysis(algorithm_data, sla_threshold_ms=args.sla_threshold)

    print_summary(stats_df, phase_df)
    save_results(stats_df, phase_df, args.results_dir)


if __name__ == '__main__':
    main()

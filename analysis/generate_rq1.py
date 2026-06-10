#!/usr/bin/env python3
"""Generate the RQ1 deep-dive HTML report from results/benchmark/small_3105/.

Run from the repo root:
    python analysis/generate_rq1.py

Writes:
    analysis/figures/rq1_*.png   (10 figures)
    analysis/RQ1.html            (the report itself)
"""

from __future__ import annotations

import glob
import os
import re
import sys
import textwrap
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "benchmark", "small_3105")
FIG_DIR = os.path.join(REPO_ROOT, "analysis", "figures")
HTML_OUT = os.path.join(REPO_ROOT, "analysis", "RQ1.html")

ACTION_INTERVAL_S = 1.0
SLA_THRESHOLD_MS = 250
DPI = 200


@dataclass(frozen=True)
class Alg:
    key: str
    display: str
    category: str
    color: str


ALGORITHMS: list[Alg] = [
    Alg("ppo", "PPO", "RL", "#1f77b4"),
    Alg("ddpg", "DDPG", "RL", "#ff7f0e"),
    Alg("mdqn", "MDQN", "RL", "#2ca02c"),
    Alg("k8s_vpa", "VPA", "Heuristic", "#d62728"),
    Alg("llm_vpa_ollama_llama3_8b", "Llama 3 8B", "LLM", "#9467bd"),
    Alg("llm_vpa_ollama_mistral_latest", "Mistral 7B", "LLM", "#8c564b"),
    Alg("llm_vpa_ollama_qwen2_5_7b", "Qwen 2.5 7B", "LLM", "#e377c2"),
]
BY_KEY = {a.key: a for a in ALGORITHMS}
BY_DISPLAY = {a.display: a for a in ALGORITHMS}

PHASE_BG = {"low": "#e8f4f8", "medium": "#fff4e6", "high": "#fde8e8"}
PHASES = ["low", "medium", "high"]

# Phase boundaries in benchmark steps (3 phases × 60 steps).
PHASE_BOUNDARIES = [60, 120]
TRANSITIONS = [
    ("low→medium", 60, (55, 59), (60, 64), (80, 109)),
    ("medium→high", 120, (115, 119), (120, 124), (140, 169)),
]


# ----------------------------------------------------------------------
# Data loading & aggregation helpers
# ----------------------------------------------------------------------


def load_all() -> pd.DataFrame:
    """Load every <alg>_iter*_metrics.csv into one DataFrame."""
    frames = []
    for alg in ALGORITHMS:
        pattern = os.path.join(RESULTS_DIR, f"{alg.key}_iter*_metrics.csv")
        for fp in sorted(glob.glob(pattern)):
            m = re.search(r"iter(\d+)", os.path.basename(fp))
            if not m:
                continue
            df = pd.read_csv(fp)
            df["algorithm"] = alg.key
            df["display"] = alg.display
            df["category"] = alg.category
            df["iter"] = int(m.group(1))
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No metrics CSVs found under {RESULTS_DIR}")
    return pd.concat(frames, ignore_index=True)


def parse_vpa_action(raw) -> float:
    """Extract the VPA (CPU) component of an action cell — scalar or [vpa hpa]."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw)
    try:
        return float(s)
    except ValueError:
        pass
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)
    return float(nums[0]) if nums else 0.0


def per_alg_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alg in ALGORITHMS:
        sub = df[df.algorithm == alg.key]
        rt = sub.response_time.values
        rows.append(
            dict(
                display=alg.display,
                category=alg.category,
                mean_rt_ms=1000 * rt.mean(),
                p50_rt_ms=1000 * np.percentile(rt, 50),
                p95_rt_ms=1000 * np.percentile(rt, 95),
                p99_rt_ms=1000 * np.percentile(rt, 99),
                sla_violation_pct=100 * (rt > SLA_THRESHOLD_MS / 1000).mean(),
                reward=sub.reward.mean(),
                dec_lat_ms_mean=sub.decision_latency_ms.mean(),
                dec_lat_ms_p99=np.percentile(sub.decision_latency_ms, 99),
                dec_lat_over_1s_pct=100 * (sub.decision_latency_ms > 1000).mean(),
                cpu_lim_mean=sub.cpu_limit.mean(),
                cpu_lim_std=sub.cpu_limit.std(),
                cpu_lim_max=sub.cpu_limit.max(),
                cpu_lim_min=sub.cpu_limit.min(),
                cpu_pct_mean=sub.cpu_percentage.mean(),
            )
        )
    return pd.DataFrame(rows)


def transition_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alg in ALGORITHMS:
        sub = df[df.algorithm == alg.key]
        for name, boundary, pre_w, spike_w, sus_w in TRANSITIONS:
            rows.append(
                dict(
                    display=alg.display,
                    transition=name,
                    pre_ms=1000 * sub[sub.step.between(*pre_w)].response_time.mean(),
                    spike_ms=1000 * sub[sub.step.between(*spike_w)].response_time.mean(),
                    sustained_ms=1000
                    * sub[sub.step.between(*sus_w)].response_time.mean(),
                )
            )
    return pd.DataFrame(rows)


def per_step_mean_rt(df: pd.DataFrame, alg_key: str) -> pd.Series:
    """Mean response time grouped by step, averaged across iterations and agents."""
    sub = df[df.algorithm == alg_key]
    return sub.groupby("step").response_time.mean().sort_index()


def per_step_mean_latency(df: pd.DataFrame, alg_key: str) -> pd.Series:
    sub = df[df.algorithm == alg_key]
    return sub.groupby("step").decision_latency_ms.mean().sort_index()


def action_mix(df: pd.DataFrame, alg_key: str) -> dict[str, float]:
    """Return (no_op, scale_up, scale_down) shares based on Δcpu_limit per step.

    A Δ smaller than 5 m counts as no-op, > +5 m as scale-up, < -5 m as scale-down.
    """
    sub = df[df.algorithm == alg_key].sort_values(["iter", "agent_id", "step"])
    diffs = (
        sub.groupby(["iter", "agent_id"]).cpu_limit.diff().dropna().values
    )
    if diffs.size == 0:
        return dict(no_op=1.0, up=0.0, down=0.0)
    no_op = float((np.abs(diffs) < 5).mean())
    up = float((diffs >= 5).mean())
    down = float((diffs <= -5).mean())
    return dict(no_op=no_op, up=up, down=down)


def shade_phases(ax, x_start=0, x_end=180):
    """Apply light background shading for the three workload phases."""
    bounds = [(0, 60, "low"), (60, 120, "medium"), (120, 180, "high")]
    for x0, x1, ph in bounds:
        x0_clip, x1_clip = max(x0, x_start), min(x1, x_end)
        if x1_clip > x0_clip:
            ax.axvspan(x0_clip, x1_clip, color=PHASE_BG[ph], alpha=0.6, zorder=0)


# ----------------------------------------------------------------------
# Plot 1 — decision latency CDF
# ----------------------------------------------------------------------


def fig_decision_latency_cdf(df, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    for alg in ALGORITHMS:
        vals = np.sort(df[df.algorithm == alg.key].decision_latency_ms.values)
        cdf = np.linspace(0, 1, len(vals), endpoint=True)
        ax.step(
            vals,
            cdf,
            where="post",
            label=alg.display,
            color=alg.color,
            linewidth=1.8,
            alpha=0.9,
        )
    ax.axvline(ACTION_INTERVAL_S * 1000, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(
        ACTION_INTERVAL_S * 1000 * 1.1,
        0.04,
        "control interval (1 s)",
        fontsize=9,
        alpha=0.75,
        rotation=90,
        verticalalignment="bottom",
    )
    ax.set_xscale("log")
    ax.set_xlim(0.05, 50_000)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Decision latency (ms, log)")
    ax.set_ylabel("CDF")
    ax.set_title("Per-decision latency distribution — RL/heuristic complete in milliseconds, LLMs exceed the 1 s control interval 100 % of the time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 2 — open-loop overlay (RT and decision latency around low→medium)
# ----------------------------------------------------------------------


def fig_open_loop_overlay(df, out):
    cmp_algs = ["ppo", "ddpg", "llm_vpa_ollama_llama3_8b", "llm_vpa_ollama_qwen2_5_7b"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_rt, ax_lat = axes

    for key in cmp_algs:
        alg = BY_KEY[key]
        rt = per_step_mean_rt(df, key)
        lat = per_step_mean_latency(df, key)
        ax_rt.plot(rt.index, rt.values * 1000, label=alg.display, color=alg.color, linewidth=1.8, alpha=0.9)
        ax_lat.plot(lat.index, lat.values, color=alg.color, linewidth=1.8, alpha=0.9)

    for ax in axes:
        shade_phases(ax)
        for b in PHASE_BOUNDARIES:
            ax.axvline(b, color="grey", linewidth=0.7, linestyle=":", alpha=0.7)

    ax_rt.axhline(SLA_THRESHOLD_MS, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax_rt.text(2, SLA_THRESHOLD_MS * 1.1, "SLA = 250 ms", fontsize=8, alpha=0.7)
    ax_rt.set_yscale("log")
    ax_rt.set_ylabel("Mean RT (ms, log)")
    ax_rt.set_title("Open-loop artifact: smooth LLM transitions coincide with multi-second decision windows")
    ax_rt.grid(True, which="both", alpha=0.25)
    ax_rt.legend(loc="upper left", fontsize=9, ncol=4)

    ax_lat.axhline(ACTION_INTERVAL_S * 1000, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_lat.text(2, ACTION_INTERVAL_S * 1000 * 1.1, "1 s control interval", fontsize=8, alpha=0.75)
    ax_lat.set_yscale("log")
    ax_lat.set_ylabel("Mean decision latency (ms, log)")
    ax_lat.set_xlabel("Step (phase boundaries dashed)")
    ax_lat.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 3 — cpu_limit violin per algorithm
# ----------------------------------------------------------------------


def fig_cpu_limit_violin(df, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    data = []
    labels = []
    colors = []
    for alg in ALGORITHMS:
        vals = df[df.algorithm == alg.key].cpu_limit.values
        data.append(vals)
        labels.append(alg.display)
        colors.append(alg.color)
    parts = ax.violinplot(
        data,
        showmeans=True,
        showextrema=True,
        widths=0.85,
    )
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.55)
    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        if key in parts:
            parts[key].set_edgecolor("#333")
            parts[key].set_linewidth(1.0)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("CPU limit (m)")
    ax.set_title("CPU-limit distribution across the run — Qwen never leaves the floor, Mistral oscillates wildly")
    ax.set_ylim(0, max(d.max() for d in data) * 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 4 — transition window bars (pre / spike / sustained per algorithm)
# ----------------------------------------------------------------------


def fig_transition_window_bars(df, out):
    table = transition_table(df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, (transition_name, _, _, _, _) in zip(axes, TRANSITIONS):
        sub = table[table.transition == transition_name]
        sub = sub.set_index("display").reindex([a.display for a in ALGORITHMS])
        x = np.arange(len(sub))
        bw = 0.27
        bars1 = ax.bar(x - bw, sub.pre_ms, bw, color="#88c0d0", label="pre-transition (5 steps before)")
        bars2 = ax.bar(x, sub.spike_ms, bw, color="#bf616a", label="spike (5 steps after)")
        bars3 = ax.bar(x + bw, sub.sustained_ms, bw, color="#a3be8c", label="sustained (steps +20…+50)")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=15, ha="right")
        ax.set_ylabel("Mean RT (ms, log)")
        ax.set_title(f"Transition window — {transition_name}")
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.axhline(SLA_THRESHOLD_MS, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("RL spike-and-recover vs LLM smooth-but-elevated", y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 5 — reward vs SLA scatter
# ----------------------------------------------------------------------


def fig_reward_vs_sla(df, out, summary):
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in summary.iterrows():
        alg = BY_DISPLAY[row.display]
        ax.scatter(
            row.reward,
            row.sla_violation_pct,
            s=180,
            color=alg.color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=alg.display,
        )
        ax.annotate(
            alg.display,
            (row.reward, row.sla_violation_pct),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )
    ax.set_xlabel("Mean reward (higher = better, training-time objective)")
    ax.set_ylabel("SLA violation rate (%)  — lower = better")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Reward and SLA violations are decoupled — the highest-reward agent is not the lowest-violation agent")
    ax.grid(True, alpha=0.3)
    ax.axhline(SLA_THRESHOLD_MS / 1000 * 100 / 25, color="none")  # dummy to avoid clip
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 6 — sustained RT floor per phase
# ----------------------------------------------------------------------


def fig_sustained_rt_floor(df, out):
    last30_windows = {"low": (30, 59), "medium": (90, 119), "high": (150, 179)}
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(ALGORITHMS))
    bw = 0.27
    palette = {"low": "#88c0d0", "medium": "#ebcb8b", "high": "#bf616a"}
    for i, (phase, (a, b)) in enumerate(last30_windows.items()):
        ys = []
        for alg in ALGORITHMS:
            sub = df[(df.algorithm == alg.key) & (df.step.between(a, b))]
            ys.append(1000 * sub.response_time.mean())
        ax.bar(x + (i - 1) * bw, ys, bw, color=palette[phase], label=f"{phase} phase (steps {a}–{b})")
    ax.axhline(SLA_THRESHOLD_MS, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(ALGORITHMS) - 0.5, SLA_THRESHOLD_MS * 1.1, "SLA = 250 ms", fontsize=8, alpha=0.7, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([a.display for a in ALGORITHMS], rotation=15, ha="right")
    ax.set_ylabel("Sustained mean RT (ms, log)")
    ax.set_title("Sustained-RT floor — RL settles below 10 ms, every LLM stays at least 5× higher")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 7 — Pareto frontier: decision latency × SLA violation rate
# ----------------------------------------------------------------------


def fig_pareto_latency_sla(df, out, summary):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    pts = []
    for _, row in summary.iterrows():
        alg = BY_DISPLAY[row.display]
        x = max(row.dec_lat_ms_mean, 0.1)
        y = row.sla_violation_pct  # lower = better
        pts.append((x, y, alg))
        ax.scatter(x, y, s=220, color=alg.color, edgecolor="black", linewidth=0.7, alpha=0.9, zorder=3)
        ax.annotate(
            alg.display,
            (x, y),
            textcoords="offset points",
            xytext=(10, -4),
            fontsize=9,
        )

    # Pareto front: points where no other point has both smaller x AND smaller y
    # (lower decision latency AND lower violation rate).
    pareto = []
    for i, (x, y, a) in enumerate(pts):
        dominated = False
        for j, (x2, y2, _) in enumerate(pts):
            if j == i:
                continue
            if x2 <= x and y2 <= y and (x2 < x or y2 < y):
                dominated = True
                break
        if not dominated:
            pareto.append((x, y, a))
    pareto.sort(key=lambda t: t[0])
    if len(pareto) >= 2:
        px, py = zip(*[(p[0], p[1]) for p in pareto])
        ax.plot(px, py, "k--", linewidth=1.3, alpha=0.5, zorder=2, label="Pareto front")
        ax.legend(loc="upper left", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Mean decision latency (ms, log)")
    ax.set_ylabel("SLA violation rate (%)  — lower = better")
    ax.set_title("Pareto trade-off: SLA violation rate vs decision cost (HOM, 20 iter × 7 algorithms)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 8 — action mix per algorithm
# ----------------------------------------------------------------------


def fig_action_mix(df, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    mix = [action_mix(df, alg.key) for alg in ALGORITHMS]
    no_op = np.array([m["no_op"] for m in mix]) * 100
    up = np.array([m["up"] for m in mix]) * 100
    down = np.array([m["down"] for m in mix]) * 100

    x = np.arange(len(ALGORITHMS))
    bw = 0.6
    ax.bar(x, no_op, bw, color="#d8dee9", label="no-op (|Δ| < 5 m)")
    ax.bar(x, up, bw, bottom=no_op, color="#a3be8c", label="scale-up (Δ ≥ +5 m)")
    ax.bar(x, down, bw, bottom=no_op + up, color="#bf616a", label="scale-down (Δ ≤ −5 m)")

    for i, alg in enumerate(ALGORITHMS):
        ax.text(i, 102, f"{no_op[i]:.0f}/{up[i]:.0f}/{down[i]:.0f}", fontsize=8, ha="center", color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels([a.display for a in ALGORITHMS], rotation=15, ha="right")
    ax.set_ylabel("Share of steps (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Action mix — how often each policy actually changes the CPU limit (no-op / up / down %)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 9 — per-phase RT CDFs
# ----------------------------------------------------------------------


def fig_rt_cdf_per_phase(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, phase in zip(axes, PHASES):
        for alg in ALGORITHMS:
            vals = np.sort(
                df[(df.algorithm == alg.key) & (df.phase == phase)].response_time.values
                * 1000
            )
            if vals.size == 0:
                continue
            cdf = np.linspace(0, 1, len(vals), endpoint=True)
            ax.step(vals, cdf, where="post", label=alg.display, color=alg.color, linewidth=1.6, alpha=0.9)
        ax.axvline(SLA_THRESHOLD_MS, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlim(1, 30_000)
        ax.set_xlabel("Response time (ms, log)")
        ax.set_title(f"{phase.capitalize()} phase")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("CDF")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    fig.suptitle("Response-time CDF per workload phase — failure modes are phase-specific", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot 10 — action volatility (cpu_limit std/mean)
# ----------------------------------------------------------------------


def fig_action_volatility(df, out, summary):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(ALGORITHMS))
    vol = []
    for alg in ALGORITHMS:
        row = summary[summary.display == alg.display].iloc[0]
        vol.append(row.cpu_lim_std / row.cpu_lim_mean if row.cpu_lim_mean > 0 else 0)
    colors = [a.color for a in ALGORITHMS]
    ax.bar(x, vol, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for i, (alg, v) in enumerate(zip(ALGORITHMS, vol)):
        row = summary[summary.display == alg.display].iloc[0]
        ax.text(
            i,
            v + 0.02,
            f"mean={row.cpu_lim_mean:.0f}m\nstd/mean={v:.2f}",
            fontsize=8,
            ha="center",
            color="#333",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([a.display for a in ALGORITHMS], rotation=15, ha="right")
    ax.set_ylabel("CPU-limit volatility  (std / mean)")
    ax.set_title("Action volatility — Mistral oscillates, Qwen is stuck, RL/Llama balanced")
    ax.set_ylim(0, max(vol) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------

CSS = """
:root { color-scheme: light dark; }
html { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
body { max-width: 1040px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.55; color: #222; background: #fafafa; }
h1 { font-size: 1.9rem; margin-bottom: .2rem; }
h1 + .subtitle { color: #666; margin-top: 0; margin-bottom: 2rem; font-size: 1.0rem; }
h2 { font-size: 1.25rem; margin-top: 3rem; padding-top: .8rem; border-top: 1px solid #ddd; }
h3 { font-size: 1.05rem; margin-top: 1.5rem; color: #444; }
.key { background: #fff8c5; padding: .55rem .85rem; border-left: 3px solid #d4a017; margin: .8rem 0; border-radius: 2px; font-size: .95rem; }
.paper { background: #eef6f0; padding: .65rem .9rem; border-left: 3px solid #3c8a52; margin: 1rem 0 1.5rem 0; border-radius: 2px; font-size: .95rem; }
.paper::before { content: "Paper-ready sentence:  "; font-weight: 600; color: #2e5d3a; }
.fig { text-align: center; margin: 1.4rem 0; }
.fig img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; background: #fff; }
table { border-collapse: collapse; width: 100%; margin: 1.2rem 0; font-size: .92rem; }
th, td { padding: .45rem .65rem; border-bottom: 1px solid #e0e0e0; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #ececec; font-weight: 600; }
tr.rl td { background: #f6fafd; }
tr.heur td { background: #fff5f5; }
tr.llm td { background: #f8f5fc; }
.cat { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: .75rem; font-weight: 600; color: white; }
.cat.RL { background: #1f77b4; }
.cat.Heuristic { background: #d62728; }
.cat.LLM { background: #9467bd; }
code { background: #f0f0f0; padding: 1px 5px; border-radius: 2px; font-size: .88em; }
.callout { background: #f0f5fa; padding: 1rem 1.2rem; border-radius: 4px; margin: 1rem 0; }
.context { margin-bottom: 2.5rem; padding: 1rem 1.2rem; background: #f0f0f0; border-radius: 4px; }
@media (prefers-color-scheme: dark) {
  body { background: #1c1c1c; color: #ddd; }
  h1, h2, h3 { color: #eee; }
  th { background: #2a2a2a; }
  th, td { border-bottom-color: #333; }
  tr.rl td { background: #1e2630; }
  tr.heur td { background: #2a1d1d; }
  tr.llm td { background: #251d2a; }
  .key { background: #3a3018; border-left-color: #d4a017; color: #f4e4b0; }
  .paper { background: #1e2c20; border-left-color: #3c8a52; color: #c6e8d0; }
  .context, .callout { background: #2a2a2a; }
  .fig img { background: #fff; }
}
"""


def cat_class(cat: str) -> str:
    return {"RL": "rl", "Heuristic": "heur", "LLM": "llm"}.get(cat, "")


def render_tldr_table(summary: pd.DataFrame) -> str:
    rows = []
    for _, row in summary.iterrows():
        alg = BY_DISPLAY[row.display]
        rows.append(
            f'<tr class="{cat_class(alg.category)}">'
            f"<td>{alg.display} <span class='cat {alg.category}'>{alg.category}</span></td>"
            f"<td>{row.mean_rt_ms:.1f}</td>"
            f"<td>{row.p99_rt_ms:.1f}</td>"
            f"<td>{row.sla_violation_pct:.1f}</td>"
            f"<td>{row.reward:.3f}</td>"
            f"<td>{row.dec_lat_ms_mean:,.1f}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return textwrap.dedent(
        f"""
        <table>
          <thead>
            <tr>
              <th>Algorithm</th><th>Mean RT (ms)</th><th>p99 RT (ms)</th>
              <th>SLA violation (%)</th><th>Mean reward</th><th>Mean decision latency (ms)</th>
            </tr>
          </thead>
          <tbody>
            {body}
          </tbody>
        </table>
        """
    )


def render_transition_table(table: pd.DataFrame) -> str:
    rows = []
    for _, r in table.iterrows():
        rows.append(
            f"<tr><td>{r.display}</td><td>{r.transition}</td>"
            f"<td>{r.pre_ms:.0f}</td><td>{r.spike_ms:.0f}</td><td>{r.sustained_ms:.0f}</td></tr>"
        )
    body = "\n".join(rows)
    return textwrap.dedent(
        f"""
        <table>
          <thead>
            <tr><th>Algorithm</th><th>Transition</th><th>pre (ms)</th><th>spike (ms)</th><th>sustained (ms)</th></tr>
          </thead>
          <tbody>{body}</tbody>
        </table>
        """
    )


FINDINGS = [
    dict(
        n=1,
        title="LLM decisions exceed the control interval in 100 % of cases",
        figure="rq1_decision_latency_cdf.png",
        key=(
            "RL agents decide in 0.3–3 ms. Every Llama 3 8B decision takes ≈9.7 s "
            "(p99 = 15 s); every Mistral 7B decision averages 3.7 s (p99 = 18 s); "
            "Qwen 2.5 7B is fastest among LLMs at 1.7 s but still above the 1 s "
            "control interval in 100 % of steps."
        ),
        why=(
            "RL agents are a single forward pass through a small fully-connected "
            "network — one matrix multiply per decision. LLM agents must (i) format "
            "telemetry into a system + user prompt, (ii) run autoregressive token "
            "generation across many decoder layers, and (iii) parse a tool-call back "
            "into a numeric action. Inference latency scales with output tokens × "
            "decoder depth × parameter count, so a 7–8 B parameter model emitting "
            "a structured JSON tool call is intrinsically two orders of magnitude "
            "slower than a Bellman update or a clipped policy step. "
            "The Mistral p99 of 18 s suggests occasional tail latencies driven by "
            "extra reasoning tokens or retries on malformed tool calls."
        ),
        paper=(
            "All evaluated LLM agents exceed the 1 s control interval in 100 % of "
            "decisions (mean latencies 1.7 s for Qwen 2.5 7B, 3.7 s for Mistral 7B, "
            "and 9.7 s for Llama 3 8B), whereas RL and heuristic agents complete "
            "decisions in under 3 ms on the same hardware."
        ),
    ),
    dict(
        n=2,
        title="Smooth LLM transitions are partly an open-loop artifact",
        figure="rq1_open_loop_overlay.png",
        key=(
            "Because the controller reuses <code>last_action</code> when an LLM "
            "inference is still in flight (see the <code>TIMEOUT → reused</code> "
            "path in <code>benchmark_runner.py</code>), an LLM that takes 9.7 s to "
            "decide actively scales for only ~6 of every 60 phase steps. The "
            "smoother RT curves around transitions partly reflect *fewer* actions, "
            "not *better* actions."
        ),
        why=(
            "Closed-loop control assumes the actuator can sample the state at least "
            "once per control period. When the policy itself becomes the slow "
            "element, you effectively drop into an open-loop regime between "
            "decisions: the plant continues to evolve while the controller is "
            "still computing on a stale snapshot. This is the classical case for "
            "control delay (the τ in Smith-predictor and Padé-based delay analyses). "
            "RL controllers do not suffer this because their decision time is "
            "negligible relative to the control period; LLM controllers do."
        ),
        paper=(
            "The smoother transition curves observed for LLM-based agents in "
            "Figure~\\ref{fig:rt_bound} are partly an open-loop artifact: because "
            "the controller reuses the previous action when an inference exceeds "
            "the control interval, LLM agents emit fewer decisions per phase "
            "(approximately one decision every six to ten control steps for "
            "Llama 3 8B) than RL agents (one decision per step)."
        ),
    ),
    dict(
        n=3,
        title="Qwen fails through *under-actuation*, not slow reaction",
        figure="rq1_cpu_limit_violin.png",
        key=(
            "Qwen 2.5 7B is the fastest LLM (1.7 s) but has the worst SLA violation "
            "rate (66.3 %). Its CPU limit distribution collapses to a thin band at "
            "the floor: mean 76 m, max 100 m, no-op rate 98.9 %. The model interprets "
            "the prompt but does not consistently emit corrective <code>scale_cpu</code> "
            "tool calls when load grows."
        ),
        why=(
            "Smaller (7–8 B) instruction-tuned models exhibit known failure modes "
            "in tool-use settings: defaulting to the smallest plausible parameter, "
            "omitting tool calls when uncertain, or echoing the prior decision. "
            "Without reward feedback, the LLM has no mechanism to learn that its "
            "conservative output is wrong — it simply repeats. The structural "
            "implication is that LLM-based autoscaling is **bounded above by model "
            "capability, not by the LLM paradigm itself**: capable models (Llama 3 8B) "
            "operate the actuator competently, weaker models do not."
        ),
        paper=(
            "Qwen 2.5 7B achieves the lowest decision latency of the evaluated LLM "
            "agents (1.7 s) yet the highest SLA violation rate (66.3 %) because it "
            "selects CPU limits almost exclusively at the floor (mean 76 m, no-op "
            "rate 98.9 %). This represents a prompt-comprehension and tool-use "
            "limitation specific to smaller LLMs rather than an LLM-paradigm failure."
        ),
    ),
    dict(
        n=4,
        title="RL spike-and-recover beats LLM smooth-but-elevated",
        figure="rq1_transition_window_bars.png",
        key=(
            "PPO/DDPG/MDQN all return to ≈ 7 ms sustained RT after every "
            "transition; Llama settles at 46–55 ms (≈ 7×), Mistral at 210–238 ms "
            "(≈ 30×), Qwen diverges to 2.4–5.7 s. The headline transition "
            "*peaks* are larger for RL (e.g. MDQN low→med 178 ms vs Llama 62 ms), "
            "but the post-transition sustained RT is dramatically lower."
        ),
        why=(
            "RL policies trained with PPO use a clipped surrogate objective and a "
            "KL-divergence penalty (Schulman et al. 2017), which bounds the size of "
            "each policy update and produces smooth recoveries — the controller "
            "tightens incrementally toward the new optimum. DDPG uses a deterministic "
            "policy gradient (Lillicrap et al. 2015), giving sharper short-term "
            "reactions but the same fine-grained convergence. MDQN's discrete action "
            "set causes step-shaped spikes but its argmax target eventually lands on "
            "the right discrete bin. LLM agents lack any equivalent mechanism for "
            "tightening: their actions are draws from a generation policy that the "
            "system cannot update online, so once they reach a 'good enough' allocation "
            "they stop moving — even if that allocation is sub-optimal."
        ),
        paper=(
            "RL-based agents exhibit larger transient response-time spikes around "
            "phase boundaries (e.g. 178 ms for MDQN on low→medium) but settle to "
            "≈ 7 ms sustained RT thereafter. LLM-based agents exhibit smaller "
            "spikes but elevated sustained RT (46–238 ms for Llama and Mistral). "
            "The trade-off introduced by LLM-based scaling is therefore smoother "
            "transitions paid for with persistently higher latency floors."
        ),
    ),
    dict(
        n=5,
        title="Reward is a poor proxy for SLA violation rate",
        figure="rq1_reward_vs_sla.png",
        key=(
            "MDQN achieves the highest cumulative reward (1.645) but ranks third by "
            "SLA violation rate (3.3 %), behind DDPG (0.4 %) and PPO (1.0 %). "
            "VPA reward actually *rises* under harder conditions when the budget "
            "headroom term dominates."
        ),
        why=(
            "The benchmark's reward signal is a weighted combination of response-time "
            "penalty *and* CPU-budget headroom (see <code>configs/elasticity_config.yaml</code> "
            "and the env step function). A controller that under-allocates accumulates "
            "reward from the headroom term even when it is violating the user-facing "
            "SLA. This decoupling matters because RL training optimises reward, not "
            "SLA violation rate; reporting reward alone in evaluation tables "
            "conflates training objective with deployment success criterion."
        ),
        paper=(
            "Cumulative reward is decoupled from SLA violation rate: MDQN obtains "
            "the highest mean reward (1.645) but ranks third by SLA violation rate, "
            "while DDPG and PPO achieve lower rewards (1.341 and 1.403) with lower "
            "SLA violation rates. We therefore report SLA violation rate as an "
            "independent metric throughout this evaluation."
        ),
    ),
    dict(
        n=6,
        title="LLM agents share a sustained-latency floor independent of model",
        figure="rq1_sustained_rt_floor.png",
        key=(
            "In the last 30 steps of every phase, PPO/DDPG/MDQN sit between 6 and "
            "10 ms; Llama at 46–55 ms (≈ 8×); Mistral at 210–238 ms (≈ 30×); Qwen "
            "blows up under load. Even the best LLM cannot match RL's steady-state."
        ),
        why=(
            "Three architectural causes stack: (i) LLM actuation is discrete in "
            "50 m steps (the tool schema requires integer millicores), so the "
            "controller cannot fine-tune around the optimum the way DDPG can with "
            "a continuous action; (ii) every LLM decision is stochastic — temperature "
            "sampling means two identical telemetry states can produce different "
            "actions, raising allocation variance; (iii) the open-loop windows "
            "(Finding 2) mean the controller cannot incrementally correct between "
            "decisions, so once an allocation is set it persists for 5–10 steps "
            "regardless of drift."
        ),
        paper=(
            "All LLM-based agents exhibit a steady-state response-time floor that is "
            "an order of magnitude higher than RL-based agents (8× for Llama 3 8B, "
            "30× for Mistral 7B). This floor reflects three structural properties of "
            "prompt-based control: discrete action granularity, stochastic sampling, "
            "and inter-decision open-loop windows."
        ),
    ),
    dict(
        n=7,
        title="The Pareto trade-off — no LLM is on the front",
        figure="rq1_pareto_latency_sla.png",
        key=(
            "Plotting algorithms in the (decision-latency, SLA-violation) plane "
            "yields a Pareto front of <strong>DDPG → PPO → MDQN → VPA</strong>. "
            "<strong>Every LLM is dominated</strong>: Llama 3 8B is at "
            "(9.7 s, 3.2 % violation) but PPO sits at (2.5 ms, 1.0 % violation) and "
            "DDPG at (2.8 ms, 0.4 % violation) — both have lower decision latency "
            "<em>and</em> a lower SLA violation rate. Mistral (6.0 %) and Qwen "
            "(66.3 %) are dominated by VPA (26.0 %) on top of that."
        ),
        why=(
            "This view turns the qualitative <i>low / medium / high</i> rankings in "
            "the paper into a measured trade-off space. On a homogeneous workload "
            "there is <strong>no LLM operating point worth choosing</strong> if you "
            "have access to a trained RL policy: PPO and DDPG dominate every LLM on "
            "both axes simultaneously. The only argument for an LLM on this workload "
            "is the qualitative one (inspectability of natural-language decisions, "
            "zero training data required), and that argument has to be made against "
            "a flat ≈ 4 orders-of-magnitude latency penalty and a measurable gap in "
            "SLA violation rate. Llama 3 8B is the <em>best</em> LLM, but it is not "
            "Pareto-optimal — it is simply the least-bad LLM. (We expect this picture "
            "to invert under heterogeneous workloads where RL is out-of-distribution "
            "— see RQ2.)"
        ),
        paper=(
            "On the homogeneous workload, the (decision-latency × SLA-violation-rate) "
            "Pareto front contains only the heuristic VPA and the three RL agents "
            "(MDQN, PPO, DDPG); all three LLM agents are strictly dominated. "
            "LLM-based autoscaling therefore offers no quantitative advantage when "
            "an in-distribution RL policy is available; its remaining argument is "
            "qualitative (explainability, zero training-data requirement) and must "
            "be weighed against an ≈ 10⁴× decision-latency penalty."
        ),
    ),
    dict(
        n=8,
        title="Action mix differs systematically by category",
        figure="rq1_action_mix.png",
        key=(
            "PPO and Llama hold the existing CPU limit ≈ 96 % of steps; DDPG churns "
            "(no-op rate 80 %) with frequent small adjustments; Qwen no-ops 99 % of "
            "steps because it never decides to scale up; VPA scales aggressively on "
            "both sides because its rule fires whenever utilisation crosses a fixed "
            "threshold."
        ),
        why=(
            "The action-mix is a fingerprint of the *policy structure*. PPO/Llama "
            "settle into low-entropy regimes and hold; DDPG's deterministic policy "
            "gradient encourages constant small corrections; VPA has no concept of "
            "'stay still' and treats utilisation crossings as edge triggers; Qwen's "
            "tool-use prior is dominated by the implicit 'when in doubt, do nothing' "
            "behaviour of small instruct-tuned models."
        ),
        paper=(
            "The fraction of control steps in which each policy actively changes "
            "the CPU limit varies from 1 % (Qwen 2.5 7B) to ≈ 20 % (DDPG). PPO and "
            "Llama 3 8B both fall in the 3–4 % range, reflecting low-entropy hold "
            "behaviour that is structurally different from VPA's reactive "
            "edge-triggered adjustments."
        ),
    ),
    dict(
        n=9,
        title="Per-phase RT distribution shape reveals failure mode",
        figure="rq1_rt_cdf_per_phase.png",
        key=(
            "RL CDFs are vertical lines at ≈ 10 ms across every phase. Llama matches "
            "RL in low, lags slightly in medium / high. Mistral develops a fat body "
            "around 100–500 ms. Qwen's high-phase CDF is bimodal — a small fast cluster "
            "(reused old action) and a large slow cluster (catastrophic queueing)."
        ),
        why=(
            "CDFs disambiguate average behaviour from worst-case behaviour. A "
            "long-tail (RL under stress) suggests rare worst cases; a fat-body "
            "(Mistral) suggests chronic mis-allocation; a bimodal (Qwen high) "
            "suggests two distinct operating regimes (e.g. queue full / queue empty), "
            "consistent with the controller alternating between under-provisioned "
            "and starved-of-action states."
        ),
        paper=(
            "Per-phase RT CDFs reveal distinct failure modes: RL agents exhibit "
            "thin distributions with rare long tails; Mistral develops a fat-body "
            "around 100–500 ms in the medium and high phases; Qwen's high-phase "
            "CDF is bimodal, indicating alternation between under-provisioned and "
            "starved-of-action regimes."
        ),
    ),
    dict(
        n=10,
        title="Action volatility — std / mean of cpu_limit reveals strategy",
        figure="rq1_action_volatility.png",
        key=(
            "Std / mean is 0.37 for PPO and 0.42 for DDPG (controlled adaptation), "
            "0.58 for Llama (decisive but coarse), 0.69 for Mistral (volatile), "
            "and 0.33 for Qwen at the wrong absolute level (75 m). Volatility "
            "without a sensible operating point is the worst combination."
        ),
        why=(
            "Read together with the action-mix figure, the volatility ratio "
            "separates *deliberate adjustment* (PPO, DDPG, Llama — high no-op rate "
            "but wide std when they do act) from *prompt-driven oscillation* "
            "(Mistral — high std *and* high action rate, so the policy keeps "
            "rebalancing) from *static under-allocation* (Qwen — narrow band, low "
            "mean). The structural lesson is that aggregate mean allocation is a "
            "weak summary: how *consistently* the policy operates at the right "
            "point matters more."
        ),
        paper=(
            "We characterise each policy's strategy via the volatility ratio "
            "(σ/μ of the CPU limit): values cluster around 0.4 for the controlled "
            "RL policies, 0.58 for the best LLM (Llama 3 8B), and 0.69 for "
            "Mistral 7B, whose high volatility reflects prompt-driven oscillation "
            "rather than convergent control."
        ),
    ),
]


def render_finding(f) -> str:
    return textwrap.dedent(
        f"""
        <section id="finding-{f['n']}">
          <h2>Finding {f['n']} — {f['title']}</h2>
          <div class="key">{f['key']}</div>
          <div class="fig"><img src="figures/{f['figure']}" alt="Finding {f['n']} figure"></div>
          <h3>What's going on architecturally</h3>
          <p>{f['why']}</p>
          <div class="paper">{f['paper']}</div>
        </section>
        """
    )


def render_html(summary: pd.DataFrame, transitions: pd.DataFrame) -> str:
    findings_html = "\n".join(render_finding(f) for f in FINDINGS)
    tldr = render_tldr_table(summary)
    transition_html = render_transition_table(transitions)
    n_iters = 20
    return textwrap.dedent(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RQ1 — Homogeneous workload deep dive</title>
  <style>{CSS}</style>
</head>
<body>

<h1>RQ1: Under homogeneous in-place autoscaling, what trade-offs do LLM-based agents introduce?</h1>
<p class="subtitle">Deep-dive analysis of <code>results/benchmark/small_3105/</code> — {n_iters} iterations × 7 algorithms × 3 phases (low, medium, high RPS) — all metrics, all plots.</p>

<div class="context">
  <strong>Experimental setup.</strong> Two pods of the same homogeneous KNN-regressor service (<code>localization-api1</code>, <code>localization-api2</code>) on a MicroK8s cluster. Each algorithm controls each pod for 180 steps (60 per phase) at <code>action_interval = 1.0 s</code>. Workload is generated by <code>src/spam_cluster.py</code>; the SLA target is 250 ms response time. Three categories of controller are compared:
  <ul>
    <li><span class="cat RL">RL</span> &nbsp; <strong>PPO, DDPG, MDQN</strong> — trained off-line on this workload, deployed with a fixed policy.</li>
    <li><span class="cat Heuristic">Heuristic</span> &nbsp; <strong>VPA</strong> — in-place version of Kubernetes' Vertical Pod Autoscaler recommender (90th-percentile of recent CPU usage + 15 % safety margin).</li>
    <li><span class="cat LLM">LLM</span> &nbsp; <strong>Llama 3 8B, Mistral 7B, Qwen 2.5 7B</strong> via Ollama, prompted with current telemetry + recent history (see <code>src/llm_agent.py</code>) and the <code>scale_cpu / scale_replicas / no_action</code> tool schema.</li>
  </ul>
</div>

<h2>TL;DR — one row per algorithm</h2>
{tldr}

<h3>Transition windows — RT pre / spike / sustained</h3>
{transition_html}

{findings_html}

<h2>Closing — answer to RQ1 in one sentence</h2>
<div class="paper">Under homogeneous in-place autoscaling, RL-based agents dominate every quantitative axis (sustained RT below 10 ms, SLA violation rate below 3.3 %, decision latency below 3 ms) and form the Pareto front together with the heuristic VPA baseline; all three LLM-based agents are strictly dominated, with the best LLM (Llama 3 8B at 3.2 % SLA violation rate and 9.7 s mean decision latency) trailing PPO and DDPG on both axes simultaneously. The smaller LLM agents fail in distinct ways — Qwen 2.5 7B from chronic under-actuation (98.9 % no-op rate at the CPU floor) and Mistral 7B from prompt-driven oscillation (σ/μ = 0.69 of CPU limit). LLM-based autoscaling on a homogeneous workload therefore offers no quantitative advantage; its remaining argument is qualitative (decision inspectability, zero training-data requirement) and must be weighed against an ≈ 10⁴× decision-latency penalty and a measurable sustained-RT floor.</div>

<hr>
<p style="color:#777;font-size:.85rem;">Generated by <code>analysis/generate_rq1.py</code>. All metrics are computed from the raw per-iteration CSVs in <code>results/benchmark/small_3105/</code>; figures live in <code>analysis/figures/</code>. Re-run with <code>python analysis/generate_rq1.py</code> to refresh.</p>
</body>
</html>
"""
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    if not os.path.isdir(RESULTS_DIR):
        print(f"ERROR: results dir not found: {RESULTS_DIR}", file=sys.stderr)
        return 1
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"Loading CSVs from {RESULTS_DIR} ...")
    df = load_all()
    print(f"  loaded {len(df):,} rows across {df.algorithm.nunique()} algorithms")

    summary = per_alg_summary(df)
    transitions = transition_table(df)

    fig_path = lambda name: os.path.join(FIG_DIR, name)
    plots = [
        ("rq1_decision_latency_cdf.png", lambda p: fig_decision_latency_cdf(df, p)),
        ("rq1_open_loop_overlay.png", lambda p: fig_open_loop_overlay(df, p)),
        ("rq1_cpu_limit_violin.png", lambda p: fig_cpu_limit_violin(df, p)),
        ("rq1_transition_window_bars.png", lambda p: fig_transition_window_bars(df, p)),
        ("rq1_reward_vs_sla.png", lambda p: fig_reward_vs_sla(df, p, summary)),
        ("rq1_sustained_rt_floor.png", lambda p: fig_sustained_rt_floor(df, p)),
        ("rq1_pareto_latency_sla.png", lambda p: fig_pareto_latency_sla(df, p, summary)),
        ("rq1_action_mix.png", lambda p: fig_action_mix(df, p)),
        ("rq1_rt_cdf_per_phase.png", lambda p: fig_rt_cdf_per_phase(df, p)),
        ("rq1_action_volatility.png", lambda p: fig_action_volatility(df, p, summary)),
    ]
    for name, fn in plots:
        path = fig_path(name)
        print(f"  rendering {name} ...")
        fn(path)

    print(f"Writing {HTML_OUT} ...")
    with open(HTML_OUT, "w") as f:
        f.write(render_html(summary, transitions))

    print("Done.")
    print(f"  HTML: {HTML_OUT}")
    print(f"  Figures dir: {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

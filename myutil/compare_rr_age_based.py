#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOWER_BETTER = [
    "averagePacketLatency",
    "packetLatencyP95",
    "packetLatencyP99",
    "averageFlitLatency",
    "flitLatencyP95",
    "flitLatencyP99",
    "sa2ServiceRatioGini",
    "sa2ServiceRatioCv",
]

HIGHER_BETTER = [
    "packetsReceivedPerSystemCycle",
    "packetDeliveryRatio",
    "sa2ServiceRatioJain",
    "sa2GrantRatio",
]


@dataclass
class TrafficPair:
    name: str
    rr_csv: Path
    age_csv: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare RR vs age-based-RR sweep outputs")
    p.add_argument("--m5out", default="m5out", help="Root m5out directory")
    p.add_argument("--out", default="submitdata/rr_vs_age_based", help="Output directory")
    p.add_argument(
        "--traffic",
        action="append",
        default=["uniform", "transpose", "complement"],
        help="Traffic pattern name to compare (repeatable)",
    )
    return p.parse_args()


def first_valid(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[0])


def auc(x: pd.Series, y: pd.Series) -> Optional[float]:
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(df) < 2:
        return None
    df = df.sort_values("x")
    return float(np.trapz(df["y"].to_numpy(), df["x"].to_numpy()))


def pct_improve(rr: pd.Series, age: pd.Series, lower_better: bool) -> pd.Series:
    rr_n = pd.to_numeric(rr, errors="coerce")
    age_n = pd.to_numeric(age, errors="coerce")
    denom = rr_n.replace(0, np.nan)
    if lower_better:
        return (rr_n - age_n) / denom * 100.0
    return (age_n - rr_n) / denom * 100.0


def load_pair(pair: TrafficPair) -> pd.DataFrame:
    rr = pd.read_csv(pair.rr_csv)
    age = pd.read_csv(pair.age_csv)

    rr = rr[rr["status"].isin(["ok", "cached"])].copy()
    age = age[age["status"].isin(["ok", "cached"])].copy()

    common_cols = sorted(set(rr.columns).intersection(age.columns))
    keep = ["injection_rate"] + [
        c
        for c in common_cols
        if c
        not in {
            "injection_rate",
            "stage",
            "status",
            "outdir",
            "run_seconds",
            "error",
        }
    ]

    rr = rr[keep].copy().sort_values("injection_rate")
    age = age[keep].copy().sort_values("injection_rate")

    merged = rr.merge(age, on="injection_rate", suffixes=("_rr", "_age"), how="inner")
    merged["traffic"] = pair.name

    for metric in LOWER_BETTER:
        rr_col = f"{metric}_rr"
        age_col = f"{metric}_age"
        if rr_col in merged.columns and age_col in merged.columns:
            merged[f"{metric}_improve_pct"] = pct_improve(
                merged[rr_col], merged[age_col], lower_better=True
            )

    for metric in HIGHER_BETTER:
        rr_col = f"{metric}_rr"
        age_col = f"{metric}_age"
        if rr_col in merged.columns and age_col in merged.columns:
            merged[f"{metric}_gain_pct"] = pct_improve(
                merged[rr_col], merged[age_col], lower_better=False
            )

    return merged


def summarize_traffic(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["points"] = int(len(df))

    rr_del = pd.to_numeric(df.get("packetDeliveryRatio_rr"), errors="coerce")
    age_del = pd.to_numeric(df.get("packetDeliveryRatio_age"), errors="coerce")

    rr_sat = df.loc[rr_del < 0.99, "injection_rate"]
    age_sat = df.loc[age_del < 0.99, "injection_rate"]
    out["satRate_rr"] = float(rr_sat.iloc[0]) if not rr_sat.empty else np.nan
    out["satRate_age"] = float(age_sat.iloc[0]) if not age_sat.empty else np.nan
    out["satShift_age_minus_rr"] = out["satRate_age"] - out["satRate_rr"]

    def auc_improve(metric: str, lower_better: bool) -> float:
        rr_col = f"{metric}_rr"
        age_col = f"{metric}_age"
        if rr_col not in df.columns or age_col not in df.columns:
            return np.nan
        rr_auc = auc(df["injection_rate"], df[rr_col])
        age_auc = auc(df["injection_rate"], df[age_col])
        if rr_auc is None or age_auc is None or rr_auc == 0:
            return np.nan
        if lower_better:
            return (rr_auc - age_auc) / rr_auc * 100.0
        return (age_auc - rr_auc) / rr_auc * 100.0

    out["aucImprove_avgPktLat_pct"] = auc_improve("averagePacketLatency", True)
    out["aucImprove_p95_pct"] = auc_improve("packetLatencyP95", True)
    out["aucImprove_p99_pct"] = auc_improve("packetLatencyP99", True)
    out["aucGain_throughput_pct"] = auc_improve("packetsReceivedPerSystemCycle", False)

    for metric in [
        "averagePacketLatency_improve_pct",
        "packetLatencyP95_improve_pct",
        "packetLatencyP99_improve_pct",
        "packetsReceivedPerSystemCycle_gain_pct",
        "sa2ServiceRatioGini_improve_pct",
    ]:
        if metric in df.columns:
            s = pd.to_numeric(df[metric], errors="coerce").dropna()
            out[f"mean_{metric}"] = float(s.mean()) if not s.empty else np.nan

    if "packetLatencyP99_improve_pct" in df.columns:
        s = pd.to_numeric(df["packetLatencyP99_improve_pct"], errors="coerce")
        if s.notna().any():
            idx = s.idxmax()
            out["bestP99Improve_pct"] = float(s.loc[idx])
            out["bestP99Improve_rate"] = float(df.loc[idx, "injection_rate"])

    return out


def plot_traffic(df: pd.DataFrame, out_png: Path) -> None:
    x = pd.to_numeric(df["injection_rate"], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axes[0, 0].plot(x, df["averagePacketLatency_rr"], label="RR", linewidth=1.8)
    axes[0, 0].plot(x, df["averagePacketLatency_age"], label="Age-based-RR", linewidth=1.8)
    axes[0, 0].set_title("Average Packet Latency")
    axes[0, 0].set_ylabel("ticks")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x, df["packetLatencyP99_rr"], label="RR", linewidth=1.8)
    axes[0, 1].plot(x, df["packetLatencyP99_age"], label="Age-based-RR", linewidth=1.8)
    axes[0, 1].set_title("Packet Tail Latency (P99)")
    axes[0, 1].set_ylabel("ticks")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, df["packetsReceivedPerSystemCycle_rr"], label="RR", linewidth=1.8)
    axes[1, 0].plot(x, df["packetsReceivedPerSystemCycle_age"], label="Age-based-RR", linewidth=1.8)
    axes[1, 0].set_title("Throughput")
    axes[1, 0].set_ylabel("pkt/system_cycle")
    axes[1, 0].set_xlabel("Injection Rate")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, df["packetLatencyP99_improve_pct"], label="P99 Improve %", linewidth=1.8)
    axes[1, 1].plot(x, df["packetsReceivedPerSystemCycle_gain_pct"], label="Throughput Gain %", linewidth=1.8)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Relative Gain (Age-based-RR vs RR)")
    axes[1, 1].set_ylabel("%")
    axes[1, 1].set_xlabel("Injection Rate")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_overview(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(summary))
    labels = summary["traffic"].tolist()

    axes[0].bar(x - 0.15, summary["aucImprove_p99_pct"], width=0.3, label="P99 AUC Improve %")
    axes[0].bar(
        x + 0.15,
        summary["aucGain_throughput_pct"],
        width=0.3,
        label="Throughput AUC Gain %",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("AUC-based Policy Gain")
    axes[0].set_ylabel("%")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend()

    axes[1].bar(x, summary["satShift_age_minus_rr"], width=0.45)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Saturation Shift (Age-RR minus RR)")
    axes[1].set_ylabel("Injection rate")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def make_report(summary: pd.DataFrame, out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# RR vs Age-based-RR Comparative Analysis")
    lines.append("")
    lines.append("## Method")
    lines.append("- Same traffic pattern, same injection rate, one-to-one alignment.")
    lines.append("- RR as baseline; age-based-RR as treatment.")
    lines.append("- Tail latency uses strict sample-based p95/p99 from Garnet C++ stats.")
    lines.append("")
    lines.append("## Key Findings")

    for _, row in summary.iterrows():
        t = row["traffic"]
        p99_auc = row.get("aucImprove_p99_pct", np.nan)
        thr_auc = row.get("aucGain_throughput_pct", np.nan)
        sat_shift = row.get("satShift_age_minus_rr", np.nan)
        best_p99 = row.get("bestP99Improve_pct", np.nan)
        best_rate = row.get("bestP99Improve_rate", np.nan)
        lines.append(
            "- "
            + f"{t}: P99 AUC improvement={p99_auc:.3f}%, throughput AUC gain={thr_auc:.3f}%, "
            + f"saturation shift={sat_shift:.4f}, best P99 gain={best_p99:.3f}% @ rate={best_rate:.2f}."
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Age-priority generally benefits tail latency under contention by serving older requests first.")
    lines.append("- Throughput changes are usually smaller than tail-latency changes, because both policies still preserve RR tie-break fairness.")
    lines.append("- The gain is expected to concentrate near medium/high load where SA-II conflicts are frequent.")
    lines.append("")
    lines.append("## Generated Artifacts")
    lines.append("- merged comparison table: merged_comparison.csv")
    lines.append("- per-traffic summary: per_traffic_summary.csv")
    lines.append("- per-traffic figures: figures/<traffic>_rr_vs_age.png")
    lines.append("- overview figure: figures/overview_auc_and_saturation.png")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    m5out = Path(args.m5out)
    out = Path(args.out)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    traffic_names = []
    for t in args.traffic:
        if t not in traffic_names:
            traffic_names.append(t)

    pairs: List[TrafficPair] = []
    for t in traffic_names:
        rr = m5out / f"sweep_RR_0p01_{t}" / "sweep_summary.csv"
        age = m5out / f"sweep_AGEBASEDRR_0p01_{t}" / "sweep_summary.csv"
        if rr.exists() and age.exists():
            pairs.append(TrafficPair(name=t, rr_csv=rr, age_csv=age))

    if not pairs:
        raise SystemExit("No valid RR/Age-based-RR sweep pairs found.")

    merged_all: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, float]] = []

    for pair in pairs:
        df = load_pair(pair)
        merged_all.append(df)

        s = summarize_traffic(df)
        s["traffic"] = pair.name
        summary_rows.append(s)

        plot_traffic(df, fig_dir / f"{pair.name}_rr_vs_age.png")

    merged_df = pd.concat(merged_all, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("traffic")

    merged_df.to_csv(out / "merged_comparison.csv", index=False)
    summary_df.to_csv(out / "per_traffic_summary.csv", index=False)

    plot_overview(summary_df, fig_dir / "overview_auc_and_saturation.png")
    make_report(summary_df, out / "analysis_report.md")

    print(f"Wrote: {out / 'merged_comparison.csv'}")
    print(f"Wrote: {out / 'per_traffic_summary.csv'}")
    print(f"Wrote: {out / 'analysis_report.md'}")
    print(f"Wrote figures to: {fig_dir}")


if __name__ == "__main__":
    main()

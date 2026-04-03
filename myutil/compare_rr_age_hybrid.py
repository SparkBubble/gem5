#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SweepPaths:
    rr: Path
    age: Path
    hybrid: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare RR, age-based-RR, and hybrid SA-II policies")
    p.add_argument("--m5out", default="m5out", help="m5out root")
    p.add_argument(
        "--out",
        default="submitdata/rr_age_hybrid_compare",
        help="output directory",
    )
    p.add_argument(
        "--traffic",
        action="append",
        default=["uniform", "transpose", "complement"],
        help="traffic pattern (repeatable)",
    )
    p.add_argument(
        "--highload-threshold",
        type=float,
        default=0.5,
        help="high-load threshold on injection rate",
    )
    p.add_argument(
        "--near-best-latency-eps",
        type=float,
        default=0.03,
        help="hybrid near-best tolerance for latency metrics",
    )
    p.add_argument(
        "--near-best-throughput-eps",
        type=float,
        default=0.01,
        help="hybrid near-best tolerance for throughput metrics",
    )
    return p.parse_args()


def unique_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["status"].isin(["ok", "cached"])].copy()


def to_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def safe_mean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return float("nan")
    return float(v.mean())


def safe_winrate_positive(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return float("nan")
    return float((v > 0).mean() * 100.0)


def auc(x: pd.Series, y: pd.Series) -> float:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    valid = xx.notna() & yy.notna()
    if valid.sum() < 2:
        return float("nan")
    xv = xx[valid].to_numpy(dtype=float)
    yv = yy[valid].to_numpy(dtype=float)
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    return float(np.trapz(yv, xv))


def fmt(v: float, digits: int = 3) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:.{digits}f}"


def build_paths(m5out: Path, traffic: str) -> SweepPaths:
    return SweepPaths(
        rr=m5out / f"sweep_RR_0p01_{traffic}" / "sweep_summary.csv",
        age=m5out / f"sweep_AGEBASEDRR_0p01_{traffic}" / "sweep_summary.csv",
        hybrid=m5out / f"sweep_HYBRIDRRAGE_0p01_{traffic}" / "sweep_summary.csv",
    )


def merge_three(rr: pd.DataFrame, age: pd.DataFrame, hyb: pd.DataFrame, traffic: str) -> pd.DataFrame:
    keep = [
        "injection_rate",
        "averagePacketLatency",
        "packetLatencyP95",
        "packetLatencyP99",
        "packetsReceivedPerSystemCycle",
        "sa2ServiceRatioGini",
        "sa2GrantRatio",
    ]

    rr2 = rr[keep].copy().rename(columns={c: f"{c}_rr" for c in keep if c != "injection_rate"})
    age2 = age[keep].copy().rename(columns={c: f"{c}_age" for c in keep if c != "injection_rate"})
    hyb2 = hyb[keep].copy().rename(columns={c: f"{c}_hyb" for c in keep if c != "injection_rate"})

    m = rr2.merge(age2, on="injection_rate", how="inner").merge(hyb2, on="injection_rate", how="inner")
    m = m.sort_values("injection_rate").copy()
    m["traffic"] = traffic

    # Lower-is-better improvements: baseline -> hybrid
    for metric in ["averagePacketLatency", "packetLatencyP95", "packetLatencyP99", "sa2ServiceRatioGini"]:
        rr_col = f"{metric}_rr"
        age_col = f"{metric}_age"
        hyb_col = f"{metric}_hyb"
        rr_den = to_num(m, rr_col).replace(0, np.nan)
        age_den = to_num(m, age_col).replace(0, np.nan)
        m[f"{metric}_hyb_vs_rr_improve_pct"] = (to_num(m, rr_col) - to_num(m, hyb_col)) / rr_den * 100.0
        m[f"{metric}_hyb_vs_age_improve_pct"] = (to_num(m, age_col) - to_num(m, hyb_col)) / age_den * 100.0

    # Higher-is-better gains: baseline -> hybrid
    for metric in ["packetsReceivedPerSystemCycle", "sa2GrantRatio"]:
        rr_col = f"{metric}_rr"
        age_col = f"{metric}_age"
        hyb_col = f"{metric}_hyb"
        rr_den = to_num(m, rr_col).replace(0, np.nan)
        age_den = to_num(m, age_col).replace(0, np.nan)
        m[f"{metric}_hyb_vs_rr_gain_pct"] = (to_num(m, hyb_col) - to_num(m, rr_col)) / rr_den * 100.0
        m[f"{metric}_hyb_vs_age_gain_pct"] = (to_num(m, hyb_col) - to_num(m, age_col)) / age_den * 100.0

    return m


def summarize_traffic(
    m: pd.DataFrame,
    highload_threshold: float,
    near_best_latency_eps: float,
    near_best_throughput_eps: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["traffic"] = str(m["traffic"].iloc[0])
    out["points"] = int(len(m))

    inj = to_num(m, "injection_rate")
    high = m[inj >= highload_threshold]

    # Core pairwise deltas for hybrid
    out["mean_hyb_vs_rr_p99_improve_pct"] = safe_mean(m["packetLatencyP99_hyb_vs_rr_improve_pct"])
    out["mean_hyb_vs_age_p99_improve_pct"] = safe_mean(m["packetLatencyP99_hyb_vs_age_improve_pct"])
    out["mean_hyb_vs_rr_thr_gain_pct"] = safe_mean(m["packetsReceivedPerSystemCycle_hyb_vs_rr_gain_pct"])
    out["mean_hyb_vs_age_thr_gain_pct"] = safe_mean(m["packetsReceivedPerSystemCycle_hyb_vs_age_gain_pct"])

    out["high_hyb_vs_rr_p99_improve_pct"] = safe_mean(high["packetLatencyP99_hyb_vs_rr_improve_pct"])
    out["high_hyb_vs_age_p99_improve_pct"] = safe_mean(high["packetLatencyP99_hyb_vs_age_improve_pct"])
    out["high_hyb_vs_rr_thr_gain_pct"] = safe_mean(high["packetsReceivedPerSystemCycle_hyb_vs_rr_gain_pct"])
    out["high_hyb_vs_age_thr_gain_pct"] = safe_mean(high["packetsReceivedPerSystemCycle_hyb_vs_age_gain_pct"])

    out["winrate_hyb_vs_rr_p99_pct"] = safe_winrate_positive(m["packetLatencyP99_hyb_vs_rr_improve_pct"])
    out["winrate_hyb_vs_age_p99_pct"] = safe_winrate_positive(m["packetLatencyP99_hyb_vs_age_improve_pct"])

    # AUC-level aggregate
    rr_p99_auc = auc(m["injection_rate"], m["packetLatencyP99_rr"])
    age_p99_auc = auc(m["injection_rate"], m["packetLatencyP99_age"])
    hyb_p99_auc = auc(m["injection_rate"], m["packetLatencyP99_hyb"])

    rr_thr_auc = auc(m["injection_rate"], m["packetsReceivedPerSystemCycle_rr"])
    age_thr_auc = auc(m["injection_rate"], m["packetsReceivedPerSystemCycle_age"])
    hyb_thr_auc = auc(m["injection_rate"], m["packetsReceivedPerSystemCycle_hyb"])

    out["auc_hyb_vs_rr_p99_improve_pct"] = (
        (rr_p99_auc - hyb_p99_auc) / rr_p99_auc * 100.0 if rr_p99_auc and not pd.isna(rr_p99_auc) else float("nan")
    )
    out["auc_hyb_vs_age_p99_improve_pct"] = (
        (age_p99_auc - hyb_p99_auc) / age_p99_auc * 100.0 if age_p99_auc and not pd.isna(age_p99_auc) else float("nan")
    )
    out["auc_hyb_vs_rr_thr_gain_pct"] = (
        (hyb_thr_auc - rr_thr_auc) / rr_thr_auc * 100.0 if rr_thr_auc and not pd.isna(rr_thr_auc) else float("nan")
    )
    out["auc_hyb_vs_age_thr_gain_pct"] = (
        (hyb_thr_auc - age_thr_auc) / age_thr_auc * 100.0 if age_thr_auc and not pd.isna(age_thr_auc) else float("nan")
    )

    # Hybrid-focused publishable signals
    rr_p99 = to_num(m, "packetLatencyP99_rr")
    age_p99 = to_num(m, "packetLatencyP99_age")
    hyb_p99 = to_num(m, "packetLatencyP99_hyb")

    rr_thr = to_num(m, "packetsReceivedPerSystemCycle_rr")
    age_thr = to_num(m, "packetsReceivedPerSystemCycle_age")
    hyb_thr = to_num(m, "packetsReceivedPerSystemCycle_hyb")

    best_p99 = pd.concat([rr_p99, age_p99], axis=1).min(axis=1)
    best_thr = pd.concat([rr_thr, age_thr], axis=1).max(axis=1)

    valid = best_p99.notna() & hyb_p99.notna() & best_thr.notna() & hyb_thr.notna()

    strict_pareto = (hyb_p99 <= best_p99) & (hyb_thr >= best_thr) & valid
    near_best_balance = (
        (hyb_p99 <= best_p99 * (1.0 + near_best_latency_eps))
        & (hyb_thr >= best_thr * (1.0 - near_best_throughput_eps))
        & valid
    )

    out["hyb_strict_pareto_pct"] = float(strict_pareto.mean() * 100.0) if valid.any() else float("nan")
    out["hyb_near_best_balance_pct"] = float(near_best_balance.mean() * 100.0) if valid.any() else float("nan")

    # Worst-case tail guard (lower is better)
    out["worst_p99_rr"] = float(rr_p99.max())
    out["worst_p99_age"] = float(age_p99.max())
    out["worst_p99_hyb"] = float(hyb_p99.max())

    # Best and worst hybrid-vs-best gap on P99
    hyb_vs_best_gap_pct = (best_p99 - hyb_p99) / best_p99.replace(0, np.nan) * 100.0
    out["best_hyb_vs_bestp99_gap_pct"] = float(hyb_vs_best_gap_pct.max())
    out["worst_hyb_vs_bestp99_gap_pct"] = float(hyb_vs_best_gap_pct.min())
    out["mean_hyb_vs_bestp99_gap_pct"] = safe_mean(hyb_vs_best_gap_pct)

    return out


def summarize_by_load_bucket(m: pd.DataFrame) -> pd.DataFrame:
    inj = to_num(m, "injection_rate")
    bucket = pd.Series(np.where(inj < 0.3, "low", np.where(inj < 0.6, "mid", "high")), index=m.index)
    m2 = m.copy()
    m2["load_bucket"] = bucket

    rows: List[Dict[str, Any]] = []
    for lb, g in m2.groupby("load_bucket"):
        rows.append(
            {
                "traffic": str(g["traffic"].iloc[0]),
                "load_bucket": lb,
                "points": int(len(g)),
                "mean_hyb_vs_rr_p99_improve_pct": safe_mean(g["packetLatencyP99_hyb_vs_rr_improve_pct"]),
                "mean_hyb_vs_age_p99_improve_pct": safe_mean(g["packetLatencyP99_hyb_vs_age_improve_pct"]),
                "mean_hyb_vs_rr_thr_gain_pct": safe_mean(g["packetsReceivedPerSystemCycle_hyb_vs_rr_gain_pct"]),
                "mean_hyb_vs_age_thr_gain_pct": safe_mean(g["packetsReceivedPerSystemCycle_hyb_vs_age_gain_pct"]),
            }
        )
    return pd.DataFrame(rows)


def plot_curves(all_df: pd.DataFrame, out_png: Path) -> None:
    traffics = sorted(all_df["traffic"].unique())
    fig, axes = plt.subplots(len(traffics), 2, figsize=(12, 3.8 * len(traffics)), sharex=True)
    if len(traffics) == 1:
        axes = np.array([axes])

    for i, t in enumerate(traffics):
        g = all_df[all_df["traffic"] == t].sort_values("injection_rate")
        x = to_num(g, "injection_rate")

        axes[i, 0].plot(x, g["packetLatencyP99_rr"], label="RR", linewidth=1.8)
        axes[i, 0].plot(x, g["packetLatencyP99_age"], label="Age-based-RR", linewidth=1.8)
        axes[i, 0].plot(x, g["packetLatencyP99_hyb"], label="Hybrid", linewidth=1.8)
        axes[i, 0].set_title(f"{t}: Packet P99")
        axes[i, 0].set_ylabel("ticks")
        axes[i, 0].grid(alpha=0.3)

        axes[i, 1].plot(x, g["packetsReceivedPerSystemCycle_rr"], label="RR", linewidth=1.8)
        axes[i, 1].plot(x, g["packetsReceivedPerSystemCycle_age"], label="Age-based-RR", linewidth=1.8)
        axes[i, 1].plot(x, g["packetsReceivedPerSystemCycle_hyb"], label="Hybrid", linewidth=1.8)
        axes[i, 1].set_title(f"{t}: Throughput")
        axes[i, 1].set_ylabel("pkt/system_cycle")
        axes[i, 1].grid(alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Injection Rate")

    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=180)
    plt.close(fig)


def plot_hybrid_delta(all_df: pd.DataFrame, out_png: Path) -> None:
    traffics = sorted(all_df["traffic"].unique())
    fig, axes = plt.subplots(len(traffics), 2, figsize=(12, 3.8 * len(traffics)), sharex=True)
    if len(traffics) == 1:
        axes = np.array([axes])

    for i, t in enumerate(traffics):
        g = all_df[all_df["traffic"] == t].sort_values("injection_rate")
        x = to_num(g, "injection_rate")

        axes[i, 0].plot(x, g["packetLatencyP99_hyb_vs_rr_improve_pct"], label="Hybrid vs RR", linewidth=1.8)
        axes[i, 0].plot(x, g["packetLatencyP99_hyb_vs_age_improve_pct"], label="Hybrid vs Age", linewidth=1.8)
        axes[i, 0].axhline(0.0, color="black", linewidth=0.8)
        axes[i, 0].set_title(f"{t}: Hybrid P99 Improve %")
        axes[i, 0].set_ylabel("%")
        axes[i, 0].grid(alpha=0.3)

        axes[i, 1].plot(
            x,
            g["packetsReceivedPerSystemCycle_hyb_vs_rr_gain_pct"],
            label="Hybrid vs RR",
            linewidth=1.8,
        )
        axes[i, 1].plot(
            x,
            g["packetsReceivedPerSystemCycle_hyb_vs_age_gain_pct"],
            label="Hybrid vs Age",
            linewidth=1.8,
        )
        axes[i, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[i, 1].set_title(f"{t}: Hybrid Throughput Gain %")
        axes[i, 1].set_ylabel("%")
        axes[i, 1].grid(alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Injection Rate")

    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=180)
    plt.close(fig)


def plot_hybrid_balance(summary_df: pd.DataFrame, out_png: Path) -> None:
    s = summary_df.sort_values("traffic")
    x = np.arange(len(s))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width, s["hyb_strict_pareto_pct"], width=width, label="Hybrid strict Pareto %")
    ax.bar(x, s["hyb_near_best_balance_pct"], width=width, label="Hybrid near-best balance %")
    ax.bar(x + width, s["winrate_hyb_vs_rr_p99_pct"], width=width, label="Hybrid wins RR on P99 %")

    ax.set_xticks(x)
    ax.set_xticklabels(s["traffic"].tolist())
    ax.set_ylabel("% of injection rates")
    ax.set_title("Hybrid stability/compromise signals")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=180)
    plt.close(fig)


def write_report(
    summary_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    out_md: Path,
    near_best_latency_eps: float,
    near_best_throughput_eps: float,
) -> None:
    lines: List[str] = []
    lines.append("# RR / Age-based-RR / Hybrid 三策略对比分析（中文）")
    lines.append("")
    lines.append("## 1. 口径")
    lines.append("- 数据来源：m5out 中三策略 sweep_summary.csv。")
    lines.append("- 对齐方式：按 injection_rate 做三方内连接，仅保留 ok/cached。")
    lines.append("- 时延改善率定义（低者优）：(baseline - hybrid) / baseline * 100%。")
    lines.append("- 吞吐增益定义（高者优）：(hybrid - baseline) / baseline * 100%。")
    lines.append(
        "- Hybrid 近最优平衡判据：P99 不劣于最佳基线的 "
        + f"{near_best_latency_eps * 100:.1f}% 且吞吐不低于最佳基线的 {near_best_throughput_eps * 100:.1f}%。"
    )
    lines.append("")
    lines.append("## 2. 关键结果（强调 Hybrid）")

    for _, r in summary_df.sort_values("traffic").iterrows():
        lines.append(
            "- "
            + f"{r['traffic']}: Hybrid 相对 RR 的 P99 平均改善 {fmt(r['mean_hyb_vs_rr_p99_improve_pct'])}%"
            + f"，相对 Age 的 P99 平均改善 {fmt(r['mean_hyb_vs_age_p99_improve_pct'])}%"
            + f"；相对 RR 吞吐平均增益 {fmt(r['mean_hyb_vs_rr_thr_gain_pct'])}%"
            + f"，相对 Age 吞吐平均增益 {fmt(r['mean_hyb_vs_age_thr_gain_pct'])}%。"
        )
        lines.append(
            "- "
            + f"{r['traffic']}: 高负载段 Hybrid 相对 RR 的 P99 平均改善 {fmt(r['high_hyb_vs_rr_p99_improve_pct'])}%"
            + f"，相对 Age 的 P99 平均改善 {fmt(r['high_hyb_vs_age_p99_improve_pct'])}%"
            + f"；Hybrid 严格 Pareto 占比 {fmt(r['hyb_strict_pareto_pct'], 1)}%"
            + f"，近最优平衡占比 {fmt(r['hyb_near_best_balance_pct'], 1)}%。"
        )
        lines.append(
            "- "
            + f"{r['traffic']}: 最坏 P99 (RR/Age/Hybrid)=({fmt(r['worst_p99_rr'],0)}/{fmt(r['worst_p99_age'],0)}/{fmt(r['worst_p99_hyb'],0)})"
            + f"，Hybrid 对最佳基线的 P99 差距均值 {fmt(r['mean_hyb_vs_bestp99_gap_pct'])}%。"
        )

    lines.append("")
    lines.append("## 3. 论文可写的差异点建议")
    lines.append("- 若三策略均值接近，可重点写 Hybrid 的高负载稳健性（high-load P99 改善）与最坏点保护（worst P99 对比）。")
    lines.append("- 用“近最优平衡占比”作为核心卖点：强调 Hybrid 在大多数注入率上接近两端最优，而非只追求单一指标极值。")
    lines.append("- 当 Hybrid 在某些 traffic 下不绝对最优，可写成 trade-off：以较小吞吐代价换取更稳定尾延迟，或反之。")

    lines.append("")
    lines.append("## 4. 分段统计（low/mid/high）")
    lines.append("")
    lines.append("| traffic | bucket | points | hyb_vs_rr_p99% | hyb_vs_age_p99% | hyb_vs_rr_thr% | hyb_vs_age_thr% |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for _, r in bucket_df.sort_values(["traffic", "load_bucket"]).iterrows():
        lines.append(
            "| "
            + f"{r['traffic']} | {r['load_bucket']} | {int(r['points'])} | "
            + f"{fmt(r['mean_hyb_vs_rr_p99_improve_pct'])} | {fmt(r['mean_hyb_vs_age_p99_improve_pct'])} | "
            + f"{fmt(r['mean_hyb_vs_rr_thr_gain_pct'])} | {fmt(r['mean_hyb_vs_age_thr_gain_pct'])} |"
        )

    lines.append("")
    lines.append("## 5. 产物")
    lines.append("- 三策略明细：merged_three_policy.csv")
    lines.append("- 交通模式摘要：summary_by_traffic.csv")
    lines.append("- 负载分段摘要：summary_by_traffic_load_bucket.csv")
    lines.append("- 主图1：figures/curves_p99_thr_all_traffic.png")
    lines.append("- 主图2：figures/hybrid_delta_vs_baselines.png")
    lines.append("- 主图3：figures/hybrid_balance_signals.png")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    m5out = Path(args.m5out)
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    traffics = unique_keep_order(args.traffic)

    merged_list: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    bucket_list: List[pd.DataFrame] = []

    for t in traffics:
        paths = build_paths(m5out, t)
        for p in [paths.rr, paths.age, paths.hybrid]:
            if not p.exists():
                raise FileNotFoundError(f"missing input CSV: {p}")

        rr = load_summary(paths.rr)
        age = load_summary(paths.age)
        hyb = load_summary(paths.hybrid)

        m = merge_three(rr, age, hyb, t)
        if m.empty:
            continue

        merged_list.append(m)
        summary_rows.append(
            summarize_traffic(
                m,
                highload_threshold=args.highload_threshold,
                near_best_latency_eps=args.near_best_latency_eps,
                near_best_throughput_eps=args.near_best_throughput_eps,
            )
        )
        bucket_list.append(summarize_by_load_bucket(m))

    if not merged_list:
        raise RuntimeError("no aligned data points across three policies")

    merged_all = pd.concat(merged_list, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("traffic")
    bucket_df = pd.concat(bucket_list, ignore_index=True)

    merged_all.to_csv(out_dir / "merged_three_policy.csv", index=False)
    summary_df.to_csv(out_dir / "summary_by_traffic.csv", index=False)
    bucket_df.to_csv(out_dir / "summary_by_traffic_load_bucket.csv", index=False)

    plot_curves(merged_all, fig_dir / "curves_p99_thr_all_traffic.png")
    plot_hybrid_delta(merged_all, fig_dir / "hybrid_delta_vs_baselines.png")
    plot_hybrid_balance(summary_df, fig_dir / "hybrid_balance_signals.png")

    write_report(
        summary_df,
        bucket_df,
        out_dir / "analysis_report_three_policy_cn.md",
        near_best_latency_eps=args.near_best_latency_eps,
        near_best_throughput_eps=args.near_best_throughput_eps,
    )

    print(f"wrote {out_dir / 'merged_three_policy.csv'}")
    print(f"wrote {out_dir / 'summary_by_traffic.csv'}")
    print(f"wrote {out_dir / 'summary_by_traffic_load_bucket.csv'}")
    print(f"wrote {out_dir / 'analysis_report_three_policy_cn.md'}")
    print(f"wrote {fig_dir / 'curves_p99_thr_all_traffic.png'}")
    print(f"wrote {fig_dir / 'hybrid_delta_vs_baselines.png'}")
    print(f"wrote {fig_dir / 'hybrid_balance_signals.png'}")


if __name__ == "__main__":
    main()

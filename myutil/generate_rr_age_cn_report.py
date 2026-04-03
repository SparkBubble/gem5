#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd

# Prefer an explicit system-installed CJK font to avoid missing glyph warnings.
_cjk_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _cjk_font.exists():
    fm.fontManager.addfont(str(_cjk_font))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(_cjk_font)).get_name()
else:
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "DejaVu Sans",
    ]
plt.rcParams["axes.unicode_minus"] = False


def _safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.mean())


def _safe_winrate(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float((s > 0).mean() * 100.0)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for traffic, g in df.groupby("traffic"):
        g = g.sort_values("injection_rate")

        p99 = pd.to_numeric(g["packetLatencyP99_improve_pct"], errors="coerce")
        p95 = pd.to_numeric(g["packetLatencyP95_improve_pct"], errors="coerce")
        avg_lat = pd.to_numeric(g["averagePacketLatency_improve_pct"], errors="coerce")
        thr = pd.to_numeric(g["packetsReceivedPerSystemCycle_gain_pct"], errors="coerce")

        idx_best = p99.idxmax()
        idx_worst = p99.idxmin()

        high = g[g["injection_rate"] >= 0.5]
        high_p99 = pd.to_numeric(high["packetLatencyP99_improve_pct"], errors="coerce")

        rows.append(
            {
                "traffic": traffic,
                "points": int(len(g)),
                "mean_p99_improve_pct": _safe_mean(p99),
                "winrate_p99_pct": _safe_winrate(p99),
                "mean_p95_improve_pct": _safe_mean(p95),
                "mean_avg_latency_improve_pct": _safe_mean(avg_lat),
                "mean_thr_gain_pct": _safe_mean(thr),
                "highload_mean_p99_improve_pct": _safe_mean(high_p99),
                "highload_winrate_p99_pct": _safe_winrate(high_p99),
                "best_p99_improve_pct": float(p99.loc[idx_best]),
                "best_p99_rate": float(g.loc[idx_best, "injection_rate"]),
                "worst_p99_improve_pct": float(p99.loc[idx_worst]),
                "worst_p99_rate": float(g.loc[idx_worst, "injection_rate"]),
                "worst_rr_p99": float(g.loc[idx_worst, "packetLatencyP99_rr"]),
                "worst_age_p99": float(g.loc[idx_worst, "packetLatencyP99_age"]),
            }
        )

    return pd.DataFrame(rows).sort_values("traffic")


def plot_main_abs_ratio(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for traffic, g in df.groupby("traffic"):
        g = g.sort_values("injection_rate")
        x = pd.to_numeric(g["injection_rate"], errors="coerce")
        rr = pd.to_numeric(g["packetLatencyP99_rr"], errors="coerce")
        age = pd.to_numeric(g["packetLatencyP99_age"], errors="coerce")

        delta = age - rr
        ratio = age / rr.replace(0, np.nan)

        axes[0].plot(x, delta, linewidth=1.8, label=traffic)
        axes[1].plot(x, ratio, linewidth=1.8, label=traffic)

    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("主图A: Packet P99 绝对差 (Age-based-RR - RR)")
    axes[0].set_ylabel("ticks")
    axes[0].set_yscale("symlog", linthresh=1e4)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_title("主图B: Packet P99 倍率 (Age-based-RR / RR)")
    axes[1].set_xlabel("Injection Rate")
    axes[1].set_ylabel("ratio")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_appendix_pct(df: pd.DataFrame, out_p99_png: Path, out_thr_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for traffic, g in df.groupby("traffic"):
        g = g.sort_values("injection_rate")
        ax.plot(g["injection_rate"], g["packetLatencyP99_improve_pct"], linewidth=1.8, label=traffic)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("附录图A: Packet P99 相对改善率 (%)")
    ax.set_xlabel("Injection Rate")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_p99_png, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for traffic, g in df.groupby("traffic"):
        g = g.sort_values("injection_rate")
        ax.plot(
            g["injection_rate"],
            g["packetsReceivedPerSystemCycle_gain_pct"],
            linewidth=1.8,
            label=traffic,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("附录图B: 吞吐相对增益 (%)")
    ax.set_xlabel("Injection Rate")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_thr_png, dpi=170)
    plt.close(fig)


def write_cn_report(summary: pd.DataFrame, out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# RR 与 Age-based-RR 对比分析（中文）")
    lines.append("")
    lines.append("## 1. 研究目标")
    lines.append("- 在相同流量模式、相同注入率下，对比 SA-II 使用 RR 与 age-based-RR 的性能差异。")
    lines.append("- 重点关注平均时延、严格尾时延（P95/P99）、吞吐以及公平性相关指标。")
    lines.append("")
    lines.append("## 2. 方法说明")
    lines.append("- 数据来源：m5out 下 6 组 sweep 结果（RR 与 AGEBASEDRR，各含 uniform/transpose/complement）。")
    lines.append("- 对齐方式：以 injection_rate 做一一对齐，仅保留 status=ok/cached 点。")
    lines.append("- 相对改善率定义（低者优）：(RR - Age) / RR * 100%。")
    lines.append("- 吞吐增益定义（高者优）：(Age - RR) / RR * 100%。")
    lines.append("")
    lines.append("## 3. 关键结论")

    for _, r in summary.iterrows():
        lines.append(
            "- "
            + f"{r['traffic']}: "
            + f"P99 平均改善 {r['mean_p99_improve_pct']:.3f}%，"
            + f"P99 胜率 {r['winrate_p99_pct']:.1f}%，"
            + f"高负载(>=0.5) P99 平均改善 {r['highload_mean_p99_improve_pct']:.3f}%，"
            + f"吞吐平均增益 {r['mean_thr_gain_pct']:.3f}%。"
        )
        lines.append(
            "- "
            + f"{r['traffic']} 最优/最差 P99 改善点: "
            + f"best {r['best_p99_improve_pct']:.3f}% @ {r['best_p99_rate']:.2f}; "
            + f"worst {r['worst_p99_improve_pct']:.3f}% @ {r['worst_p99_rate']:.2f}."
        )

    lines.append("")
    lines.append("## 4. 关于 \"<-100%\" 的解释")
    lines.append("- 该现象是相对改善率定义导致的数学结果，不是作图错误。")
    lines.append("- 当 Age-based-RR 的 P99 超过 RR 的 2 倍时，改善率会小于 -100%。")
    lines.append("- 例如 complement 的最差点（注入率约 0.37）：")

    comp = summary[summary["traffic"] == "complement"]
    if not comp.empty:
        c = comp.iloc[0]
        lines.append(
            "  - "
            + f"RR P99={c['worst_rr_p99']:.0f}, Age P99={c['worst_age_p99']:.0f}, "
            + f"改善率={c['worst_p99_improve_pct']:.3f}%。"
        )
        ratio = c["worst_age_p99"] / c["worst_rr_p99"] if c["worst_rr_p99"] != 0 else float("nan")
        lines.append("  - " + f"倍率视角 Age/RR={ratio:.3f}x，更直观。")

    lines.append("")
    lines.append("## 5. 建议的论文展示方式")
    lines.append("- 主图使用：绝对差（ticks）+ 倍率（Age/RR），避免百分比在极值点造成误读。")
    lines.append("- 附录使用：百分比改善率曲线，用于跨指标统一比较。")
    lines.append("- 同时报告：均值、胜率、高负载分段统计、最优/最差局部点。")
    lines.append("")
    lines.append("## 6. 产物清单")
    lines.append("- 主图：figures/main_p99_abs_ratio_all_traffic.png")
    lines.append("- 附录图A：figures/appendix_p99_percent_all_traffic.png")
    lines.append("- 附录图B：figures/appendix_throughput_percent_all_traffic.png")
    lines.append("- 明细表：merged_comparison.csv")
    lines.append("- 摘要表：per_traffic_summary_cn.csv")
    lines.append("- 补充诊断：complement_p99_detailed.csv, complement_p99_worst15.csv")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out = Path("submitdata/rr_vs_age_based")
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    merged = pd.read_csv(out / "merged_comparison.csv")

    summary = build_summary(merged)
    summary.to_csv(out / "per_traffic_summary_cn.csv", index=False)

    plot_main_abs_ratio(merged, fig / "main_p99_abs_ratio_all_traffic.png")
    plot_appendix_pct(
        merged,
        fig / "appendix_p99_percent_all_traffic.png",
        fig / "appendix_throughput_percent_all_traffic.png",
    )

    write_cn_report(summary, out / "analysis_report.md")
    print(f"wrote {out / 'analysis_report.md'}")
    print(f"wrote {out / 'per_traffic_summary_cn.csv'}")
    print(f"wrote {fig / 'main_p99_abs_ratio_all_traffic.png'}")
    print(f"wrote {fig / 'appendix_p99_percent_all_traffic.png'}")
    print(f"wrote {fig / 'appendix_throughput_percent_all_traffic.png'}")


if __name__ == "__main__":
    main()

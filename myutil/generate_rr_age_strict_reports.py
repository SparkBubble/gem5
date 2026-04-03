#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _mean(s: pd.Series) -> float:
    v = _to_num(s).dropna()
    if v.empty:
        return float("nan")
    return float(v.mean())


def _win_rate_positive(s: pd.Series) -> float:
    v = _to_num(s).dropna()
    if v.empty:
        return float("nan")
    return float((v > 0).mean() * 100.0)


def _auc(x: pd.Series, y: pd.Series) -> float:
    xx = _to_num(x)
    yy = _to_num(y)
    valid = xx.notna() & yy.notna()
    if valid.sum() < 2:
        return float("nan")
    xv = xx[valid].to_numpy(dtype=float)
    yv = yy[valid].to_numpy(dtype=float)
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    return float(np.trapz(yv, xv))


def _fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _build_strict_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for traffic, g in df.groupby("traffic"):
        g = g.sort_values("injection_rate").copy()

        inj = _to_num(g["injection_rate"])

        rr_p99 = _to_num(g["packetLatencyP99_rr"])
        age_p99 = _to_num(g["packetLatencyP99_age"])
        valid_p99 = rr_p99.gt(0) & age_p99.gt(0)

        p99_imp = (rr_p99 - age_p99) / rr_p99 * 100.0
        p99_imp = p99_imp.where(valid_p99)

        rr_p95 = _to_num(g["packetLatencyP95_rr"])
        age_p95 = _to_num(g["packetLatencyP95_age"])
        valid_p95 = rr_p95.gt(0) & age_p95.gt(0)
        p95_imp = ((rr_p95 - age_p95) / rr_p95 * 100.0).where(valid_p95)

        rr_avg = _to_num(g["averagePacketLatency_rr"])
        age_avg = _to_num(g["averagePacketLatency_age"])
        valid_avg = rr_avg.gt(0) & age_avg.gt(0)
        avg_imp = ((rr_avg - age_avg) / rr_avg * 100.0).where(valid_avg)

        rr_thr = _to_num(g["packetsReceivedPerSystemCycle_rr"])
        age_thr = _to_num(g["packetsReceivedPerSystemCycle_age"])
        valid_thr = rr_thr.gt(0) & age_thr.ge(0)
        thr_gain = ((age_thr - rr_thr) / rr_thr * 100.0).where(valid_thr)

        valid_high_p99 = valid_p99 & inj.ge(0.5)

        best_idx = p99_imp.idxmax(skipna=True) if p99_imp.notna().any() else None
        worst_idx = p99_imp.idxmin(skipna=True) if p99_imp.notna().any() else None

        auc_rr_p99 = _auc(inj[valid_p99], rr_p99[valid_p99])
        auc_age_p99 = _auc(inj[valid_p99], age_p99[valid_p99])
        auc_p99_imp = float("nan")
        if not pd.isna(auc_rr_p99) and auc_rr_p99 != 0 and not pd.isna(auc_age_p99):
            auc_p99_imp = (auc_rr_p99 - auc_age_p99) / auc_rr_p99 * 100.0

        auc_rr_thr = _auc(inj[valid_thr], rr_thr[valid_thr])
        auc_age_thr = _auc(inj[valid_thr], age_thr[valid_thr])
        auc_thr_gain = float("nan")
        if not pd.isna(auc_rr_thr) and auc_rr_thr != 0 and not pd.isna(auc_age_thr):
            auc_thr_gain = (auc_age_thr - auc_rr_thr) / auc_rr_thr * 100.0

        rows.append(
            {
                "traffic": str(traffic),
                "total_points": int(len(g)),
                "valid_points_p99": int(valid_p99.sum()),
                "valid_points_p95": int(valid_p95.sum()),
                "valid_points_avg": int(valid_avg.sum()),
                "valid_points_thr": int(valid_thr.sum()),
                "mean_p99_improve_pct": _mean(p99_imp),
                "winrate_p99_pct": _win_rate_positive(p99_imp),
                "highload_mean_p99_improve_pct": _mean(p99_imp.where(valid_high_p99)),
                "highload_winrate_p99_pct": _win_rate_positive(p99_imp.where(valid_high_p99)),
                "mean_p95_improve_pct": _mean(p95_imp),
                "mean_avg_latency_improve_pct": _mean(avg_imp),
                "mean_thr_gain_pct": _mean(thr_gain),
                "auc_p99_improve_pct": auc_p99_imp,
                "auc_thr_gain_pct": auc_thr_gain,
                "best_p99_improve_pct": _as_float(p99_imp.loc[best_idx]) if best_idx is not None else float("nan"),
                "best_p99_rate": _as_float(g.loc[best_idx, "injection_rate"]) if best_idx is not None else float("nan"),
                "worst_p99_improve_pct": _as_float(p99_imp.loc[worst_idx]) if worst_idx is not None else float("nan"),
                "worst_p99_rate": _as_float(g.loc[worst_idx, "injection_rate"]) if worst_idx is not None else float("nan"),
                "worst_rr_p99": _as_float(rr_p99.loc[worst_idx]) if worst_idx is not None else float("nan"),
                "worst_age_p99": _as_float(age_p99.loc[worst_idx]) if worst_idx is not None else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values("traffic")


def _write_main_report(summary: pd.DataFrame, out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# RR vs Age-based-RR 对比（正文版，严格口径）")
    lines.append("")
    lines.append("## 1. 口径与定义")
    lines.append("- 数据源：submitdata/rr_vs_age_based/merged_comparison.csv。")
    lines.append("- 仅使用同 injection_rate 的 RR/AGE 成对样本。")
    lines.append("- 对于时延指标，仅统计 RR>0 且 AGE>0 的点。")
    lines.append("- P99 改善率定义：$(RR - AGE) / RR \\times 100\\%$。")
    lines.append("- 吞吐增益定义：$(AGE - RR) / RR \\times 100\\%$。")
    lines.append("- 高负载区间定义：injection_rate >= 0.5。")
    lines.append("")
    lines.append("## 2. 主结论（按 traffic）")

    for _, r in summary.iterrows():
        ratio = float("nan")
        if pd.notna(r["worst_rr_p99"]) and r["worst_rr_p99"] != 0 and pd.notna(r["worst_age_p99"]):
            ratio = r["worst_age_p99"] / r["worst_rr_p99"]
        lines.append(
            "- "
            + f"{r['traffic']}: P99 平均改善 {_fmt(r['mean_p99_improve_pct'])}%"
            + f"，P99 胜率 {_fmt(r['winrate_p99_pct'], 1)}%"
            + f"，高负载 P99 平均改善 {_fmt(r['highload_mean_p99_improve_pct'])}%"
            + f"，吞吐平均增益 {_fmt(r['mean_thr_gain_pct'])}%"
            + f"，P99-AUC 改善 {_fmt(r['auc_p99_improve_pct'])}%"
            + f"，吞吐-AUC 增益 {_fmt(r['auc_thr_gain_pct'])}%。"
        )
        lines.append(
            "- "
            + f"{r['traffic']} 极值: best {_fmt(r['best_p99_improve_pct'])}% @ {_fmt(r['best_p99_rate'], 2)}"
            + f"；worst {_fmt(r['worst_p99_improve_pct'])}% @ {_fmt(r['worst_p99_rate'], 2)}"
            + f"（RR={_fmt(r['worst_rr_p99'], 0)}, AGE={_fmt(r['worst_age_p99'], 0)}, AGE/RR={_fmt(ratio)}x）。"
        )

    lines.append("")
    lines.append("## 3. 可复现产物")
    lines.append("- 严格摘要表：submitdata/rr_vs_age_based/per_traffic_summary_strict_cn.csv")
    lines.append("- 正文报告：submitdata/rr_vs_age_based/analysis_report_main_cn.md")
    lines.append("- 附录报告：submitdata/rr_vs_age_based/analysis_report_appendix_cn.md")
    lines.append("")
    lines.append("## 4. 备注")
    lines.append("- 若 AGE 的 P99 超过 RR 两倍，则改善率会小于 -100%，这是定义导致的正常数学结果。")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_appendix_report(summary: pd.DataFrame, out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# RR vs Age-based-RR 对比（附录版，严格口径）")
    lines.append("")
    lines.append("## A1. 统计口径")
    lines.append("- 与正文版一致：仅统计 RR/AGE 成对有效点。")
    lines.append("- 所有数值直接由 merged_comparison.csv 复算，不依赖已有改善率列。")
    lines.append("")
    lines.append("## A2. 汇总表")
    lines.append("")
    lines.append("| traffic | total | valid_p99 | mean_p99_impr% | win_p99% | highload_mean_p99% | highload_win_p99% | mean_p95_impr% | mean_avg_lat_impr% | mean_thr_gain% | auc_p99_impr% | auc_thr_gain% | best_p99% @rate | worst_p99% @rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")

    for _, r in summary.iterrows():
        best = f"{_fmt(r['best_p99_improve_pct'])}% @{_fmt(r['best_p99_rate'], 2)}"
        worst = f"{_fmt(r['worst_p99_improve_pct'])}% @{_fmt(r['worst_p99_rate'], 2)}"
        lines.append(
            "| "
            + f"{r['traffic']} | {int(r['total_points'])} | {int(r['valid_points_p99'])} | "
            + f"{_fmt(r['mean_p99_improve_pct'])} | {_fmt(r['winrate_p99_pct'], 1)} | "
            + f"{_fmt(r['highload_mean_p99_improve_pct'])} | {_fmt(r['highload_winrate_p99_pct'], 1)} | "
            + f"{_fmt(r['mean_p95_improve_pct'])} | {_fmt(r['mean_avg_latency_improve_pct'])} | "
            + f"{_fmt(r['mean_thr_gain_pct'])} | {_fmt(r['auc_p99_improve_pct'])} | {_fmt(r['auc_thr_gain_pct'])} | "
            + f"{best} | {worst} |"
        )

    lines.append("")
    lines.append("## A3. 极值点解释")
    for _, r in summary.iterrows():
        ratio = float("nan")
        if pd.notna(r["worst_rr_p99"]) and r["worst_rr_p99"] != 0 and pd.notna(r["worst_age_p99"]):
            ratio = r["worst_age_p99"] / r["worst_rr_p99"]
        lines.append(
            "- "
            + f"{r['traffic']} worst 点：RR={_fmt(r['worst_rr_p99'], 0)}, "
            + f"AGE={_fmt(r['worst_age_p99'], 0)}, "
            + f"AGE/RR={_fmt(ratio)}x, 改善率={_fmt(r['worst_p99_improve_pct'])}% @ {_fmt(r['worst_p99_rate'], 2)}。"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = Path("submitdata/rr_vs_age_based")
    merged_path = out_dir / "merged_comparison.csv"

    if not merged_path.exists():
        raise FileNotFoundError(f"missing file: {merged_path}")

    df = pd.read_csv(merged_path)
    summary = _build_strict_summary(df)

    summary_csv = out_dir / "per_traffic_summary_strict_cn.csv"
    main_md = out_dir / "analysis_report_main_cn.md"
    appendix_md = out_dir / "analysis_report_appendix_cn.md"

    summary.to_csv(summary_csv, index=False)
    _write_main_report(summary, main_md)
    _write_appendix_report(summary, appendix_md)

    print(f"wrote {summary_csv}")
    print(f"wrote {main_md}")
    print(f"wrote {appendix_md}")


if __name__ == "__main__":
    main()

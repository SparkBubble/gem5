#!/usr/bin/env python3
"""Adaptive injection-rate sweep runner for gem5 Garnet synthetic traffic.

Features:
- Coarse sweep over [min_rate, max_rate]
- Detect steep intervals and run fine-grained points there
- Run gem5 command per point and parse stats via util/extract_garnet_stats.py
- Generate CSV + Markdown summary + curve plots

Example:
  /usr/bin/python3 util/sweep_garnet_injection.py \
    --gem5-bin ./build/NULL/gem5.opt \
    --config configs/example/garnet_synth_traffic.py \
    --num-cpus 4 --num-dirs 4 --network garnet --topology Mesh_XY --mesh-rows 2 \
    --sim-cycles 10000000 --synthetic uniform_random \
    --workdir m5out/sweep_adaptive
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class RunResult:
    injection_rate: float
    stage: str
    status: str
    outdir: str
    run_seconds: float
    error: str = ""
    metrics: Optional[Dict[str, Any]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive sweep for gem5 Garnet injection rate")

    parser.add_argument("--gem5-bin", default="./build/NULL/gem5.opt")
    parser.add_argument("--config", default="configs/example/garnet_synth_traffic.py")
    parser.add_argument("--extractor", default="util/extract_garnet_stats.py")
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--num-dirs", type=int, default=4)
    parser.add_argument("--network", default="garnet")
    parser.add_argument("--topology", default="Mesh_XY")
    parser.add_argument("--mesh-rows", type=int, default=2)
    parser.add_argument("--sim-cycles", type=int, default=10_000_000)
    parser.add_argument("--synthetic", default="uniform_random")

    parser.add_argument("--min-rate", type=float, default=0.0)
    parser.add_argument("--max-rate", type=float, default=1.0)
    parser.add_argument("--coarse-step", type=float, default=0.1)
    parser.add_argument("--fine-step", type=float, default=0.02)

    parser.add_argument("--lat-steep-ratio", type=float, default=0.30)
    parser.add_argument("--thr-steep-ratio", type=float, default=0.20)
    parser.add_argument("--max-fine-intervals", type=int, default=6)

    parser.add_argument("--workdir", default="m5out/sweep_adaptive")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--plot-format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots",
    )

    parser.add_argument(
        "--extra-gem5-arg",
        action="append",
        default=[],
        help="Extra argument passed to garnet_synth_traffic.py (repeatable)",
    )

    return parser.parse_args()


def frange(start: float, stop: float, step: float) -> List[float]:
    vals: List[float] = []
    i = 0
    while True:
        v = start + i * step
        if v > stop + 1e-12:
            break
        vals.append(round(v, 10))
        i += 1
    if not vals or abs(vals[-1] - stop) > 1e-9:
        vals.append(round(stop, 10))
    return vals


def rate_tag(rate: float) -> str:
    # fixed-width tag to keep lexicographic order aligned with numeric order
    return f"{rate:.4f}".replace(".", "p")


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    garnet = metrics.get("garnet", {})
    latency = garnet.get("latency", {})
    throughput = garnet.get("throughput", {})
    link = garnet.get("linkUtilization", {})
    fairness = metrics.get("fairness", {}).get("sa2", {})
    fairness_totals = fairness.get("totals", {})
    fairness_dist = fairness.get("distributions", {})
    service_stats = fairness_dist.get("serviceRatioStats", {})
    runtime = metrics.get("runtime", {})

    return {
        "simTicks": runtime.get("simTicks"),
        "systemCycles": runtime.get("systemCycles"),
        "rubyCycles": runtime.get("rubyCycles"),
        "averagePacketLatency": latency.get("averagePacketLatency"),
        "averagePacketNetworkLatency": latency.get("averagePacketNetworkLatency"),
        "averagePacketQueueingLatency": latency.get("averagePacketQueueingLatency"),
        "averageFlitLatency": latency.get("averageFlitLatency"),
        "averageHops": latency.get("averageHops"),
        "queueingShareOfPacketLatency": latency.get("queueingShareOfPacketLatency"),
        "packetsInjectedTotal": throughput.get("packetsInjectedTotal"),
        "packetsReceivedTotal": throughput.get("packetsReceivedTotal"),
        "flitsInjectedTotal": throughput.get("flitsInjectedTotal"),
        "flitsReceivedTotal": throughput.get("flitsReceivedTotal"),
        "packetDeliveryRatio": throughput.get("packetDeliveryRatio"),
        "flitDeliveryRatio": throughput.get("flitDeliveryRatio"),
        "packetsReceivedPerSystemCycle": throughput.get("packetsReceivedPerSystemCycle"),
        "packetsReceivedPerRubyCycle": throughput.get("packetsReceivedPerRubyCycle"),
        "flitsReceivedPerSystemCycle": throughput.get("flitsReceivedPerSystemCycle"),
        "flitsReceivedPerRubyCycle": throughput.get("flitsReceivedPerRubyCycle"),
        "avgLinkUtilization": link.get("avgLinkUtilization"),
        "extInLinkUtilization": link.get("extInLinkUtilization"),
        "extOutLinkUtilization": link.get("extOutLinkUtilization"),
        "intLinkUtilization": link.get("intLinkUtilization"),
        "sa2Available": fairness.get("available"),
        "sa2TotalRequests": fairness_totals.get("requests"),
        "sa2TotalGrants": fairness_totals.get("grants"),
        "sa2TotalDenials": fairness_totals.get("denials"),
        "sa2GrantRatio": fairness_totals.get("grantRatio"),
        "sa2RequestJain": fairness_dist.get("requestJainIndex"),
        "sa2GrantJain": fairness_dist.get("grantJainIndex"),
        "sa2ServiceRatioJain": fairness_dist.get("serviceRatioJainIndex"),
        "sa2ServiceRatioGini": service_stats.get("gini"),
        "sa2ServiceRatioCv": service_stats.get("cv"),
        "sa2ServiceRatioMaxMinRatio": service_stats.get("maxMinRatio"),
        "sa2ServiceRatioP90P10Ratio": service_stats.get("p90p10Ratio"),
    }


def run_cmd(cmd: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_one_rate(
    args: argparse.Namespace,
    repo_root: Path,
    workdir: Path,
    rate: float,
    stage: str,
) -> RunResult:
    outdir = workdir / f"rate_{rate_tag(rate)}"
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = outdir / "garnet_metrics.json"
    stats_path = outdir / "stats.txt"

    if metrics_path.exists() and not args.force_rerun:
        try:
            metrics = load_json(metrics_path)
            return RunResult(
                injection_rate=rate,
                stage=stage,
                status="cached",
                outdir=str(outdir),
                run_seconds=0.0,
                metrics=metrics,
            )
        except Exception as exc:  # noqa: BLE001
            return RunResult(
                injection_rate=rate,
                stage=stage,
                status="failed",
                outdir=str(outdir),
                run_seconds=0.0,
                error=f"failed to load cached metrics: {exc}",
            )

    gem5_cmd = [
        args.gem5_bin,
        "-d",
        str(outdir),
        args.config,
        f"--num-cpus={args.num_cpus}",
        f"--num-dirs={args.num_dirs}",
        f"--network={args.network}",
        f"--topology={args.topology}",
        f"--mesh-rows={args.mesh_rows}",
        f"--sim-cycles={args.sim_cycles}",
        f"--synthetic={args.synthetic}",
        f"--injectionrate={rate:.6f}",
    ]

    for ext in args.extra_gem5_arg:
        gem5_cmd.append(ext)

    if args.dry_run:
        return RunResult(
            injection_rate=rate,
            stage=stage,
            status="dry-run",
            outdir=str(outdir),
            run_seconds=0.0,
        )

    start = time.time()
    run_res = run_cmd(gem5_cmd, cwd=repo_root)
    elapsed = time.time() - start

    if run_res.returncode != 0:
        err = (run_res.stderr or "")[-4000:]
        out = (run_res.stdout or "")[-4000:]
        return RunResult(
            injection_rate=rate,
            stage=stage,
            status="failed",
            outdir=str(outdir),
            run_seconds=elapsed,
            error=f"gem5 failed (code {run_res.returncode})\nstdout:\n{out}\nstderr:\n{err}",
        )

    if not stats_path.exists():
        return RunResult(
            injection_rate=rate,
            stage=stage,
            status="failed",
            outdir=str(outdir),
            run_seconds=elapsed,
            error="gem5 finished but stats.txt not found",
        )

    extract_cmd = [
        args.python,
        args.extractor,
        "--stats",
        str(stats_path),
        "--out",
        str(metrics_path),
    ]
    ext_res = run_cmd(extract_cmd, cwd=repo_root)
    if ext_res.returncode != 0 or not metrics_path.exists():
        err = (ext_res.stderr or "")[-4000:]
        out = (ext_res.stdout or "")[-4000:]
        return RunResult(
            injection_rate=rate,
            stage=stage,
            status="failed",
            outdir=str(outdir),
            run_seconds=elapsed,
            error=f"extractor failed (code {ext_res.returncode})\nstdout:\n{out}\nstderr:\n{err}",
        )

    try:
        metrics = load_json(metrics_path)
    except Exception as exc:  # noqa: BLE001
        return RunResult(
            injection_rate=rate,
            stage=stage,
            status="failed",
            outdir=str(outdir),
            run_seconds=elapsed,
            error=f"failed to parse extractor output: {exc}",
        )

    return RunResult(
        injection_rate=rate,
        stage=stage,
        status="ok",
        outdir=str(outdir),
        run_seconds=elapsed,
        metrics=metrics,
    )


def safe_num(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:  # noqa: BLE001
        return None


def detect_steep_intervals(
    rows: List[Dict[str, Any]],
    lat_steep_ratio: float,
    thr_steep_ratio: float,
    max_intervals: int,
) -> List[Tuple[float, float]]:
    rows_sorted = sorted(rows, key=lambda x: x["injection_rate"])
    scored: List[Tuple[float, float, float]] = []
    scored_all: List[Tuple[float, float, float]] = []

    for i in range(len(rows_sorted) - 1):
        a = rows_sorted[i]
        b = rows_sorted[i + 1]

        r1, r2 = a["injection_rate"], b["injection_rate"]
        if r2 <= r1:
            continue

        l1 = safe_num(a.get("averagePacketLatency"))
        l2 = safe_num(b.get("averagePacketLatency"))
        t1 = safe_num(a.get("packetsReceivedPerSystemCycle"))
        t2 = safe_num(b.get("packetsReceivedPerSystemCycle"))

        lat_ratio = 0.0
        thr_ratio = 0.0

        if l1 is not None and l2 is not None and l1 > 0:
            lat_ratio = (l2 - l1) / l1

        if t1 is not None and t2 is not None and abs(t1) > 1e-12:
            thr_ratio = abs((t2 - t1) / t1)

        if lat_ratio >= lat_steep_ratio or thr_ratio >= thr_steep_ratio:
            score = max(lat_ratio / max(lat_steep_ratio, 1e-12), thr_ratio / max(thr_steep_ratio, 1e-12))
            scored.append((score, r1, r2))

        # Keep a relative score for fallback use when strict thresholds find nothing.
        relative = max(lat_ratio / max(lat_steep_ratio, 1e-12), thr_ratio / max(thr_steep_ratio, 1e-12))
        scored_all.append((relative, r1, r2))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[: max(0, max_intervals)]

    # Fallback: if no interval passes strict thresholds, still refine top relative slopes.
    if not chosen and scored_all:
        scored_all.sort(key=lambda x: x[0], reverse=True)
        fallback_n = min(max(1, min(2, max_intervals)), len(scored_all))
        chosen = scored_all[:fallback_n]

    intervals = [(r1, r2) for _, r1, r2 in chosen]
    intervals = sorted(intervals)
    merged: List[Tuple[float, float]] = []

    for s, e in intervals:
        if not merged or s > merged[-1][1] + 1e-12:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    return merged


def build_fine_points(intervals: List[Tuple[float, float]], fine_step: float) -> List[float]:
    fine: List[float] = []
    for s, e in intervals:
        points = frange(s + fine_step, e - fine_step, fine_step)
        for p in points:
            if s < p < e:
                fine.append(round(p, 10))
    return sorted(set(fine))


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(rows: List[Dict[str, Any]], path: Path, top_n: int = 50) -> None:
    if not rows:
        path.write_text("No data\n", encoding="utf-8")
        return

    show = rows[:top_n]
    cols = [
        "injection_rate",
        "stage",
        "status",
        "averagePacketLatency",
        "packetsReceivedPerSystemCycle",
        "packetDeliveryRatio",
        "sa2GrantRatio",
        "sa2ServiceRatioJain",
        "sa2ServiceRatioGini",
        "sa2ServiceRatioCv",
        "sa2ServiceRatioMaxMinRatio",
        "avgLinkUtilization",
    ]

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for row in show:
        vals = []
        for c in cols:
            v = row.get(c)
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v) if v is not None else "")
        lines.append("| " + " | ".join(vals) + " |")

    if len(rows) > top_n:
        lines.append("")
        lines.append(f"Only first {top_n} rows shown. Full data in CSV.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_plots(rows: List[Dict[str, Any]], outdir: Path, fmt: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return "matplotlib not available; skipped plot generation"

    rows_ok = [r for r in rows if r.get("status") in {"ok", "cached"}]
    rows_ok.sort(key=lambda r: r["injection_rate"])
    if not rows_ok:
        return "no successful rows to plot"

    x = [r["injection_rate"] for r in rows_ok]

    y_lat = [safe_num(r.get("averagePacketLatency")) for r in rows_ok]
    y_thr = [safe_num(r.get("packetsReceivedPerSystemCycle")) for r in rows_ok]
    y_sa2_grant_ratio = [safe_num(r.get("sa2GrantRatio")) for r in rows_ok]
    y_sr_jain = [safe_num(r.get("sa2ServiceRatioJain")) for r in rows_ok]
    y_sr_gini = [safe_num(r.get("sa2ServiceRatioGini")) for r in rows_ok]
    y_sr_cv = [safe_num(r.get("sa2ServiceRatioCv")) for r in rows_ok]

    fig, axes = plt.subplots(4, 1, figsize=(9, 15), sharex=True)

    axes[0].plot(x, y_lat, marker="o", linewidth=1.5)
    axes[0].set_ylabel("Avg Packet Latency")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Garnet Sweep: Latency vs Injection Rate")

    axes[1].plot(x, y_thr, marker="o", color="tab:green", linewidth=1.5)
    axes[1].set_ylabel("Pkt Throughput\n(pkt/system_cycle)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Throughput vs Injection Rate")

    axes[2].plot(x, y_sa2_grant_ratio, marker="o", color="tab:orange", linewidth=1.5)
    axes[2].set_xlabel("Injection Rate")
    axes[2].set_ylabel("Grant Ratio")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("SA-II Grant Ratio vs Injection Rate")

    axes[3].plot(x, y_sr_jain, marker="o", color="tab:purple", linewidth=1.5, label="ServiceRatio Jain")
    axes[3].plot(x, y_sr_gini, marker="s", color="tab:brown", linewidth=1.2, label="ServiceRatio Gini")
    axes[3].plot(x, y_sr_cv, marker="^", color="tab:blue", linewidth=1.0, label="ServiceRatio CV")
    axes[3].set_xlabel("Injection Rate")
    axes[3].set_ylabel("Fairness Metrics")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title("SA-II Service-Ratio Fairness")
    axes[3].legend()

    fig.tight_layout()
    out_file = outdir / f"sweep_curves.{fmt}"
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    return None


def summarize_failures(results: List[RunResult]) -> str:
    fails = [r for r in results if r.status == "failed"]
    if not fails:
        return ""

    lines = ["Failed points:"]
    for r in fails:
        lines.append(f"- rate={r.injection_rate:.4f}, stage={r.stage}, outdir={r.outdir}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()

    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = repo_root / workdir
    workdir.mkdir(parents=True, exist_ok=True)

    coarse_rates = frange(args.min_rate, args.max_rate, args.coarse_step)

    all_results: List[RunResult] = []

    print(f"[1/4] Running coarse sweep on {len(coarse_rates)} points")
    for rate in coarse_rates:
        res = run_one_rate(args, repo_root, workdir, rate, stage="coarse")
        all_results.append(res)
        print(f"  rate={rate:.4f} stage=coarse status={res.status} time={res.run_seconds:.2f}s")

    coarse_rows: List[Dict[str, Any]] = []
    for r in all_results:
        if r.metrics is None:
            continue
        row = {
            "injection_rate": r.injection_rate,
            "stage": r.stage,
            "status": r.status,
            "outdir": r.outdir,
            "run_seconds": round(r.run_seconds, 6),
        }
        row.update(flatten_metrics(r.metrics))
        coarse_rows.append(row)

    intervals = detect_steep_intervals(
        coarse_rows,
        lat_steep_ratio=args.lat_steep_ratio,
        thr_steep_ratio=args.thr_steep_ratio,
        max_intervals=args.max_fine_intervals,
    )

    fine_rates = build_fine_points(intervals, args.fine_step)
    coarse_set = {round(r, 10) for r in coarse_rates}
    fine_rates = [r for r in fine_rates if round(r, 10) not in coarse_set]

    print(f"[2/4] Steep intervals detected: {intervals if intervals else 'none'}")
    print(f"      Fine points to run: {len(fine_rates)}")

    for rate in fine_rates:
        res = run_one_rate(args, repo_root, workdir, rate, stage="fine")
        all_results.append(res)
        print(f"  rate={rate:.4f} stage=fine   status={res.status} time={res.run_seconds:.2f}s")

    final_rows: List[Dict[str, Any]] = []
    for r in all_results:
        row = {
            "injection_rate": r.injection_rate,
            "stage": r.stage,
            "status": r.status,
            "outdir": r.outdir,
            "run_seconds": round(r.run_seconds, 6),
            "error": r.error,
        }
        if r.metrics is not None:
            row.update(flatten_metrics(r.metrics))
        final_rows.append(row)

    final_rows.sort(key=lambda x: (x["injection_rate"], 0 if x["stage"] == "coarse" else 1))

    csv_path = workdir / "sweep_summary.csv"
    md_path = workdir / "sweep_summary.md"
    json_path = workdir / "sweep_results.json"

    print("[3/4] Writing summary artifacts")
    write_csv(final_rows, csv_path)
    write_markdown_table(final_rows, md_path)
    json_path.write_text(json.dumps(final_rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print("[4/4] Generating plots")
    plot_err = draw_plots(final_rows, workdir, args.plot_format)

    print("Done")
    print(f"- CSV:  {csv_path}")
    print(f"- MD:   {md_path}")
    print(f"- JSON: {json_path}")
    print(f"- Plot: {workdir / ('sweep_curves.' + args.plot_format)}")

    fail_text = summarize_failures(all_results)
    if fail_text:
        print(fail_text)

    if plot_err:
        print(f"Plot note: {plot_err}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract useful metrics from gem5 stats.txt with Garnet-focused analytics.

Usage:
  python3 util/extract_garnet_stats.py --stats m5out/stats.txt --pretty
  python3 util/extract_garnet_stats.py --stats m5out/stats.txt --out m5out/garnet_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STAT_LINE_RE = re.compile(r"^(\S+)\s+(.*)$")
UNITS_TAIL_RE = re.compile(r"\s+\([^)]*\)\s*$")
TRAFFIC_DIST_RE = re.compile(
    r"^system\.ruby\.network\.([a-zA-Z0-9_]+_traffic_distribution)\.n(\d+)\.n(\d+)$"
)
MEM_CTRL_RE = re.compile(r"^system\.mem_ctrls(\d+)\.(.+)$")
ROUTER_SA2_SCALAR_RE = re.compile(
    r"^system\.ruby\.network\.routers(\d+)\.(sa2_total_requests|sa2_total_grants|sa2_total_denials)$"
)
ROUTER_SA2_INPORT_RE = re.compile(
    r"^system\.ruby\.network\.routers(\d+)\.(sa2_inport_requests|sa2_inport_grants|sa2_inport_denials)$"
)
ROUTER_SA2_INPORT_INDEX_RE = re.compile(
    r"^system\.ruby\.network\.routers(\d+)\.(sa2_inport_requests|sa2_inport_grants|sa2_inport_denials)::(\d+)$"
)


@dataclass
class ParsedValue:
    raw: str
    scalar: Optional[float] = None
    vector: Optional[List[float]] = None
    vector_raw: Optional[List[str]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract useful metrics from gem5 stats.txt")
    parser.add_argument("--stats", default="m5out/stats.txt", help="Path to stats.txt")
    parser.add_argument("--out", default=None, help="Write JSON output to this path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include all parsed raw keys (can be large)",
    )
    return parser.parse_args()


def to_number(token: str) -> Optional[float]:
    token = token.strip()
    if not token:
        return None

    low = token.lower()
    if low == "nan":
        return math.nan
    if low == "inf":
        return math.inf
    if low == "-inf":
        return -math.inf

    # Strip trailing percent sign if it exists.
    if token.endswith("%"):
        token = token[:-1]

    try:
        return float(token)
    except ValueError:
        return None


def parse_value(value: str) -> ParsedValue:
    value = value.strip()
    if "|" not in value:
        return ParsedValue(raw=value, scalar=to_number(value))

    # Vector-like values appear as: | a ... | b ... | c ...
    segments = [seg.strip() for seg in value.split("|") if seg.strip()]
    numbers: List[float] = []
    for seg in segments:
        first_token = seg.split()[0] if seg.split() else ""
        num = to_number(first_token)
        numbers.append(num if num is not None else math.nan)

    return ParsedValue(raw=value, vector=numbers, vector_raw=segments)


def parse_stats_file(path: Path) -> Dict[str, ParsedValue]:
    parsed: Dict[str, ParsedValue] = {}
    if not path.exists():
        raise FileNotFoundError(f"stats file not found: {path}")

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("----------"):
            continue

        # Remove trailing hash comments, keep the metric payload only.
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()

        # Remove trailing "(Unspecified)" or other unit tags.
        line = UNITS_TAIL_RE.sub("", line)

        m = STAT_LINE_RE.match(line)
        if not m:
            continue

        key, value = m.group(1), m.group(2).strip()
        if not value:
            continue
        parsed[key] = parse_value(value)

    return parsed


def get_scalar(parsed: Dict[str, ParsedValue], key: str) -> Optional[float]:
    item = parsed.get(key)
    return item.scalar if item else None


def get_vector(parsed: Dict[str, ParsedValue], key: str) -> Optional[List[float]]:
    item = parsed.get(key)
    return item.vector if item else None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0 or math.isnan(b):
        return None
    if math.isnan(a):
        return None
    return a / b


def jain_index(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    total = sum(vals)
    if total == 0:
        return None
    n = len(vals)
    denom = n * sum(v * v for v in vals)
    if denom == 0:
        return None
    return (total * total) / denom


def clean_values(values: List[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        if math.isnan(v) or math.isinf(v):
            continue
        out.append(float(v))
    return out


def percentile(values: List[float], p: float) -> Optional[float]:
    vals = sorted(clean_values(values))
    if not vals:
        return None
    if p <= 0:
        return vals[0]
    if p >= 100:
        return vals[-1]

    pos = (len(vals) - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def gini_index(values: List[float]) -> Optional[float]:
    vals = sorted(clean_values(values))
    n = len(vals)
    if n == 0:
        return None
    s = sum(vals)
    if s == 0:
        return None
    # Gini = (2*sum(i*x_i)/(n*sum(x))) - (n+1)/n with i starting from 1 on sorted values.
    weighted = 0.0
    for i, v in enumerate(vals, start=1):
        weighted += i * v
    return (2.0 * weighted) / (n * s) - (n + 1) / n


def distribution_metrics(values: List[float]) -> Dict[str, Any]:
    vals = clean_values(values)
    if not vals:
        return {
            "count": 0,
            "activeCount": 0,
            "mean": None,
            "std": None,
            "cv": None,
            "min": None,
            "max": None,
            "range": None,
            "maxMinRatio": None,
            "p90": None,
            "p10": None,
            "p90p10Ratio": None,
            "gini": None,
        }

    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(var)
    vmin = min(vals)
    vmax = max(vals)
    p90 = percentile(vals, 90)
    p10 = percentile(vals, 10)

    max_min_ratio = None
    if vmin > 0:
        max_min_ratio = vmax / vmin

    p90_p10_ratio = None
    if p10 is not None and p10 > 0 and p90 is not None:
        p90_p10_ratio = p90 / p10

    cv = None
    if mean != 0:
        cv = std / mean

    return {
        "count": n,
        "activeCount": sum(1 for v in vals if v > 0),
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": vmin,
        "max": vmax,
        "range": vmax - vmin,
        "maxMinRatio": max_min_ratio,
        "p90": p90,
        "p10": p10,
        "p90p10Ratio": p90_p10_ratio,
        "gini": gini_index(vals),
    }


def build_runtime_summary(parsed: Dict[str, ParsedValue]) -> Dict[str, Any]:
    sim_ticks = get_scalar(parsed, "simTicks")
    sim_seconds = get_scalar(parsed, "simSeconds")
    host_tick_rate = get_scalar(parsed, "hostTickRate")
    sys_clk = get_scalar(parsed, "system.clk_domain.clock")
    ruby_clk = get_scalar(parsed, "system.ruby.clk_domain.clock")

    return {
        "simTicks": sim_ticks,
        "simSeconds": sim_seconds,
        "hostTickRate": host_tick_rate,
        "systemClockTickPeriod": sys_clk,
        "rubyClockTickPeriod": ruby_clk,
        "systemCycles": safe_div(sim_ticks, sys_clk),
        "rubyCycles": safe_div(sim_ticks, ruby_clk),
    }


def build_garnet_summary(parsed: Dict[str, ParsedValue], runtime: Dict[str, Any]) -> Dict[str, Any]:
    pkt_inj_total = get_scalar(parsed, "system.ruby.network.packets_injected::total")
    pkt_rcv_total = get_scalar(parsed, "system.ruby.network.packets_received::total")
    flit_inj_total = get_scalar(parsed, "system.ruby.network.flits_injected::total")
    flit_rcv_total = get_scalar(parsed, "system.ruby.network.flits_received::total")

    sys_cycles = runtime.get("systemCycles")
    ruby_cycles = runtime.get("rubyCycles")

    pkt_inj_vec = get_vector(parsed, "system.ruby.network.packets_injected")
    pkt_rcv_vec = get_vector(parsed, "system.ruby.network.packets_received")
    flit_inj_vec = get_vector(parsed, "system.ruby.network.flits_injected")
    flit_rcv_vec = get_vector(parsed, "system.ruby.network.flits_received")

    packet_latency = get_scalar(parsed, "system.ruby.network.average_packet_latency")
    packet_network_latency = get_scalar(parsed, "system.ruby.network.average_packet_network_latency")
    packet_queue_latency = get_scalar(parsed, "system.ruby.network.average_packet_queueing_latency")
    flit_latency = get_scalar(parsed, "system.ruby.network.average_flit_latency")

    return {
        "latency": {
            "averagePacketLatency": packet_latency,
            "averagePacketNetworkLatency": packet_network_latency,
            "averagePacketQueueingLatency": packet_queue_latency,
            "averageFlitLatency": flit_latency,
            "averageHops": get_scalar(parsed, "system.ruby.network.average_hops"),
            "queueingShareOfPacketLatency": safe_div(packet_queue_latency, packet_latency),
            "packetLatencySamples": get_scalar(parsed, "system.ruby.network.packet_latency_samples"),
            "packetLatencyMin": get_scalar(parsed, "system.ruby.network.packet_latency_min"),
            "packetLatencyP50": get_scalar(parsed, "system.ruby.network.packet_latency_p50"),
            "packetLatencyP95": get_scalar(parsed, "system.ruby.network.packet_latency_p95"),
            "packetLatencyP99": get_scalar(parsed, "system.ruby.network.packet_latency_p99"),
            "packetLatencyMax": get_scalar(parsed, "system.ruby.network.packet_latency_max"),
            "flitLatencySamples": get_scalar(parsed, "system.ruby.network.flit_latency_samples"),
            "flitLatencyMin": get_scalar(parsed, "system.ruby.network.flit_latency_min"),
            "flitLatencyP50": get_scalar(parsed, "system.ruby.network.flit_latency_p50"),
            "flitLatencyP95": get_scalar(parsed, "system.ruby.network.flit_latency_p95"),
            "flitLatencyP99": get_scalar(parsed, "system.ruby.network.flit_latency_p99"),
            "flitLatencyMax": get_scalar(parsed, "system.ruby.network.flit_latency_max"),
        },
        "throughput": {
            "packetsInjectedTotal": pkt_inj_total,
            "packetsReceivedTotal": pkt_rcv_total,
            "flitsInjectedTotal": flit_inj_total,
            "flitsReceivedTotal": flit_rcv_total,
            "packetDeliveryRatio": safe_div(pkt_rcv_total, pkt_inj_total),
            "flitDeliveryRatio": safe_div(flit_rcv_total, flit_inj_total),
            "packetsReceivedPerSystemCycle": safe_div(pkt_rcv_total, sys_cycles),
            "packetsReceivedPerRubyCycle": safe_div(pkt_rcv_total, ruby_cycles),
            "flitsReceivedPerSystemCycle": safe_div(flit_rcv_total, sys_cycles),
            "flitsReceivedPerRubyCycle": safe_div(flit_rcv_total, ruby_cycles),
        },
        "linkUtilization": {
            "avgLinkUtilization": get_scalar(parsed, "system.ruby.network.avg_link_utilization"),
            "extInLinkUtilization": get_scalar(parsed, "system.ruby.network.ext_in_link_utilization"),
            "extOutLinkUtilization": get_scalar(parsed, "system.ruby.network.ext_out_link_utilization"),
            "intLinkUtilization": get_scalar(parsed, "system.ruby.network.int_link_utilization"),
        },
        "perVnet": {
            "packetsInjected": pkt_inj_vec,
            "packetsReceived": pkt_rcv_vec,
            "flitsInjected": flit_inj_vec,
            "flitsReceived": flit_rcv_vec,
            "averagePacketVnetLatency": get_vector(parsed, "system.ruby.network.average_packet_vnet_latency"),
            "averagePacketVqueueLatency": get_vector(parsed, "system.ruby.network.average_packet_vqueue_latency"),
            "averageFlitVnetLatency": get_vector(parsed, "system.ruby.network.average_flit_vnet_latency"),
            "averageFlitVqueueLatency": get_vector(parsed, "system.ruby.network.average_flit_vqueue_latency"),
        },
    }


def build_tail_latency_summary(parsed: Dict[str, ParsedValue]) -> Dict[str, Any]:
    direct_percentiles: Dict[str, float] = {}
    histogram_keys: List[str] = []

    for key, val in parsed.items():
        low = key.lower()
        if "p95" in low or "p99" in low or "percentile" in low:
            if val.scalar is not None:
                direct_percentiles[key] = val.scalar
        if "latency" in low and ("hist" in low or "pdf::" in low or "bucket" in low):
            histogram_keys.append(key)

    return {
        "hasDirectPercentiles": bool(direct_percentiles),
        "directPercentiles": direct_percentiles,
        "hasLatencyHistogramCandidates": bool(histogram_keys),
        "latencyHistogramCandidateCount": len(histogram_keys),
        "suggestion": (
            "No direct p95/p99 found in stats; enable packet-latency histogram or add per-packet tracing "
            "in Garnet network interface to compute p95/p99 offline."
            if not direct_percentiles
            else "Direct percentile metrics detected."
        ),
    }


def build_fairness_summary(parsed: Dict[str, ParsedValue]) -> Dict[str, Any]:
    routers: Dict[int, Dict[str, Any]] = defaultdict(dict)

    for key, val in parsed.items():
        m_scalar = ROUTER_SA2_SCALAR_RE.match(key)
        if m_scalar and val.scalar is not None and not math.isnan(val.scalar):
            router_id = int(m_scalar.group(1))
            field = m_scalar.group(2)
            routers[router_id][field] = val.scalar
            continue

        m_vec = ROUTER_SA2_INPORT_RE.match(key)
        if m_vec and val.vector is not None:
            router_id = int(m_vec.group(1))
            field = m_vec.group(2)
            routers[router_id][field] = [
                x for x in val.vector if x is not None and not math.isnan(x)
            ]
            continue

        m_idx = ROUTER_SA2_INPORT_INDEX_RE.match(key)
        if m_idx and val.scalar is not None and not math.isnan(val.scalar):
            router_id = int(m_idx.group(1))
            field = m_idx.group(2)
            idx = int(m_idx.group(3))
            existing = routers[router_id].setdefault(field, [])
            while len(existing) <= idx:
                existing.append(0.0)
            existing[idx] = val.scalar

    if not routers:
        return {
            "note": "SA-II fairness unavailable: router sa2_* stats not found in stats.txt.",
            "sa2": {
                "available": False,
                "totals": {
                    "requests": None,
                    "grants": None,
                    "denials": None,
                    "grantRatio": None,
                },
                "distributions": {
                    "requestPerInportStats": distribution_metrics([]),
                    "grantPerInportStats": distribution_metrics([]),
                    "denialPerInportStats": distribution_metrics([]),
                    "serviceRatioStats": distribution_metrics([]),
                    "requestJainIndex": None,
                    "grantJainIndex": None,
                    "serviceRatioJainIndex": None,
                },
                "perRouter": {},
            },
        }

    total_requests = 0.0
    total_grants = 0.0
    total_denials = 0.0

    req_bins: List[float] = []
    grant_bins: List[float] = []
    denial_bins: List[float] = []
    service_ratio_bins: List[float] = []

    per_router_out: Dict[str, Any] = {}

    for rid in sorted(routers.keys()):
        data = routers[rid]

        req_total = float(data.get("sa2_total_requests", 0.0))
        grant_total = float(data.get("sa2_total_grants", 0.0))
        denial_total = float(data.get("sa2_total_denials", 0.0))

        in_reqs = [float(x) for x in data.get("sa2_inport_requests", [])]
        in_grants = [float(x) for x in data.get("sa2_inport_grants", [])]
        in_denials = [float(x) for x in data.get("sa2_inport_denials", [])]

        width = max(len(in_reqs), len(in_grants), len(in_denials))
        if width > 0:
            if len(in_reqs) < width:
                in_reqs.extend([0.0] * (width - len(in_reqs)))
            if len(in_grants) < width:
                in_grants.extend([0.0] * (width - len(in_grants)))
            if len(in_denials) < width:
                in_denials.extend([0.0] * (width - len(in_denials)))

        in_service_ratios: List[Optional[float]] = []
        for req, grant in zip(in_reqs, in_grants):
            ratio = safe_div(grant, req)
            in_service_ratios.append(ratio)
            req_bins.append(req)
            grant_bins.append(grant)
            if req > 0 and ratio is not None:
                service_ratio_bins.append(ratio)

        denial_bins.extend(in_denials)
        total_requests += req_total
        total_grants += grant_total
        total_denials += denial_total

        per_router_out[str(rid)] = {
            "totalRequests": req_total,
            "totalGrants": grant_total,
            "totalDenials": denial_total,
            "grantRatio": safe_div(grant_total, req_total),
            "inportRequests": in_reqs,
            "inportGrants": in_grants,
            "inportDenials": in_denials,
            "inportServiceRatios": in_service_ratios,
        }

    return {
        "note": "Fairness is computed from SA-II arbitration request/grant/denial stats.",
        "sa2": {
            "available": True,
            "totals": {
                "requests": total_requests,
                "grants": total_grants,
                "denials": total_denials,
                "grantRatio": safe_div(total_grants, total_requests),
            },
            "distributions": {
                "requestPerInportStats": distribution_metrics(req_bins),
                "grantPerInportStats": distribution_metrics(grant_bins),
                "denialPerInportStats": distribution_metrics(denial_bins),
                "serviceRatioStats": distribution_metrics(service_ratio_bins),
                "requestJainIndex": jain_index(req_bins),
                "grantJainIndex": jain_index(grant_bins),
                "serviceRatioJainIndex": jain_index(service_ratio_bins),
            },
            "perRouter": per_router_out,
        },
    }


def build_mem_ctrl_summary(parsed: Dict[str, ParsedValue]) -> Dict[str, Any]:
    ctrl_data: Dict[int, Dict[str, float]] = defaultdict(dict)

    for key, val in parsed.items():
        m = MEM_CTRL_RE.match(key)
        if not m or val.scalar is None or math.isnan(val.scalar):
            continue
        ctrl_id = int(m.group(1))
        field = m.group(2)
        ctrl_data[ctrl_id][field] = val.scalar

    total_read_reqs = 0.0
    total_write_reqs = 0.0
    total_read_bursts = 0.0
    total_write_bursts = 0.0

    weighted_mem_acc_num = 0.0
    weighted_mem_acc_den = 0.0

    for fields in ctrl_data.values():
        total_read_reqs += fields.get("readReqs", 0.0)
        total_write_reqs += fields.get("writeReqs", 0.0)
        rb = fields.get("dram.readBursts", 0.0)
        wb = fields.get("dram.writeBursts", 0.0)
        total_read_bursts += rb
        total_write_bursts += wb
        avg_mem_acc = fields.get("dram.avgMemAccLat")
        if avg_mem_acc is not None and not math.isnan(avg_mem_acc):
            w = rb + wb
            weighted_mem_acc_num += avg_mem_acc * w
            weighted_mem_acc_den += w

    return {
        "numControllersSeen": len(ctrl_data),
        "totalReadReqs": total_read_reqs,
        "totalWriteReqs": total_write_reqs,
        "totalReadBursts": total_read_bursts,
        "totalWriteBursts": total_write_bursts,
        "weightedAvgMemAccLat": (
            weighted_mem_acc_num / weighted_mem_acc_den if weighted_mem_acc_den > 0 else None
        ),
    }


def build_output(parsed: Dict[str, ParsedValue], include_raw: bool) -> Dict[str, Any]:
    runtime = build_runtime_summary(parsed)
    garnet = build_garnet_summary(parsed, runtime)
    tail = build_tail_latency_summary(parsed)
    fairness = build_fairness_summary(parsed)
    mem_ctrl = build_mem_ctrl_summary(parsed)

    output: Dict[str, Any] = {
        "runtime": runtime,
        "garnet": garnet,
        "tailLatency": tail,
        "fairness": fairness,
        "memoryControllers": mem_ctrl,
    }

    if include_raw:
        output["raw"] = {
            k: {
                "raw": v.raw,
                "scalar": v.scalar,
                "vector": v.vector,
                "vectorRaw": v.vector_raw,
            }
            for k, v in parsed.items()
        }

    return output


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def main() -> None:
    args = parse_args()
    stats_path = Path(args.stats)
    parsed = parse_stats_file(stats_path)
    output = build_output(parsed, include_raw=args.include_raw)
    output = sanitize_for_json(output)

    json_kwargs: Dict[str, Any] = {"ensure_ascii": True}
    if args.pretty:
        json_kwargs["indent"] = 2
        json_kwargs["sort_keys"] = True

    text = json.dumps(output, **json_kwargs)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote metrics JSON to {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()

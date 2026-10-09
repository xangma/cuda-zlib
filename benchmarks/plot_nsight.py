#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Plot source-bound Nsight GPU activities; these are instrumented diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from plot_results import configure_style, report_identifier, save_figure

ROOT = Path(__file__).resolve().parent
WORKLOADS = ("zeros", "text", "uint32", "float32", "random")
LABELS = dict(zip(WORKLOADS, ("Zeros", "Synthetic text", "Ascending uint32", "Normal float32", "Random bytes")))
SIZES = (65536, 1048576, 67108864)
STAGES = {
    "fused_decode": ("Fused decode", "#4477AA", ""),
    "framing": ("Framing", "#94A3B8", "//"),
    "discovery": ("Discovery", "#0072B2", ""),
    "sorting": ("Sorting", "#56B4E9", "//"),
    "description": ("Token description", "#E69F00", ""),
    "chain_selection": ("Chain selection", "#F0D573", "//"),
    "emission": ("Emission", "#D55E00", ""),
    "refinement": ("Refinement", "#CC79A7", "//"),
    "checksum": ("Checksum", "#009E73", ""),
    "verification": ("Status verification", "#475569", "//"),
}
MEMORY_COLOR = "#64748B"
GAP_COLOR = "#E2E8F0"
TIMELINE_CASES = (("uint32", 131072), ("float32", 67108864), ("text", 65536))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label):
    require(type(value) is int and value >= 0, f"{label} must be a nonnegative integer")
    return value


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_hash(value, label):
    require(isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value), f"Invalid {label}")


def union_and_gaps(activities):
    """Return the union and its complement without attributing gaps to a cause."""
    merged = []
    for event in sorted(activities, key=lambda event: (event["start_ns"], event["end_ns"])):
        start, end = event["start_ns"], event["end_ns"]
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = [(left[1], right[0]) for left, right in zip(merged, merged[1:]) if left[1] < right[0]]
    return sum(end - start for start, end in merged), gaps


def load_report(path):
    report = json.loads(path.read_text())
    require(report.get("schema") == 1 and report.get("complete") is True, "Expected complete Nsight schema 1")
    revision = report.get("source_revision", "")
    require(len(revision) == 40 and all(char in "0123456789abcdef" for char in revision), "Missing full source revision")
    validate_hash(report["capture_manifest_sha256"], "capture manifest SHA256")
    validate_hash(report["harness_sha256"], "harness SHA256")
    require(isinstance(report["source_sha256"], dict) and report["source_sha256"], "Missing measured source hashes")
    for source, digest in report["source_sha256"].items():
        validate_hash(digest, f"{source} SHA256")
    cases = {}
    identity = None
    for case in report["cases"]:
        key = case["workload"], integer(case["output_bytes"], "output_bytes")
        require(key not in cases, f"Duplicate case {key}")
        integer(case["encoded_bytes"], f"{key}: encoded_bytes")
        require(set(case["stage_ns"]) == set(STAGES), f"{key}: stage taxonomy differs")
        activities = case["gpu_activities"]
        require(activities, f"{key}: no GPU activities")
        totals = dict.fromkeys(STAGES, 0)
        memory_ns = 0
        for event in activities:
            start = integer(event["start_ns"], f"{key}: start_ns")
            end = integer(event["end_ns"], f"{key}: end_ns")
            duration = integer(event["duration_ns"], f"{key}: duration_ns")
            require(end > start and end - start == duration, f"{key}: inconsistent activity interval")
            require(isinstance(event["name"], str) and event["name"], f"{key}: unnamed activity")
            if event["kind"] == "kernel":
                require(event["stage"] in STAGES, f"{key}: unknown kernel stage")
                totals[event["stage"]] += duration
            else:
                require(event["kind"] in ("memset", "memcpy") and event["stage"] is None, f"{key}: unknown activity kind")
                memory_ns += duration
        require(min(event["start_ns"] for event in activities) == 0, f"{key}: origin must be first GPU activity")
        for stage, duration in case["stage_ns"].items():
            require(integer(duration, f"{key}: {stage}") == totals[stage], f"{key}: stage duration mismatch")
        kernel_ns = integer(case["kernel_sum_ns"], f"{key}: kernel_sum_ns")
        require(kernel_ns > 0 and sum(totals.values()) == kernel_ns, f"{key}: kernel sum mismatch")
        span = max(event["end_ns"] for event in activities)
        union, _ = union_and_gaps(activities)
        require(integer(case["gpu_span_ns"], f"{key}: gpu_span_ns") == span, f"{key}: GPU span mismatch")
        require(integer(case["gap_ns"], f"{key}: gap_ns") == span - union, f"{key}: gap mismatch")
        require(integer(case["memory_operation_ns"], f"{key}: memory_operation_ns") == memory_ns, f"{key}: memory sum mismatch")
        provenance = case["provenance"]
        require(provenance["source_revision"] == revision, f"{key}: source revision mismatch")
        require(provenance["harness_sha256"] == report["harness_sha256"], f"{key}: harness mismatch")
        require(provenance["source_sha256"] == report["source_sha256"], f"{key}: measured source hashes mismatch")
        native = tuple(provenance[field] for field in ("cache_key", "library_sha256", "build_sha256"))
        for digest in native:
            validate_hash(digest, f"{key}: native identity SHA256")
        if identity is None:
            identity = native
        require(native == identity, f"{key}: mixed native identities")
        require(isinstance(case["warnings"], list), f"{key}: warnings must be retained as a list")
        cases[key] = case
    expected = {(workload, size) for workload in WORKLOADS for size in SIZES} | {("uint32", 131072)}
    require(set(cases) == expected, "Expected five workloads at 64 KiB, 1 MiB, 64 MiB plus uint32 at 128 KiB")
    return report, cases


def size_label(size):
    return f"{size // 1048576} MiB" if size >= 1048576 else f"{size // 1024} KiB"


def device_label(report):
    device = report.get("device", {})
    if isinstance(device, str):
        return device
    return device.get("name") or device.get("gpu_name") or "CUDA GPU"


def legend_handles():
    return [Patch(facecolor=color, edgecolor="#FFFFFF", hatch=hatch, label=label)
            for label, color, hatch in STAGES.values()]


def warning_note(cases):
    warnings = [warning for case in cases for warning in case["warnings"]]
    if "not all cuda events" in json.dumps(warnings).lower():
        return "Nsight reports potentially missing CUDA events; plots show recorded activities. Full warnings remain in the report and manifest."
    return "Nsight warnings are retained in the report and manifest." if warnings else ""


def stage_shares(report, cases):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.97, top=0.73, bottom=0.23, wspace=0.22)
    fig.suptitle("Decompression kernel time by stage", x=0.105, y=0.97, ha="left", fontsize=19, weight="bold")
    fig.text(0.105, 0.89, f"{device_label(report)} · one warmed, instrumented call per case · stdlib level-6 streams · source {report['source_revision'][:12]}", fontsize=10.5)
    fig.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(0.1, 0.85), ncol=5,
               fontsize=9.5, frameon=False, handlelength=1.7, columnspacing=1.5)
    for ax, size in zip(axes, SIZES):
        ax.set_title(size_label(size), pad=13)
        ax.set_xlim(0, 119)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("Share of summed GPU kernel time")
        ax.set_yticks(range(len(WORKLOADS)), [LABELS[workload] for workload in WORKLOADS])
        ax.set_ylim(4.65, -0.65)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.tick_params(axis="y", length=0)
        ax.text(108, -0.75, "Kernel\nms", ha="center", va="bottom", fontsize=9, color="#475569")
        for row, workload in enumerate(WORKLOADS):
            case = cases[workload, size]
            left = 0
            for stage, (_, color, hatch) in STAGES.items():
                share = 100 * case["stage_ns"][stage] / case["kernel_sum_ns"]
                if share:
                    ax.barh(row, share, left=left, height=0.58, color=color, hatch=hatch,
                            edgecolor="white", linewidth=0.5)
                    if share >= 14:
                        rgb = matplotlib.colors.to_rgb(color)
                        ink = "white" if sum(channel * weight for channel, weight in zip(rgb, (0.2126, 0.7152, 0.0722))) < 0.52 else "#172B4D"
                        ax.text(left + share / 2, row, f"{share:.1f}%", ha="center", va="center", color=ink, fontsize=9.5)
                left += share
            ax.text(108, row, f"{case['kernel_sum_ns'] / 1e6:.3f}", ha="center", va="center", fontsize=9.5)
    fig.text(0.105, 0.14, "Denominator: sum of recorded kernel durations; memory operations and gaps excluded. No wall-time or speedup comparison.", fontsize=9)
    fig.text(0.105, 0.108, "Fused decode is one indivisible kernel; internal parsing, emission and checks cannot be separated by this trace.", fontsize=9)
    fig.text(0.105, 0.076, "All observed launches count, including guarded launches; stage shares do not establish useful work.", fontsize=8.5)
    fig.text(0.105, 0.044, warning_note(cases.values()), fontsize=8.5)
    return fig


def activity_timeline(report, cases):
    selected = [cases[key] for key in TIMELINE_CASES]
    lane_counts = [sum(case["stage_ns"][stage] > 0 for stage in STAGES) + 2 for case in selected]
    fig, axes = plt.subplots(3, 1, figsize=(15.2, 10.8), gridspec_kw={"height_ratios": [max(5, count) for count in lane_counts]})
    fig.subplots_adjust(left=0.15, right=0.97, top=0.87, bottom=0.17, hspace=0.78)
    fig.suptitle("Recorded GPU activity during decompression", x=0.15, y=0.97, ha="left", fontsize=19, weight="bold")
    fig.text(0.15, 0.931, f"{device_label(report)} · one warmed, instrumented call per panel · independent time scales · source {report['source_revision'][:12]}", fontsize=10.5)
    for ax, case in zip(axes, selected):
        stages = [stage for stage in STAGES if case["stage_ns"][stage] > 0]
        lanes = stages + ["memory", "gap"]
        labels = [STAGES[stage][0] for stage in stages] + ["Memset / memcpy", "Uncovered time"]
        positions = {stage: row for row, stage in enumerate(lanes)}
        for event in case["gpu_activities"]:
            lane = event["stage"] if event["kind"] == "kernel" else "memory"
            color, hatch = STAGES[lane][1:] if lane in STAGES else (MEMORY_COLOR, "xx")
            ax.broken_barh([(event["start_ns"] / 1e6, event["duration_ns"] / 1e6)],
                           (positions[lane] - 0.32, 0.64), facecolors=color, edgecolors="#172B4D", linewidth=0.35, hatch=hatch)
        _, gaps = union_and_gaps(case["gpu_activities"])
        if gaps:
            ax.broken_barh([(start / 1e6, (end - start) / 1e6) for start, end in gaps],
                           (positions["gap"] - 0.32, 0.64), facecolors=GAP_COLOR, edgecolors="#94A3B8", linewidth=0.35, hatch="//")
        for lane, description in (("memory", "No memory operations recorded"), ("gap", "No uncovered time")):
            if lane == "memory" and case["memory_operation_ns"] or lane == "gap" and case["gap_ns"]:
                continue
            ax.text(0.01, positions[lane], description, transform=ax.get_yaxis_transform(), fontsize=8.5, va="center", color="#64748B")
        span_ms = case["gpu_span_ns"] / 1e6
        ax.set_xlim(0, span_ms * 1.015)
        ax.set_ylim(len(lanes) - 0.35, -0.65)
        ax.set_yticks(range(len(lanes)), labels)
        ax.set_xlabel("Time since first recorded GPU activity (ms)")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        kernels = sum(event["kind"] == "kernel" for event in case["gpu_activities"])
        ax.set_title(f"{LABELS[case['workload']]} · {size_label(case['output_bytes'])} · {kernels} kernel {'launch' if kernels == 1 else 'launches'}", loc="left", pad=23)
        ax.text(0, 1.025, f"GPU span {span_ms:.3f} ms   |   kernel sum {case['kernel_sum_ns'] / 1e6:.3f} ms   |   memory sum {case['memory_operation_ns'] / 1e6:.3f} ms   |   uncovered {case['gap_ns'] / 1e6:.3f} ms",
                transform=ax.transAxes, fontsize=9, color="#475569")
    fig.text(0.15, 0.092, "Every imported kernel, memset and memcpy interval is shown. Memory operations share a lane; any overlapping intervals remain overlaid.", fontsize=9)
    fig.text(0.15, 0.067, "Uncovered time is the complement of the union of GPU activity intervals; it does not identify a CPU or synchronization cause.", fontsize=9)
    fig.text(0.15, 0.042, "Fused kernels remain indivisible. Trace timings are instrumented diagnostics, not end-to-end benchmarks.", fontsize=8.5)
    fig.text(0.15, 0.017, warning_note(selected), fontsize=8.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/nsight/rtx4090-decode.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/nsight")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report, cases = load_report(args.input)
    if args.validate_only:
        print(f"Validated {len(cases)} source-bound Nsight cases")
        return
    configure_style()
    plt.rcParams["svg.hashsalt"] = "cuda-zlib-nsight-figures-v1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    for name, renderer in (("stage-shares", stage_shares), ("activity-timeline", activity_timeline)):
        figures[name] = {"exports": save_figure(renderer(report, cases), args.output_dir, name)}
    figures["stage-shares"]["cases"] = [{
        "name": case["name"], "workload": workload, "output_bytes": size,
        "kernel_sum_ns": case["kernel_sum_ns"],
        "stage_percent": {stage: 100 * duration / case["kernel_sum_ns"] for stage, duration in case["stage_ns"].items()},
    } for size in SIZES for workload in WORKLOADS for case in [cases[workload, size]]]
    figures["stage-shares"]["denominator"] = "sum of recorded kernel durations; memory operations and gaps excluded"
    figures["activity-timeline"]["cases"] = [{
        "name": cases[key]["name"], "workload": key[0], "output_bytes": key[1],
        "gpu_span_ns": cases[key]["gpu_span_ns"], "gap_ns": cases[key]["gap_ns"],
        "activity_count": len(cases[key]["gpu_activities"]),
    } for key in TIMELINE_CASES]
    manifest = {
        "schema": 1, "kind": "nsight-diagnostics", "source_report": report_identifier(args.input),
        "source_report_sha256": sha256(args.input),
        "measurement_source": {"revision": report["source_revision"], "harness_sha256": report["harness_sha256"], "sha256": report["source_sha256"]},
        "capture_manifest_sha256": report["capture_manifest_sha256"],
        "plotter_sha256": sha256(Path(__file__)), "style_helper_sha256": sha256(ROOT / "plot_results.py"),
        "renderer": {"matplotlib": matplotlib.__version__, "backend": matplotlib.get_backend()},
        "units": {"source_time": "ns", "display_time": "ms", "stage_share": "% of kernel sum"},
        "time_origin": "first imported GPU activity, separately for each case",
        "gap_definition": "GPU span minus union of all imported GPU activity intervals",
        "stage_styles": {stage: {"label": label, "color": color, "hatch": hatch} for stage, (label, color, hatch) in STAGES.items()},
        "warnings": {case["name"]: case["warnings"] for case in cases.values() if case["warnings"]},
        "native_identity": {field: next(iter(cases.values()))["provenance"][field] for field in ("cache_key", "library_sha256", "build_sha256")},
        "figures": figures,
        "exports": {filename: digest for figure in figures.values() for filename, digest in figure["exports"].items()},
        "limitations": ["One warmed instrumented call per case; no uncertainty or speedup estimate", "Fused internal stages cannot be separated", "Guarded launches counted without inferring useful work", "Gaps do not identify a cause; interval sums are not wall time"],
    }
    (args.output_dir / "rtx4090-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Rendered {len(figures)} Nsight figure families, 6 exports and manifest")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Plot source-bound compression and decompression workflow diagnostics."""

import argparse
import json
import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from plot_nsight import STAGES
from plot_results import configure_style, report_identifier, save_figure
from plot_timeline import (
    COPY_STYLES, GIB, MIB, SECOND, bins_for_view, concrete_warning, finite,
    integer, overlapping, require, sampled_line, sampling_summary, sha256,
    union_length,
)

ROOT = Path(__file__).resolve().parent
OPERATIONS = ("compress", "decompress")
OPERATION_LABELS = {"compress": "Compression", "decompress": "Decompression"}
COMPRESSION_STYLES = {
    "fused_encode": ("Fused encode", "#9966AA", ""),
    "encoding": ("Chunk encoding", "#8E44AD", ""),
    "compression_prefix": ("Compression prefix", "#E69F00", "//"),
    "packing": ("Packing", "#0072B2", ""),
    "wrapper": ("Wrapper", "#94A3B8", "//"),
    "checksum": ("Checksum", "#009E73", ""),
    "verification": ("Status verification", "#475569", "//"),
}
GPU_STYLES = {"compress": COMPRESSION_STYLES, "decompress": dict(STAGES)}


def load_report(path, operation):
    report = json.loads(path.read_text())
    require(report.get("schema_version") == 1 and report.get("complete") is True, "Expected complete workflow schema 1")
    require(report["operation"] == operation, "Report operation differs from requested operation")
    revision = report["source_revision"]
    require(len(revision) == 40 and all(char in "0123456789abcdef" for char in revision), "Missing full source revision")
    require(report["time_origin"] is not None, "A synchronized Nsight clock origin is required")
    finite(report["time_origin"]["uncertainty_ns"], "clock uncertainty")
    whole, loop = report["views"]["whole_process"], report["views"]["measured_loop"]
    require(whole["start_ns"] < whole["end_ns"], "Invalid whole-process bounds")
    require(whole["start_ns"] <= loop["start_ns"] < loop["end_ns"] <= whole["end_ns"], "Measured loop outside process")
    phases = report["phase_intervals"]
    ids = {phase["id"] for phase in phases}
    require(phases and len(ids) == len(phases), "Missing or duplicate host phases")
    for phase in phases:
        require(phase["end_ns"] >= phase["start_ns"], "Inverted host interval")
        require(phase["parent_id"] is None or phase["parent_id"] in ids, "Unknown host parent")
    expected_leaves = {"metadata_download", "status_check", "output_download", "host_bytes", "validate"}
    require(expected_leaves <= {phase["name"] for phase in phases}, "Detailed download/check phases missing")
    kernels = []
    copies = dict.fromkeys(COPY_STYLES, 0)
    require(report["gpu_activities"], "No recorded CUDA activities")
    for event in report["gpu_activities"]:
        start, end = integer(event["start_ns"], "GPU start"), integer(event["end_ns"], "GPU end")
        require(end > start and end - start == event["duration_ns"], "Invalid GPU interval")
        if event["kind"] == "kernel":
            require(event["stage"] in GPU_STYLES[operation], f"Unknown {operation} kernel stage: {event['stage']}")
            kernels.append((start, end))
        else:
            require(event["kind"] in ("memcpy", "memset"), "Unknown GPU activity kind")
            if event["kind"] == "memcpy":
                require(event["direction"] in copies, "Unknown copy direction")
                copies[event["direction"]] += finite(event["bytes"], "copy bytes")
    binned = dict.fromkeys(COPY_STYLES, 0)
    previous_end = None
    require(report["bins"], "Missing CUDA bins")
    for bucket in report["bins"]:
        start, end = bucket["start_ns"], bucket["end_ns"]
        require(end > start and (previous_end is None or start == previous_end), "Noncontiguous CUDA bins")
        previous_end = end
        covered = union_length((max(start, a), min(end, b)) for a, b in kernels if a < end and b > start)
        require(bucket["kernel_union_ns"] == covered, "Kernel union differs from bin")
        require(math.isclose(bucket["kernel_union_fraction"], covered / (end - start), abs_tol=1e-12, rel_tol=1e-12), "Kernel fraction differs from bin")
        require(set(bucket["copy_bytes_completed"]) == set(COPY_STYLES), "Missing copy bin direction")
        for direction, count in bucket["copy_bytes_completed"].items():
            binned[direction] += finite(count, "binned bytes")
    require(binned == copies, "Completed copy bin totals differ from recorded copies")
    samples = report["samples"]
    require(len(samples) >= 2, "At least two resource samples required")
    previous = None
    for sample in samples:
        timestamp = integer(sample["timestamp_ns"], "sample time")
        require(previous is None or timestamp > previous, "Unordered resource samples")
        previous = timestamp
        require(sample["query_start_ns"] <= timestamp <= sample["query_end_ns"], "Sample outside query bracket")
        clocks = sample["metric_timestamps_ns"]
        values = {"cpu_percent": sample["cpu_percent"], "rss_bytes": sample["rss_bytes"]}
        gpu = sample.get("gpu")
        for key, field in (("gpu_memory", "used_memory_bytes"), ("gpu_owned_memory", "owned_compute_memory_bytes")):
            require(gpu is None or field in gpu, f"Missing {field}; unavailable values must be explicit nulls")
            values[key] = None if gpu is None else gpu[field]
        for key, value in values.items():
            finite(value, key, nullable=True)
            require(key in clocks and (clocks[key] is None or type(clocks[key]) is int), f"Missing or invalid {key} clock")
            require(value is None or clocks[key] is not None, f"Valid {key} has no metric clock")
    require(isinstance(report["warnings"], list), "Warnings must be retained")
    return report


def host_panel(ax, phases, semantics, start, end):
    by_id = {phase["id"]: phase for phase in phases}

    def depth(phase):
        seen, level = {phase["id"]}, 0
        while phase["parent_id"] is not None:
            require(phase["parent_id"] not in seen, "Cyclic host phase hierarchy")
            seen.add(phase["parent_id"])
            phase = by_id[phase["parent_id"]]
            level += 1
        return level

    visible = overlapping(phases, start, end)
    names = {phase["name"] for phase in visible}
    levels = {name: min(depth(phase) for phase in phases if phase["name"] == name) for name in names}
    names = sorted(names, key=lambda name: (levels[name], min(phase["start_ns"] for phase in phases if phase["name"] == name)))
    positions = {name: row for row, name in enumerate(names)}
    parents = {phase["parent_id"] for phase in phases}
    labels = []
    for name in names:
        metadata = semantics.get(name, {})
        kind = metadata.get("kind") if isinstance(metadata, dict) else metadata
        label = name.replace("_", " ")
        if kind in ("enqueue", "completed") and kind not in label:
            label += f" [{kind}]"
        prefix = "↳ " if levels[name] else ""
        labels.append(prefix + "\n".join(textwrap.wrap(label, width=29)))
    for phase in visible:
        metadata = semantics.get(phase["name"], {})
        kind = metadata.get("kind") if isinstance(metadata, dict) else metadata
        enqueue = kind == "enqueue"
        parent = phase["id"] in parents
        ax.broken_barh([(phase["start_ns"] / SECOND, (phase["end_ns"] - phase["start_ns"]) / SECOND)],
                       (positions[phase["name"]] - 0.33, 0.66), facecolors="#CBD5E1" if parent or enqueue else "#64748B",
                       edgecolors="#475569", linewidth=0.4, hatch="//" if enqueue else None)
    ax.set_yticks(range(len(names)), labels, fontsize=8.5)
    ax.set_ylim(max(0.6, len(names) - 0.35), -0.65)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Host annotations · containers and nested download/check phases", loc="left", fontsize=11, pad=9)


def cuda_panel(ax, report, start, end):
    styles = GPU_STYLES[report["operation"]]
    events = overlapping(report["gpu_activities"], start, end)
    lanes = [stage for stage in styles if any(event["kind"] == "kernel" and event["stage"] == stage for event in events)]
    lanes += [direction for direction in COPY_STYLES if any(event["kind"] == "memcpy" and event["direction"] == direction for event in events)]
    if any(event["kind"] == "memset" for event in events):
        lanes.append("memset")
    positions = {lane: row for row, lane in enumerate(lanes)}
    for event in events:
        lane = event["stage"] if event["kind"] == "kernel" else event["direction"] if event["kind"] == "memcpy" else "memset"
        _, color, hatch = styles[lane] if lane in styles else COPY_STYLES[lane] if lane in COPY_STYLES else ("Memset", "#64748B", "xx")
        ax.broken_barh([(event["start_ns"] / SECOND, event["duration_ns"] / SECOND)],
                       (positions[lane] - 0.32, 0.64), facecolors=color, edgecolors="#172B4D", linewidth=0.25, hatch=hatch)
    ax.set_yticks(range(len(lanes)), [styles[lane][0] if lane in styles else f"Copy {lane}" if lane in COPY_STYLES else "Memset" for lane in lanes], fontsize=8.5)
    ax.set_ylim(max(0.6, len(lanes) - 0.35), -0.65)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Exact recorded CUDA intervals · {OPERATION_LABELS[report['operation']].lower()} stages", loc="left", fontsize=11, pad=9)


def render(report, view):
    whole, loop = report["views"]["whole_process"], report["views"]["measured_loop"]
    bounds = whole if view == "whole_process" else {"start_ns": max(whole["start_ns"], loop["start_ns"] - 40_000_000), "end_ns": min(whole["end_ns"], loop["end_ns"] + 40_000_000)}
    start, end = bounds["start_ns"], bounds["end_ns"]
    fig, axes = plt.subplots(7, 1, figsize=(16.2, 20), sharex=True,
                             gridspec_kw={"height_ratios": [4.3, 3.0, 1.2, 1.0, 1.3, 1.1, 1.1]})
    fig.subplots_adjust(left=0.17, right=0.97, top=0.90, bottom=0.13, hspace=0.57)
    title = "whole process" if view == "whole_process" else "warmed loop"
    fig.suptitle(f"{OPERATION_LABELS[report['operation']]} workflow · {title}", x=0.17, y=0.978, ha="left", fontsize=20, weight="bold")
    workload = report["workload"]
    payload_bytes = workload["output_bytes"]  # Raw payload size for both operations, distinct from encoded upload size.
    fig.text(0.17, 0.944, f"{report['device']['name']} · {payload_bytes / MIB:g} MiB {workload['kind']} payload · {workload['iterations']} validated measured iterations · source {report['source_revision'][:12]}", fontsize=11)
    fig.text(0.17, 0.926, f"Instrumented diagnostics · {workload['warmups']} warmups · exact CUDA intervals and sampled process resources share elapsed time", fontsize=10)
    host_panel(axes[0], report["phase_intervals"], report.get("phase_semantics", {}), start, end)
    cuda_panel(axes[1], report, start, end)
    bins = bins_for_view(report, start, end)
    left = np.array([bucket["start_ns"] / SECOND for bucket in bins])
    widths = np.array([(bucket["end_ns"] - bucket["start_ns"]) / SECOND for bucket in bins])
    bottom = np.zeros(len(bins))
    for direction, (label, color, hatch) in COPY_STYLES.items():
        values = np.array([bucket["copy_bytes_completed"][direction] / MIB for bucket in bins])
        if np.any(values):
            axes[2].bar(left, values, width=widths, bottom=bottom, align="edge", label=label, color=color, hatch=hatch, linewidth=0)
            bottom += values
    axes[2].set_title("Copy bytes assigned at completion · stacked by transfer direction", loc="left", fontsize=11, pad=9)
    axes[2].set_ylabel("MiB per bin")
    axes[2].legend(loc="lower right", bbox_to_anchor=(1, 1.03), borderaxespad=0, ncol=4, frameon=False, fontsize=8)
    fractions = np.array([bucket["kernel_union_fraction"] * 100 for bucket in bins])
    edges = np.array([bucket["start_ns"] / SECOND for bucket in bins] + [bins[-1]["end_ns"] / SECOND])
    axes[3].stairs(fractions, edges, fill=True, color="#0072B2", alpha=0.75, linewidth=0.6)
    axes[3].set_title("Recorded kernel activity fraction · interval union / bin duration; not occupancy", loc="left", fontsize=11, pad=9)
    axes[3].set_ylabel("Activity")
    axes[3].set_ylim(0, 105)
    axes[3].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    samples = report["samples"]
    sampled_line(axes[4], samples, lambda sample: (sample.get("gpu") or {}).get("owned_compute_memory_bytes"), GIB,
                 "Observed process tree", "#0072B2", time_key="gpu_owned_memory", start=start, end=end)
    sampled_line(axes[4], samples, lambda sample: (sample.get("gpu") or {}).get("used_memory_bytes"), GIB,
                 "Whole GPU (device total)", "#64748B", time_key="gpu_memory", start=start, end=end, linestyle="--")
    axes[4].set_title("GPU VRAM · process allocations and whole-device usage are separate scopes", loc="left", fontsize=11, pad=9)
    axes[4].set_ylabel("GiB")
    axes[4].legend(loc="lower right", bbox_to_anchor=(1, 1.03), borderaxespad=0, ncol=2, frameon=False, fontsize=8)
    sampled_line(axes[5], samples, lambda sample: sample["cpu_percent"], 1, "Observed process tree", "#0072B2", time_key="cpu_percent", start=start, end=end)
    axes[5].set_title("Process-tree CPU · cumulative CPU-time differences at actual sample intervals", loc="left", fontsize=11, pad=9)
    axes[5].set_ylabel("% · 100 = 1 core")
    sampled_line(axes[6], samples, lambda sample: sample["rss_bytes"], GIB, "Summed process-tree RSS", "#D55E00", time_key="rss_bytes", start=start, end=end)
    axes[6].set_title("Process-tree resident memory · summed RSS can count shared pages more than once", loc="left", fontsize=11, pad=9)
    axes[6].set_ylabel("GiB RSS")
    axes[6].set_xlabel("Elapsed time since collector clock origin (s)")
    for index, ax in enumerate(axes):
        ax.set_xlim(start / SECOND, end / SECOND)
        ax.grid(axis="x", alpha=0.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        if index >= 2:
            ax.set_ylim(bottom=0)
            if index != 3:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if view == "whole_process":
            ax.axvspan(loop["start_ns"] / SECOND, loop["end_ns"] / SECOND, facecolor="#94A3B8", alpha=0.10, zorder=-1)
        for boundary in (loop["start_ns"], loop["end_ns"]):
            ax.axvline(boundary / SECOND, color="#475569", linestyle="--", linewidth=0.7, alpha=0.8)
    nominal_ms = max(bucket["end_ns"] - bucket["start_ns"] for bucket in report["bins"]) / 1e6
    sampling = sampling_summary(samples)
    fig.text(0.17, 0.095, f"Dashed lines bound the measured loop. CUDA bins: {nominal_ms:g} ms; view-boundary bins use their actual clipped width. Copies show completed bytes, not bandwidth.", fontsize=9)
    fig.text(0.17, 0.077, f"Collector cadence: median {sampling['median_ms']:.3f} ms ({sampling['minimum_ms']:.3f}–{sampling['maximum_ms']:.3f} ms). Metric points use query midpoints; CPU uses averaging-interval midpoints.", fontsize=9)
    fig.text(0.17, 0.059, f"Host/CUDA alignment: ±{report['time_origin']['uncertainty_ns'] / 1000:.3f} µs (half the origin-mark bracket). Host ranges can nest; they do not establish exclusive GPU ownership or costs.", fontsize=9)
    fig.text(0.17, 0.041, "Download, status validation, host byte construction and output validation are distinct host phases. Coarse resources cannot resolve individual short phases; nulls remain gaps.", fontsize=9)
    fig.text(0.17, 0.023, concrete_warning(report), fontsize=9)
    return fig, bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for operation in OPERATIONS:
        parser.add_argument(f"--{operation}", type=Path, help=f"Normalized {operation} workflow report")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/workflow")
    parser.add_argument("--prefix", default="rtx4090", help="hardware identifier for the manifest filename")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        parser.error("prefix must be a filename component")
    if not any(getattr(args, operation) is not None for operation in OPERATIONS):
        parser.error("provide at least one of --compress or --decompress")
    reports = {operation: load_report(getattr(args, operation), operation) for operation in OPERATIONS if getattr(args, operation) is not None}
    baseline = next(iter(reports.values()))
    for report in reports.values():
        require(report["source_revision"] == baseline["source_revision"] and report["source_sha256"] == baseline["source_sha256"], "Operations have different measured source identities")
    if args.validate_only:
        print(f"Validated {len(reports)} source-bound operation workflow(s) and their bins/resources")
        return
    configure_style()
    plt.rcParams["svg.hashsalt"] = "cuda-zlib-operation-workflows-v1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures, exports = {}, {}
    for operation, report in reports.items():
        for suffix, view in (("whole-process", "whole_process"), ("warmed-loop", "measured_loop")):
            name = f"{operation}-{suffix}"
            figure, bounds = render(report, view)
            rendered = save_figure(figure, args.output_dir, name)
            exports.update(rendered)
            bins = bins_for_view(report, bounds["start_ns"], bounds["end_ns"])
            figures[name] = {"operation": operation, "view": view, "display_bounds_ns": bounds, "exports": rendered,
                             "kernel_union_ns": sum(bucket["kernel_union_ns"] for bucket in bins),
                             "copy_bytes_completed": {direction: sum(bucket["copy_bytes_completed"][direction] for bucket in bins) for direction in COPY_STYLES}}
    helpers = {f"benchmarks/{name}": sha256(ROOT / name) for name in ("plot_results.py", "plot_nsight.py", "plot_timeline.py")}
    manifest = {
        "schema": 1, "kind": "workflow", "operations": list(reports), "plotter_sha256": sha256(Path(__file__)),
        "source_reports": {operation: {"path": report_identifier(getattr(args, operation)), "sha256": sha256(getattr(args, operation))} for operation in reports},
        "measurement_sources": {operation: {"revision": report["source_revision"], "harness_sha256": report["harness_sha256"], "sha256": report["source_sha256"]} for operation, report in reports.items()},
        "report_evidence": {operation: {field: report[field] for field in ("native_build", "extractor_sha256", "artifact_sha256", "dependencies_sha256", "extractor_dependencies_sha256")} for operation, report in reports.items()},
        "style_helpers_sha256": helpers, "exports": exports, "figures": figures,
        "warnings": {operation: report["warnings"] for operation, report in reports.items()},
        "time_origins": {operation: report["time_origin"] for operation, report in reports.items()},
        "resource_sample_intervals": {operation: sampling_summary(report["samples"]) for operation, report in reports.items()},
        "phase_semantics": {operation: report.get("phase_semantics", {}) for operation, report in reports.items()},
        "stage_styles": {operation: {stage: {"label": label, "color": color, "hatch": hatch} for stage, (label, color, hatch) in GPU_STYLES[operation].items()} for operation in reports},
        "renderer": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "backend": matplotlib.get_backend()},
        "units": {"display_time": "seconds since collector clock origin", "copy": "MiB completed per bin", "activity": "kernel interval union/actual bin duration", "memory": "GiB", "cpu": "percent; 100 is one logical CPU"},
        "limitations": ["Instrumented diagnostics, not benchmarks", "Kernel activity is not occupancy", "Completed bytes per bin are not instantaneous bandwidth", "Host phases do not establish exclusive costs or GPU ownership", "Resource points use metric-specific query clocks; CPU uses averaging-interval midpoints", "Null resources remain gaps", "Summed RSS may double-count shared pages"],
    }
    (args.output_dir / f"{args.prefix}-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Rendered {len(figures)} workflow families, {len(exports)} exports and provenance manifest")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Render synchronized host, recorded CUDA and sampled resource timelines."""

import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parent
SECOND, MIB, GIB = 1e9, 2 ** 20, 2 ** 30
COPY_STYLES = {
    "H2D": ("Host → device", "#0072B2", ""),
    "D2H": ("Device → host", "#D55E00", "//"),
    "D2D": ("Device → device", "#009E73", "xx"),
    "other": ("Other copies", "#64748B", ".."),
}
GPU_STYLES = {**STAGES, "runtime_auxiliary": ("Runtime auxiliary", "#6B7280", "xx")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, *, nullable=False):
    if value is None and nullable:
        return value
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"Invalid {label}")
    return value


def integer(value, label):
    require(type(value) is int, f"{label} must be integer nanoseconds")
    return value


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_length(intervals):
    end, total = None, 0
    for start, stop in sorted(intervals):
        if end is None or start >= end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = max(stop, end) if end is not None else stop
    return total


def load_report(path):
    report = json.loads(path.read_text())
    require(report.get("schema_version") == 1 and report.get("complete") is True, "Expected complete timeline schema 1")
    revision = report["source_revision"]
    require(len(revision) == 40 and all(char in "0123456789abcdef" for char in revision), "Missing full measured source pin")
    for view in ("whole_process", "measured_loop"):
        bounds = report["views"][view]
        require(integer(bounds["end_ns"], view) > integer(bounds["start_ns"], view), f"Invalid {view} bounds")
    whole, loop = report["views"]["whole_process"], report["views"]["measured_loop"]
    require(whole["start_ns"] <= loop["start_ns"] < loop["end_ns"] <= whole["end_ns"], "Measured loop outside whole process")
    finite(report["time_origin"]["uncertainty_ns"], "clock-anchor uncertainty")
    require(report["phase_intervals"], "No host phases")
    ids = {phase["id"] for phase in report["phase_intervals"]}
    require(len(ids) == len(report["phase_intervals"]), "Duplicate host phase IDs")
    for phase in report["phase_intervals"]:
        require(phase["end_ns"] >= phase["start_ns"], "Inverted host phase")
        require(phase["parent_id"] is None or phase["parent_id"] in ids, "Unknown host phase parent")
    kernels = []
    copied = dict.fromkeys(COPY_STYLES, 0)
    require(report["gpu_activities"], "No recorded GPU activities")
    for event in report["gpu_activities"]:
        start, end = integer(event["start_ns"], "GPU start"), integer(event["end_ns"], "GPU end")
        require(end > start and end - start == event["duration_ns"], "Inconsistent GPU interval")
        if event["kind"] == "kernel":
            require(event["stage"] in GPU_STYLES, f"Unrecognized GPU stage: {event['stage']}")
            kernels.append((start, end))
        else:
            require(event["kind"] in ("memcpy", "memset"), "Unrecognized GPU activity kind")
            if event["kind"] == "memcpy":
                require(event["direction"] in copied, "Unrecognized copy direction")
                copied[event["direction"]] += finite(event["bytes"], "copy bytes")
    binned_bytes = dict.fromkeys(COPY_STYLES, 0)
    previous_end = None
    require(report["bins"], "No recorded activity bins")
    for bucket in report["bins"]:
        start, end = bucket["start_ns"], bucket["end_ns"]
        require(end > start and (previous_end is None or start == previous_end), "Noncontiguous activity bins")
        previous_end = end
        expected_union = union_length((max(start, left), min(end, right)) for left, right in kernels if left < end and right > start)
        require(bucket["kernel_union_ns"] == expected_union, "Kernel union/bin mismatch")
        require(math.isclose(bucket["kernel_union_fraction"], expected_union / (end - start), rel_tol=1e-12, abs_tol=1e-12), "Kernel fraction/bin mismatch")
        require(set(bucket["copy_bytes_completed"]) == set(COPY_STYLES), "Missing copy bin directions")
        for direction, count in bucket["copy_bytes_completed"].items():
            binned_bytes[direction] += finite(count, "binned copy bytes")
    require(binned_bytes == copied, "Completed-copy bin total differs from recorded copy bytes")
    previous_sample = None
    require(report["samples"], "No resource samples")
    for sample in report["samples"]:
        timestamp = integer(sample["timestamp_ns"], "sample timestamp")
        require(previous_sample is None or timestamp > previous_sample, "Resource samples not strictly ordered")
        previous_sample = timestamp
        require(sample["query_start_ns"] <= sample["timestamp_ns"] <= sample["query_end_ns"], "Sample outside query bracket")
        finite(sample["cpu_percent"], "CPU percent", nullable=True)
        finite(sample["rss_bytes"], "RSS bytes", nullable=True)
        clocks = sample["metric_timestamps_ns"]
        for field in ("cpu_percent", "rss_bytes", "gpu_memory", "gpu_owned_memory"):
            require(field in clocks, f"Missing metric timestamp: {field}")
            require(clocks[field] is None or type(clocks[field]) is int, f"Invalid metric timestamp: {field}")
        for field in ("cpu_percent", "rss_bytes"):
            require(sample[field] is None or clocks[field] is not None, f"Valid {field} has no metric timestamp")
        gpu = sample.get("gpu")
        if gpu is not None:
            for field in ("used_memory_bytes", "owned_compute_memory_bytes"):
                require(field in gpu, f"GPU sample is missing {field}; unavailable values must be explicit nulls")
                finite(gpu[field], field, nullable=True)
                clock_key = "gpu_memory" if field == "used_memory_bytes" else "gpu_owned_memory"
                require(gpu[field] is None or clocks[clock_key] is not None, f"Valid {field} has no metric timestamp")
    require(isinstance(report["warnings"], list), "Warnings must be preserved")
    return report


def overlapping(events, start, end):
    return [event for event in events if event["start_ns"] < end and event["end_ns"] > start]


def phase_label(name, semantics):
    name = name.replace("_", " ").replace("h2d", "H2D").replace("d2h", "D2H")
    kind = semantics.get("kind") if isinstance(semantics, dict) else semantics
    if kind in ("enqueue", "completed") and kind not in name.lower():
        name += f" [{kind}]"
    return "\n".join(textwrap.wrap(name, width=28))


def host_panel(ax, phases, semantics, start, end):
    visible = overlapping(phases, start, end)
    names = list(dict.fromkeys(phase["name"] for phase in sorted(visible, key=lambda phase: (phase["start_ns"], -phase["end_ns"]))))
    positions = {name: row for row, name in enumerate(names)}
    parents = {phase["parent_id"] for phase in phases}
    for phase in visible:
        label = semantics.get(phase["name"], "")
        kind = label.get("kind") if isinstance(label, dict) else label
        enqueue = kind == "enqueue" or phase["name"].lower().endswith("_enqueue")
        parent = phase["id"] in parents
        ax.broken_barh([(phase["start_ns"] / SECOND, (phase["end_ns"] - phase["start_ns"]) / SECOND)],
                       (positions[phase["name"]] - 0.33, 0.66), facecolors="#CBD5E1" if parent or enqueue else "#64748B",
                       edgecolors="#475569", linewidth=0.4, hatch="//" if enqueue else None)
    ax.set_yticks(range(len(names)), [phase_label(name, semantics.get(name, {})) for name in names], fontsize=8)
    ax.set_ylim(max(0.6, len(names) - 0.35), -0.65)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Host annotations · nested phases and repeated calls", loc="left", fontsize=11, pad=9)


def cuda_panel(ax, activities, start, end):
    visible = overlapping(activities, start, end)
    stages = [stage for stage in GPU_STYLES if any(event["kind"] == "kernel" and event["stage"] == stage for event in visible)]
    kinds = [direction for direction in COPY_STYLES if any(event["kind"] == "memcpy" and event["direction"] == direction for event in visible)]
    lanes = stages + kinds
    if any(event["kind"] == "memset" for event in visible):
        lanes.append("memset")
    positions = {stage: row for row, stage in enumerate(lanes)}
    for event in visible:
        lane = event["stage"] if event["kind"] == "kernel" else event["direction"] if event["kind"] == "memcpy" else "memset"
        _, color, hatch = GPU_STYLES[lane] if lane in GPU_STYLES else COPY_STYLES[lane] if lane in COPY_STYLES else ("Memset", "#64748B", "xx")
        ax.broken_barh([(event["start_ns"] / SECOND, event["duration_ns"] / SECOND)],
                       (positions[lane] - 0.32, 0.64), facecolors=color, edgecolors="#172B4D", linewidth=0.25, hatch=hatch)
    labels = [GPU_STYLES[lane][0] if lane in GPU_STYLES else f"Copy {lane}" if lane in COPY_STYLES else "Memset" for lane in lanes]
    ax.set_yticks(range(len(lanes)), labels, fontsize=8)
    ax.set_ylim(max(0.6, len(lanes) - 0.35), -0.65)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Exact recorded CUDA intervals · stage colors match the Nsight figures", loc="left", fontsize=11, pad=9)


def sampled_line(ax, samples, getter, scale, label, color, *, time_key, start, end, linestyle="-"):
    times, values = [], []
    for sample in samples:
        value = getter(sample)
        timestamp = sample["metric_timestamps_ns"][time_key]
        if timestamp is None:
            timestamp = sample["timestamp_ns"]  # Position a missing-value gap; no reading is imputed.
        if not start <= timestamp <= end:
            continue
        times.append(timestamp / SECOND)
        values.append(np.nan if value is None else value / scale)
    ax.plot(times, values, color=color, linestyle=linestyle, linewidth=1.1, marker=".", markersize=2.0, label=label)


def concrete_warning(report):
    warnings = json.dumps(report["warnings"]).lower()
    if "not all nvtx" in warnings:
        return "Nsight warns potentially missing NVTX events; all expected host phase ranges were validated. Other warnings remain in the report and manifest."
    if "not all cuda events" in warnings:
        return "Nsight reports potentially missing CUDA events; CUDA panels show recorded activities only."
    return "Trace/collector warnings are retained in the report and manifest." if report["warnings"] else ""


def sampling_summary(samples):
    intervals_ms = np.diff([sample["timestamp_ns"] for sample in samples]) / 1e6
    require(len(intervals_ms) > 0, "At least two resource samples are required")
    return {"median_ms": float(np.median(intervals_ms)), "minimum_ms": float(np.min(intervals_ms)), "maximum_ms": float(np.max(intervals_ms))}


def bins_for_view(report, start, end):
    """Clip view-boundary bins using exact intervals and copy-completion times."""
    bins = []
    kernels = [(event["start_ns"], event["end_ns"]) for event in report["gpu_activities"] if event["kind"] == "kernel"]
    copies = [event for event in report["gpu_activities"] if event["kind"] == "memcpy"]
    for bucket in report["bins"]:
        left, right = max(start, bucket["start_ns"]), min(end, bucket["end_ns"])
        if left >= right:
            continue
        if left == bucket["start_ns"] and right == bucket["end_ns"]:
            bins.append(bucket)
            continue
        covered = union_length((max(left, a), min(right, b)) for a, b in kernels if a < right and b > left)
        completed = dict.fromkeys(COPY_STYLES, 0)
        for event in copies:
            if left <= event["end_ns"] < right or event["end_ns"] == right == end:
                completed[event["direction"]] += event["bytes"]
        bins.append({"start_ns": left, "end_ns": right, "kernel_union_ns": covered,
                     "kernel_union_fraction": covered / (right - left), "copy_bytes_completed": completed})
    require(bins, "View contains no CUDA bins")
    return bins


def render(report, view):
    whole, loop = report["views"]["whole_process"], report["views"]["measured_loop"]
    bounds = whole if view == "whole_process" else {"start_ns": max(whole["start_ns"], loop["start_ns"] - 40_000_000), "end_ns": min(whole["end_ns"], loop["end_ns"] + 40_000_000)}
    start, end = bounds["start_ns"], bounds["end_ns"]
    fig, axes = plt.subplots(7, 1, figsize=(16.2, 17.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.7, 3.0, 1.2, 1.0, 1.3, 1.1, 1.1]})
    fig.subplots_adjust(left=0.16, right=0.97, top=0.89, bottom=0.145, hspace=0.57)
    title = "Whole-process CUDA timeline" if view == "whole_process" else "Warmed-loop CUDA timeline"
    fig.suptitle(title, x=0.16, y=0.978, ha="left", fontsize=20, weight="bold")
    workload = report["workload"]
    device = report["device"]["name"]
    fig.text(0.16, 0.941, f"{device} · {workload['output_bytes'] / MIB:g} MiB {workload['kind']} · {workload['iterations']} validated measured iterations · source {report['source_revision'][:12]}", fontsize=11)
    fig.text(0.16, 0.922, f"Instrumented diagnostics · {workload['warmups']} warmups · synchronized panels share elapsed time; resource samples have lower temporal resolution", fontsize=10)
    host_panel(axes[0], report["phase_intervals"], report.get("phase_semantics", {}), start, end)
    cuda_panel(axes[1], report["gpu_activities"], start, end)
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
    axes[2].legend(loc="lower right", bbox_to_anchor=(1, 1.03), borderaxespad=0,
                   ncol=4, frameon=False, fontsize=8)
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
    axes[4].legend(loc="lower right", bbox_to_anchor=(1, 1.03), borderaxespad=0,
                   ncol=2, frameon=False, fontsize=8)
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
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) if index != 3 else None
        if view == "whole_process":
            ax.axvspan(loop["start_ns"] / SECOND, loop["end_ns"] / SECOND, facecolor="#94A3B8", alpha=0.10, zorder=-1)
        for boundary in (loop["start_ns"], loop["end_ns"]):
            ax.axvline(boundary / SECOND, color="#475569", linestyle="--", linewidth=0.7, alpha=0.8)
    bin_widths = sorted({bucket["end_ns"] - bucket["start_ns"] for bucket in report["bins"]})
    nominal = max(bin_widths) / 1e6
    uncertainty_us = report["time_origin"]["uncertainty_ns"] / 1000
    sampling = sampling_summary(report["samples"])
    fig.text(0.16, 0.105, f"Dashed lines bound the measured loop; whole-process shading locates it. CUDA bins: {nominal:g} ms; view-boundary bins use their actual clipped width.", fontsize=9)
    fig.text(0.16, 0.085, f"Collector cadence: median {sampling['median_ms']:.3f} ms ({sampling['minimum_ms']:.3f}–{sampling['maximum_ms']:.3f} ms). Metric points use query midpoints; CPU uses averaging-interval midpoints.", fontsize=9)
    fig.text(0.16, 0.065, f"Host/CUDA alignment: ±{uncertainty_us:.3f} µs (half the origin-mark bracket). Host annotations do not establish GPU ownership or exclusive costs.", fontsize=9)
    fig.text(0.16, 0.045, "Transfers are completed bytes per bin, not instantaneous bandwidth. Coarse resource samples cannot resolve individual short phases; nulls remain gaps.", fontsize=9)
    fig.text(0.16, 0.025, concrete_warning(report), fontsize=9)
    return fig, bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/timeline/rtx4090-float32.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/timeline")
    parser.add_argument("--prefix", default="rtx4090", help="hardware identifier for the manifest filename")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        parser.error("prefix must be a filename component")
    report = load_report(args.input)
    if args.validate_only:
        print(f"Validated timeline: {len(report['gpu_activities'])} CUDA activities, {len(report['samples'])} resource samples")
        return
    configure_style()
    plt.rcParams["svg.hashsalt"] = "cuda-zlib-rich-timeline-v1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures, exports = {}, {}
    for name, view in (("whole-process", "whole_process"), ("warmed-loop", "measured_loop")):
        figure, bounds = render(report, view)
        rendered = save_figure(figure, args.output_dir, name)
        exports.update(rendered)
        bins = bins_for_view(report, bounds["start_ns"], bounds["end_ns"])
        figures[name] = {"exports": rendered, "view": view, "display_bounds_ns": bounds,
                         "kernel_union_ns": sum(bucket["kernel_union_ns"] for bucket in bins),
                         "copy_bytes_completed": {direction: sum(bucket["copy_bytes_completed"][direction] for bucket in bins) for direction in COPY_STYLES}}
    helpers = {f"benchmarks/{name}": sha256(ROOT / name) for name in ("plot_results.py", "plot_nsight.py")}
    manifest = {
        "schema": 1, "kind": "timeline", "source_report": report_identifier(args.input),
        "source_report_sha256": sha256(args.input), "plotter_sha256": sha256(Path(__file__)),
        "style_helper_sha256": helpers["benchmarks/plot_results.py"], "style_helpers_sha256": helpers, "exports": exports, "figures": figures,
        "measurement_source": {"revision": report["source_revision"], "harness_sha256": report["harness_sha256"], "sha256": report["source_sha256"]},
        "native_build": report["native_build"], "extractor_sha256": report["extractor_sha256"],
        "artifact_sha256": report["artifact_sha256"], "time_origin": report["time_origin"],
        "gpu_activity_count": len(report["gpu_activities"]), "host_phase_count": len(report["phase_intervals"]),
        "sample_count": len(report["samples"]), "bin_count": len(report["bins"]),
        "resource_sample_intervals": sampling_summary(report["samples"]),
        "resource_timestamp_policy": {"cpu_percent": "midpoint of the averaging interval between process-query midpoints", "rss_bytes": "process-query span midpoint", "gpu_memory": "NVML memory-query midpoint", "gpu_owned_memory": "NVML compute-process-query midpoint"},
        "warnings": report["warnings"], "phase_semantics": report.get("phase_semantics", {}),
        "renderer": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "backend": matplotlib.get_backend()},
        "units": {"display_time": "seconds since collector clock origin", "copy": "MiB completed per bin", "kernel_activity": "union(kernel intervals)/actual bin duration", "memory": "GiB", "cpu": "percent; 100 means one logical CPU"},
        "limitations": ["Instrumented diagnostics, not end-to-end benchmarks", "Kernel activity is not occupancy", "Host annotations do not establish GPU ownership", "Fused internal stages remain indivisible", "Missing resource values remain gaps", "Summed RSS may count shared pages more than once"],
    }
    (args.output_dir / f"{args.prefix}-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("Rendered two synchronized timelines, six exports and provenance manifest")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Render the measured RTX 3090 benchmark as PNG, SVG, and PDF figures."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
WORKLOADS = ("zeros", "text", "uint32", "float32", "random")
SIZES = (65536, 1048576, 67108864)
LABELS = {
    "zeros": "Zeros", "text": "Synthetic text",
    "uint32": "Ascending uint32", "float32": "Normal float32",
    "random": "Random bytes",
}
BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
COMPRESSION = (
    ("CUDA resident", "cuda_compress_device_device", BLUE, "o", "-"),
    ("CUDA host → host", "cuda_compress_host_host", ORANGE, "D", "--"),
    ("CPU zlib level 1", "cpu_compress_level1", GREEN, "^", "-."),
    ("CPU zlib level 6", "cpu_compress_level6", PURPLE, "s", ":"),
)
DECOMPRESSION = (
    ("CUDA resident", "cuda_decompress_level6_device_device", BLUE, "o", "-"),
    ("CUDA host → bytes", "cuda_decompress_level6_host_host", ORANGE, "D", "--"),
    ("CUDA host → array", "cuda_decompress_level6_host_array", PURPLE, "s", ":"),
    ("CPU zlib → bytes", "cpu_decompress_level6", GREEN, "^", "-."),
)
STREAM_LAYOUT = (
    ("CUDA · codec stream", "cuda_decompress_codec_device_device", BLUE, "o", "-"),
    ("CUDA · level-6 stream", "cuda_decompress_level6_device_device", ORANGE, "D", "--"),
    ("CPU · codec stream", "cpu_decompress_codec", GREEN, "^", "-."),
    ("CPU · level-6 stream", "cpu_decompress_level6", PURPLE, "s", ":"),
)
ENCODED_SIZE = (
    ("CUDA codec", "cuda", BLUE, "o"),
    ("CPU zlib level 1", "zlib1", GREEN, "^"),
    ("CPU zlib level 6", "zlib6", PURPLE, "s"),
)
CPU_SPEEDUP = (
    ("Host-to-host compression", "cpu_compress_level1", "cuda_compress_host_host", BLUE),
    ("Host-to-host decompression", "cpu_decompress_level6", "cuda_decompress_level6_host_host", ORANGE),
)
ALL_KEYS = {series[1] for series in COMPRESSION + DECOMPRESSION + STREAM_LAYOUT}
ALL_KEYS.add("cuda_compress_host_device")


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}: expected a positive finite number")
    return value


def equal(actual, expected, label):
    positive(actual, label)
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{label}: stored value disagrees with its source samples")


def load_results(path):
    """Reject partial runs and inconsistent timing summaries before rendering."""
    source = path.read_bytes()
    report = json.loads(source)
    args = report["arguments"]
    if report["schema_version"] != 1 or args["cpu_only"]:
        raise ValueError("Expected a schema-1 CUDA benchmark report")
    if tuple(args["sizes"]) != SIZES or tuple(args["workloads"]) != WORKLOADS:
        raise ValueError("Expected all five workloads at 64 KiB, 1 MiB, and 64 MiB")
    for field in ("samples", "cpu_samples"):
        if not isinstance(args[field], int) or isinstance(args[field], bool) or args[field] < 1:
            raise ValueError(f"{field}: expected a positive integer")
    expected = {(workload, size) for workload in WORKLOADS for size in SIZES}
    cases = {}
    for case in report["cases"]:
        identity = (case["workload"], case["input_bytes"])
        if identity not in expected or identity in cases:
            raise ValueError(f"Unexpected or duplicate case: {identity}")
        if not case["validation"].startswith("all outputs byte-exact"):
            raise ValueError(f"Missing byte-exact validation: {identity}")
        if set(case["timings"]) != ALL_KEYS:
            raise ValueError(f"Incomplete or unknown timing keys: {identity}")
        for key, timing in case["timings"].items():
            samples = timing["seconds"]
            count = args["cpu_samples"] if key.startswith("cpu_") else args["samples"]
            if len(samples) != count:
                raise ValueError(f"Wrong sample count: {identity} {key}")
            for sample in samples:
                positive(sample, f"{identity} {key} sample")
            median = statistics.median(samples)
            equal(timing["median_seconds"], median, f"{identity} {key} median")
            equal(timing["min_seconds"], min(samples), f"{identity} {key} minimum")
            equal(timing["max_seconds"], max(samples), f"{identity} {key} maximum")
            equal(timing["mib_per_second"], identity[1] / 2**20 / median,
                  f"{identity} {key} throughput")
        for key in ("cuda", "zlib1", "zlib6"):
            encoded = positive(case["encoded_bytes"][key], f"{identity} {key} encoded bytes")
            if not isinstance(encoded, int):
                raise ValueError(f"Noninteger encoded length: {identity} {key}")
            equal(case["encoded_percent"][key], 100 * encoded / identity[1],
                  f"{identity} {key} encoded percentage")
        cases[identity] = case
    if set(cases) != expected:
        raise ValueError("Expected 15 complete benchmark cases")
    return report, cases, hashlib.sha256(source).hexdigest()


def throughput(case, key):
    """Return stored median throughput and sample extrema in throughput units."""
    timing = case["timings"][key]
    median = timing["mib_per_second"]
    rates = case["input_bytes"] / 2**20 / np.asarray(timing["seconds"])
    return median, median - float(rates.min()), float(rates.max()) - median


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#64748B",
        "axes.linewidth": 0.8, "text.color": "#172B4D",
        "axes.labelcolor": "#172B4D", "xtick.color": "#334155",
        "ytick.color": "#334155", "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "grid.color": "#CBD5E1", "grid.linewidth": 0.6,
        "grid.alpha": 0.7, "svg.fonttype": "none",
        "svg.hashsalt": "cuda-zlib-benchmark-figures-v1",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def header(fig, title, subtitle):
    fig.text(0.065, 0.956, title, fontsize=19, weight="bold", va="top")
    fig.text(0.065, 0.904, subtitle, fontsize=11, va="top", color="#475569")


def footer(fig, report, additional, timing=True, method=None):
    environment, args = report["environment"], report["arguments"]
    gpu = environment["gpu"].removeprefix("NVIDIA GeForce ")
    cpu = environment["cpu"].removeprefix("AMD Ryzen ").removesuffix(" 64-Cores")
    fig.text(0.065, 0.085, f"{gpu} + {cpu} · same workstation · {report['created_utc'][:10]}",
             fontsize=9, color="#475569")
    if method is None:
        if timing:
            method = (f"Warm medians; {args['samples']} CUDA / {args['cpu_samples']} CPU samples; "
                      "error bars = sample min–max, not confidence intervals. Cold startup excluded.")
        else:
            method = "Encoded percentages include all stream framing; lower is smaller. 100% means no size reduction. Labels rounded to 0.001%."
    fig.text(0.065, 0.059, method, fontsize=9, color="#475569")
    fig.text(0.065, 0.033, additional, fontsize=9, color="#475569")


def log_ticks(ax, axis="y"):
    locator = FixedLocator([10**power for power in range(-3, 7)])
    formatter = FuncFormatter(lambda x, _: f"{x:g}")
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(locator)
    target.set_major_formatter(formatter)
    target.set_minor_locator(NullLocator())


def throughput_panels(report, cases, series, title, subtitle):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.84, bottom=0.195,
                        wspace=0.23, hspace=0.41)
    header(fig, title, subtitle)
    bounds = [throughput(cases[(w, size)], s[1]) for w in WORKLOADS
              for size in SIZES for s in series]
    minimum = min(median - low for median, low, high in bounds)
    maximum = max(median + high for median, low, high in bounds)
    limits = (10**math.floor(math.log10(minimum / 1.3)), maximum * 1.45)
    for ax, workload in zip(axes.flat, WORKLOADS):
        ax.set_title(LABELS[workload], loc="left", pad=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(limits)
        ax.set_xlim(SIZES[0] / 2**20 / 1.5, SIZES[-1] / 2**20 * 1.5)
        ax.set_xticks(np.asarray(SIZES) / 2**20, ["64 KiB", "1 MiB", "64 MiB"])
        ax.xaxis.set_minor_locator(NullLocator())
        log_ticks(ax)
        ax.grid(axis="y", which="major")
        for label, key, color, marker, linestyle in series:
            values = np.asarray([throughput(cases[(workload, size)], key) for size in SIZES])
            ax.errorbar(np.asarray(SIZES) / 2**20, values[:, 0],
                        yerr=values[:, 1:].T, label=label, color=color,
                        marker=marker, linestyle=linestyle, linewidth=1.8,
                        markersize=5.5, capsize=3, elinewidth=1)
        ax.set_xlabel("Uncompressed input size")
    axes[0, 0].set_ylabel("Throughput (MiB/s)")
    axes[1, 0].set_ylabel("Throughput (MiB/s)")
    # The sixth panel carries the shared legend without obscuring measured data.
    legend_ax = axes[1, 2]
    legend_ax.set_axis_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, frameon=False, loc="upper left",
                     bbox_to_anchor=(0, 1.05), borderaxespad=0, labelspacing=1.1)
    host_note = ("Host outputs: transfers included\nBytes: Python bytes; array: NumPy"
                 if series == DECOMPRESSION else "Host → host: transfers included")
    legend_ax.text(0, 0.18, f"Resident: device → device\n{host_note}\nCPU: single thread\nLines connect measured sizes only.",
                   transform=legend_ax.transAxes, va="top", fontsize=9.5, color="#475569",
                   linespacing=1.4)
    footer(fig, report, "API allocations and codec validation included; payload generation, initial resident uploads, and post-timing oracle comparisons excluded.")
    return fig


def encoded_size(report, cases):
    fig, ax = plt.subplots(figsize=(11.5, 6.7))
    fig.subplots_adjust(left=0.17, right=0.94, top=0.81, bottom=0.205)
    header(fig, "Compressed size at 64 MiB", "Complete encoded stream size as a percentage of the original input")
    ax.set_xscale("log")
    values = [cases[(w, SIZES[-1])]["encoded_percent"][s[1]]
              for w in WORKLOADS for s in ENCODED_SIZE]
    ax.set_xlim(min(values) / 1.5, max(100, max(values)) * 2.8)
    ax.set_ylim(4.5, -0.6)
    ax.set_yticks(range(5), [LABELS[w] for w in WORKLOADS])
    ax.tick_params(axis="y", length=0, pad=10)
    ax.spines["left"].set_visible(False)
    log_ticks(ax, "x")
    ax.grid(axis="x", zorder=0)
    offsets = (-0.24, 0, 0.24)
    for index, workload in enumerate(WORKLOADS):
        if index % 2 == 0:
            ax.axhspan(index - 0.45, index + 0.45, color="#F1F5F9", zorder=0)
        for (label, key, color, marker), offset in zip(ENCODED_SIZE, offsets):
            value = cases[(workload, SIZES[-1])]["encoded_percent"][key]
            ax.plot(value, index + offset, marker=marker, color=color,
                    markersize=6.5, linestyle="none", zorder=3)
            ax.annotate(f"{value:.3f}%", (value, index + offset),
                        xytext=(8, 0), textcoords="offset points", fontsize=9.5,
                        va="center", color=color)
    ax.axvline(100, color="#64748B", linestyle="--", linewidth=1, zorder=1)
    ax.annotate("100%", (100, -0.6), xytext=(0, 7), textcoords="offset points",
                fontsize=9, color="#64748B", ha="center", va="bottom")
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", label=label)
               for label, key, color, marker in ENCODED_SIZE]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.167, 0.865),
               frameon=False, ncols=3, columnspacing=2.2)
    ax.set_xlabel("Encoded size / input size (%) · logarithmic scale", labelpad=9)
    footer(fig, report, "CUDA uses independent 32 KiB chunks; stdlib zlib levels 1 and 6 are comparison settings, with no level equivalence implied.", timing=False)
    return fig


def decode_stream_layout(report, cases):
    fig, ax = plt.subplots(figsize=(11.5, 6.7))
    fig.subplots_adjust(left=0.17, right=0.94, top=0.78, bottom=0.205)
    header(fig, "Decompression depends on stream layout", "64 MiB inputs · resident CUDA and CPU decoding of the same two encoded streams")
    ax.set_xscale("log")
    bounds = [throughput(cases[(w, SIZES[-1])], s[1]) for w in WORKLOADS for s in STREAM_LAYOUT]
    ax.set_xlim(min(m - low for m, low, high in bounds) / 1.6,
                max(m + high for m, low, high in bounds) * 2)
    ax.set_ylim(4.5, -0.5)
    ax.set_yticks(range(5), [LABELS[w] for w in WORKLOADS])
    ax.tick_params(axis="y", length=0, pad=10)
    ax.spines["left"].set_visible(False)
    log_ticks(ax, "x")
    ax.grid(axis="x", zorder=0)
    offsets = (-0.3, -0.1, 0.1, 0.3)
    for index, workload in enumerate(WORKLOADS):
        if index % 2 == 0:
            ax.axhspan(index - 0.46, index + 0.46, color="#F1F5F9", zorder=0)
        for (label, key, color, marker, linestyle), offset in zip(STREAM_LAYOUT, offsets):
            median, low, high = throughput(cases[(workload, SIZES[-1])], key)
            ax.errorbar(median, index + offset, xerr=[[low], [high]],
                        marker=marker, color=color, markersize=6, linestyle="none",
                        capsize=3, elinewidth=1, zorder=3)
            ax.annotate(f"{median:,.0f}", (median + high, index + offset),
                        xytext=(8, 0), textcoords="offset points", fontsize=9.5,
                        va="center", color=color)
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", label=label)
               for label, key, color, marker, linestyle in STREAM_LAYOUT]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.167, 0.868),
               frameon=False, ncols=2, columnspacing=3.2, labelspacing=0.8)
    ax.set_xlabel("Throughput (MiB/s) · logarithmic scale", labelpad=9)
    footer(fig, report, "Codec stream: CUDA encoder output (independent 32 KiB chunks). Level-6 stream: single stdlib zlib stream. Transfers excluded.")
    return fig


def cpu_speedup(report, cases):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.78, bottom=0.22, wspace=0.14)
    header(fig, "GPU speedup over CPU", "64 MiB inputs · CUDA host-to-host workflows return bytes and include copies and transfers")
    baselines = ("CPU: single-thread stdlib zlib level 1", "CPU: same stdlib level-6 bytes, single thread")
    ratios = [[cases[(w, SIZES[-1])]["timings"][cpu]["median_seconds"] /
               cases[(w, SIZES[-1])]["timings"][cuda]["median_seconds"]
               for w in WORKLOADS] for _, cpu, cuda, _ in CPU_SPEEDUP]
    limit = max(1, max(value for values in ratios for value in values)) * 1.19
    for ax, (title, cpu, cuda, color), baseline, values in zip(axes, CPU_SPEEDUP, baselines, ratios):
        ax.set_title(title, loc="left", pad=29)
        ax.text(0, 1.045, baseline, transform=ax.transAxes, fontsize=9, color="#475569")
        ax.barh(range(len(WORKLOADS)), values, height=0.56, color=color, alpha=0.9, zorder=3)
        ax.set_xlim(0, limit)
        ax.set_ylim(len(WORKLOADS) - 0.5, -0.6)
        ax.set_yticks(range(len(WORKLOADS)), [LABELS[w] for w in WORKLOADS])
        ax.tick_params(axis="y", length=0, pad=10)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", zorder=0)
        ax.axvline(1, color="#64748B", linestyle="--", linewidth=1, zorder=4)
        ax.annotate("1×", (1, -0.6), xytext=(4, -4), textcoords="offset points",
                    fontsize=9, color="#64748B", ha="left", va="top")
        for index, value in enumerate(values):
            ax.annotate(f"{value:.2f}×", (value, index), xytext=(7, 0),
                        textcoords="offset points", fontsize=10, va="center", color=color)
        ax.set_xlabel("Speedup over CPU (×) · linear scale", labelpad=10)
    method = (f"Ratio of warm CPU / CUDA median time; {report['arguments']['samples']} CUDA / "
              f"{report['arguments']['cpu_samples']} CPU samples. Values above 1× favor CUDA. No error bars; cold startup excluded.")
    footer(fig, report, "Compression compares each encoder's output with no level equivalence implied. Decompression uses identical level-6 streams.",
           method=method)
    return fig


def save_figure(fig, output, name):
    """Suppress timestamps and fix SVG IDs so repeated exports are reproducible."""
    metadata = {
        "png": {"Software": "cuda-zlib benchmark plotting"},
        "svg": {"Date": None, "Creator": "cuda-zlib benchmark plotting"},
        "pdf": {"CreationDate": None, "ModDate": None, "Creator": "cuda-zlib benchmark plotting"},
    }
    exports = {}
    for extension in ("png", "svg", "pdf"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=180, metadata=metadata[extension])
        if extension == "svg":
            # Matplotlib emits trailing spaces in multiline path attributes.
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        exports[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    return exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "rtx3090-ffi-20261006-2.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    report, cases, source_hash = load_results(args.input)
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("compression-throughput", COMPRESSION, list(SIZES), "throughput_mib_per_second",
         lambda: throughput_panels(report, cases, COMPRESSION, "Compression throughput",
                                   "Warm medians across five synthetic workloads · common logarithmic axes")),
        ("encoded-size", ENCODED_SIZE, [SIZES[-1]], "encoded_percent",
         lambda: encoded_size(report, cases)),
        ("decompression-throughput", DECOMPRESSION, list(SIZES), "throughput_mib_per_second",
         lambda: throughput_panels(report, cases, DECOMPRESSION, "Decompression of stdlib level-6 streams",
                                   "Each backend decodes identical compressed bytes · common logarithmic axes")),
        ("decode-stream-layout", STREAM_LAYOUT, [SIZES[-1]], "throughput_mib_per_second",
         lambda: decode_stream_layout(report, cases)),
    )
    figures = {}
    throughput_statistic = "uncompressed MiB / median elapsed seconds, checked against recorded timing samples"
    throughput_errors = "sample throughput minimum and maximum from inverse individual elapsed times; not confidence intervals"
    for name, series, sizes, metric, render in specifications:
        print(f"Rendering {name}", flush=True)
        figures[name] = {
            "exports_sha256": save_figure(render(), args.output_dir, name),
            "workloads": list(WORKLOADS), "input_bytes": sizes, "metric": metric,
            "series": [{"label": s[0], "json_key": s[1]} for s in series],
            "point_statistic": throughput_statistic if metric == "throughput_mib_per_second" else "100 * complete encoded bytes / uncompressed input bytes",
            "error_bars": throughput_errors if metric == "throughput_mib_per_second" else None,
        }
    print("Rendering cpu-speedup", flush=True)
    figures["cpu-speedup"] = {
        "exports_sha256": save_figure(cpu_speedup(report, cases), args.output_dir, "cpu-speedup"),
        "workloads": list(WORKLOADS), "input_bytes": [SIZES[-1]], "metric": "cpu_speedup_ratio",
        "point_statistic": "CPU median elapsed seconds / CUDA median elapsed seconds; greater than 1 favors CUDA",
        "axis_scale": "linear", "reference_ratio": 1, "error_bars": None,
        "cuda_timing_scope": "Host-to-host with Python bytes outputs; copies and host/device transfers included",
        "cpu_timing_scope": "Same-host, single-thread stdlib zlib",
        "series": [{"label": label, "numerator_timing_key": cpu, "denominator_timing_key": cuda,
                    "ratio_formula": f"timings.{cpu}.median_seconds / timings.{cuda}.median_seconds"}
                   for label, cpu, cuda, color in CPU_SPEEDUP],
    }
    try:
        source_name = args.input.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source_name = args.input.name
    manifest = {
        "schema_version": 1, "source": source_name, "source_sha256": source_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "renderer": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "backend": "Agg"},
        "source_created_utc": report["created_utc"], "validated_cases": len(cases),
        "validated_timing_keys": sorted(ALL_KEYS),
        "samples_per_measurement": {"cuda": report["arguments"]["samples"], "cpu": report["arguments"]["cpu_samples"]},
        "throughput_units": "MiB/s; uncompressed bytes / 2**20 / seconds",
        "point_statistic": "Defined separately for each figure's metric",
        "error_bars": "Defined per figure; encoded-size and cpu-speedup have no error bars",
        "size_units": "100 * complete encoded bytes / uncompressed input bytes",
        "timing_scope": report["methodology"], "cold_startup_excluded": True,
        "figures": figures,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(figures)} figures in PNG, SVG, and PDF; validated {len(cases)} cases")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Validate comparable codec profiles and plot warm API throughput before/after."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

from plot_results import LABELS, WORKLOADS, configure_style, equal, positive, save_figure

ROOT = Path(__file__).resolve().parent
SIZE = 64 * 1024**2
OPERATIONS = ("compress", "decode_frozen", "decode_zlib6")
SOURCE_FILES = {"__init__.py", "_codec.py", "_decode_kernels.py",
                "_encode_kernels.py", "_errors.py", "jax.py"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COLORS = ("#0072B2", "#D55E00")


def digest(value, label):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{label}: expected a SHA256 digest")


def integer(value, label):
    positive(value, label)
    if not isinstance(value, int):
        raise ValueError(f"{label}: expected a positive integer")


def read_profile(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    if report["schema_version"] != 1:
        raise ValueError(f"{path.name}: expected a schema-1 profile")
    args = report["arguments"]
    integer(args["samples"], "samples")
    if not isinstance(args["device"], int) or isinstance(args["device"], bool) or args["device"] < 0:
        raise ValueError("device: expected a nonnegative integer")
    sizes, workloads = args["sizes"], args["workloads"]
    for size in sizes:
        integer(size, "input size")
    if len(set(sizes)) != len(sizes) or len(set(workloads)) != len(workloads):
        raise ValueError("Duplicate requested size or workload")
    if not workloads or any(w not in WORKLOADS for w in workloads):
        raise ValueError("Unknown or missing workload")
    if SIZE not in sizes:
        raise ValueError("The comparison chart requires 64 MiB cases")
    source = report["source_sha256"]
    if set(source) != SOURCE_FILES:
        raise ValueError("Expected source hashes for all six codec modules")
    for name, value in source.items():
        digest(value, f"source {name}")
    environment = report["environment"]
    for key in ("python", "cupy", "numpy", "zlib_runtime"):
        if not isinstance(environment[key], str) or not environment[key]:
            raise ValueError(f"Missing environment field: {key}")
    rows = environment["gpu_before"]
    if args["device"] >= len(rows):
        raise ValueError("Selected device missing from GPU snapshot")
    fields = [field.strip() for field in rows[args["device"]].split(",")]
    if len(fields) != 6 or not all(fields[:2]):
        raise ValueError("Expected GPU name and driver in the selected snapshot row")
    after_rows = environment["gpu_after"]
    if args["device"] >= len(after_rows):
        raise ValueError("Selected device missing from final GPU snapshot")
    after_fields = [field.strip() for field in after_rows[args["device"]].split(",")]
    if len(after_fields) != 6 or after_fields[:2] != fields[:2]:
        raise ValueError("Selected GPU name or driver changed during profiling")
    cases = {}
    expected = {(w, size) for w in workloads for size in sizes}
    for case in report["cases"]:
        identity = (case["workload"], case["input_bytes"])
        if identity not in expected or identity in cases:
            raise ValueError(f"Unexpected or duplicate case: {identity}")
        if case["byte_exact"] is not True or case["cpu_codec_forbidden"] is not True:
            raise ValueError(f"Missing correctness or CPU-codec guard: {identity}")
        integer(case["encoded_bytes"], f"{identity} encoded bytes")
        for key in ("input_sha256", "encoded_sha256", "frozen_stream_sha256", "stdlib_stream_sha256"):
            digest(case[key], f"{identity} {key}")
        if set(case["timings"]) != set(OPERATIONS):
            raise ValueError(f"Incomplete or unknown operations: {identity}")
        for operation in OPERATIONS:
            wall = case["timings"][operation]["wall"]
            samples = wall["seconds"]
            if len(samples) != args["samples"]:
                raise ValueError(f"Wrong sample count: {identity} {operation}")
            for value in samples:
                positive(value, f"{identity} {operation} sample")
            median = statistics.median(samples)
            for key, value in (("median_seconds", median), ("min_seconds", min(samples)),
                               ("max_seconds", max(samples)),
                               ("mib_per_second", identity[1] / 2**20 / median)):
                equal(wall[key], value, f"{identity} {operation} {key}")
        cases[identity] = case
    if set(cases) != expected:
        raise ValueError("Partial profile: missing requested cases")
    # Utilisation, temperature, and allocated memory naturally differ between runs.
    selected_gpu = {"name": fields[0], "driver": fields[1], "ordinal": args["device"]}
    return report, cases, selected_gpu, hashlib.sha256(raw).hexdigest()


def rates(case, operation):
    wall = case["timings"][operation]["wall"]
    values = case["input_bytes"] / 2**20 / np.asarray(wall["seconds"])
    return {"median": wall["mib_per_second"], "minimum": float(values.min()),
            "maximum": float(values.max())}


def compare(before, after):
    b_report, b_cases, b_gpu, b_digest = before
    a_report, a_cases, a_gpu, a_digest = after
    for key in ("sizes", "workloads", "samples", "seed", "device"):
        if b_report["arguments"][key] != a_report["arguments"][key]:
            raise ValueError(f"Profiles use different {key}")
    if b_report["methodology"] != a_report["methodology"]:
        raise ValueError("Profiles use different timing methodology")
    environment_keys = set(b_report["environment"]) - {"gpu_before", "gpu_after"}
    if environment_keys != set(a_report["environment"]) - {"gpu_before", "gpu_after"}:
        raise ValueError("Profiles record different environment fields")
    for key in environment_keys:
        if b_report["environment"][key] != a_report["environment"][key]:
            raise ValueError(f"Profiles use different environment {key}")
    if b_gpu != a_gpu:
        raise ValueError("Profiles use different selected GPU name, driver, or ordinal")
    comparisons = []
    for identity, b_case in b_cases.items():
        a_case = a_cases[identity]
        for key in ("input_sha256", "frozen_stream_sha256", "stdlib_stream_sha256"):
            if b_case[key] != a_case[key]:
                raise ValueError(f"Different {key}: {identity}")
        if b_case["frozen_stream_sha256"] != b_case["encoded_sha256"]:
            raise ValueError(f"Baseline frozen stream is not its own encoded output: {identity}")
        timings = {}
        for operation in OPERATIONS:
            b_wall = b_case["timings"][operation]["wall"]
            a_wall = a_case["timings"][operation]["wall"]
            timings[operation] = {
                "before_seconds": b_wall["median_seconds"], "after_seconds": a_wall["median_seconds"],
                "speedup": b_wall["median_seconds"] / a_wall["median_seconds"],
                "before_mib_per_second": rates(b_case, operation),
                "after_mib_per_second": rates(a_case, operation),
            }
        comparisons.append({
            "workload": identity[0], "input_bytes": identity[1],
            "input_sha256": b_case["input_sha256"],
            "frozen_stream_sha256": b_case["frozen_stream_sha256"],
            "stdlib_stream_sha256": b_case["stdlib_stream_sha256"],
            "encoded_bytes": {"before": b_case["encoded_bytes"], "after": a_case["encoded_bytes"]},
            "encoded_sha256": {"before": b_case["encoded_sha256"], "after": a_case["encoded_sha256"]},
            "encoded_percent": {"before": 100 * b_case["encoded_bytes"] / identity[1],
                                "after": 100 * a_case["encoded_bytes"] / identity[1]},
            "encoded_size_after_over_before": a_case["encoded_bytes"] / b_case["encoded_bytes"],
            "compression_ratio": {"before": identity[1] / b_case["encoded_bytes"],
                                  "after": identity[1] / a_case["encoded_bytes"]},
            "timings": timings,
        })
    comparison = {
        "schema_version": 1, "profile_sha256": {"before": b_digest, "after": a_digest},
        "source_sha256": {"before": b_report["source_sha256"], "after": a_report["source_sha256"]},
        "changed_source_files": sorted(k for k in SOURCE_FILES
                                       if b_report["source_sha256"][k] != a_report["source_sha256"][k]),
        "environment": {k: b_report["environment"][k] for k in sorted(environment_keys)},
        "gpu": b_gpu, "samples_per_measurement": b_report["arguments"]["samples"],
        "seed": b_report["arguments"]["seed"], "validated_cases": len(comparisons),
        "methodology": b_report["methodology"],
        "point_statistic": "uncompressed MiB / median synchronized API wall seconds",
        "error_bars": "individual sample throughput minimum and maximum; not confidence intervals",
        "speedup": "before median seconds / after median seconds; greater than 1 is faster",
        "kernel_event_profiles_plotted": False,
        "comparison_limitations": "Recorded environment and selected GPU name/driver/ordinal match; "
                                  "profiles do not establish exclusive workstation use.",
        "cases": comparisons,
    }
    if "aggregation" in b_report or "aggregation" in a_report:
        if "aggregation" not in b_report or "aggregation" not in a_report:
            raise ValueError("Both profiles must carry aggregation provenance")
        b_aggregation, a_aggregation = b_report["aggregation"], a_report["aggregation"]
        for key in ("run_order", "runs", "script_sha256"):
            if b_aggregation[key] != a_aggregation[key]:
                raise ValueError(f"Aggregates have different {key} provenance")
        comparison["aggregation"] = {"before": b_aggregation, "after": a_aggregation}
    return comparison


def render(comparison):
    configure_style()
    selected = [case for case in comparison["cases"] if case["input_bytes"] == SIZE]
    workloads = [w for w in WORKLOADS if any(c["workload"] == w for c in selected)]
    lookup = {c["workload"]: c for c in selected}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.3), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.705, bottom=0.22, wspace=0.16)
    fig.text(0.055, 0.955, "Before / after codec optimisation", fontsize=20, weight="bold", va="top")
    fig.text(0.055, 0.895, "64 MiB inputs · warm resident API throughput · matching input and decode-stream hashes",
             fontsize=11, color="#475569", va="top")
    summaries = [case["timings"][op][f"{which}_mib_per_second"]
                 for case in selected for op in OPERATIONS for which in ("before", "after")]
    limits = (min(s["minimum"] for s in summaries) / 1.6,
              max(s["maximum"] for s in summaries) * 3)
    titles = ("Compression", "Frozen-stream decoding", "Stdlib level-6 decoding")
    subtitles = ("Each encoder's own output", "Identical baseline-encoded bytes", "Identical stdlib-encoded bytes")
    for ax, operation, title, subtitle in zip(axes, OPERATIONS, titles, subtitles):
        ax.set_title(title, loc="left", pad=31)
        ax.text(0, 1.05, subtitle, transform=ax.transAxes, fontsize=9, color="#475569")
        ax.set_xscale("log")
        ax.set_xlim(limits)
        ax.set_ylim(len(workloads) - 0.5, -0.5)
        ax.set_yticks(range(len(workloads)), [LABELS[w] for w in workloads])
        ax.tick_params(axis="y", length=0, pad=10)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(FixedLocator([10**p for p in range(-2, 8)]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="x", zorder=0)
        for index, workload in enumerate(workloads):
            if index % 2 == 0:
                ax.axhspan(index - 0.45, index + 0.45, color="#F1F5F9", zorder=0)
            timing = lookup[workload]["timings"][operation]
            for which, offset, color, marker in zip(("before", "after"), (-0.13, 0.13), COLORS, ("o", "D")):
                rate = timing[f"{which}_mib_per_second"]
                ax.errorbar(rate["median"], index + offset,
                            xerr=[[rate["median"] - rate["minimum"]], [rate["maximum"] - rate["median"]]],
                            color=color, marker=marker, linestyle="none", markersize=6,
                            capsize=3, elinewidth=1, zorder=3)
            ax.text(0.985, index, f"{timing['speedup']:.2f}×", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=9, color="#334155")
        ax.set_xlabel("Throughput (MiB/s) · log scale", labelpad=10)
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", label=label)
               for c, m, label in zip(COLORS, ("o", "D"), ("Before", "After"))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.135, 0.858),
               frameon=False, ncols=2, columnspacing=2.2)
    fig.text(0.47, 0.825, "Row labels show speedup (after / before throughput).", fontsize=9, color="#475569")
    sample_note = f"{comparison['samples_per_measurement']} samples per point"
    if "aggregation" in comparison:
        runs = len(comparison["aggregation"]["before"]["included_run_ids"])
        sample_note += f" combined from {runs} runs per version (ABBA)"
    fig.text(0.055, 0.113, f"{comparison['gpu']['name']} · {sample_note} · "
             "error bars: sample min–max, not confidence intervals", fontsize=9, color="#475569")
    fig.text(0.055, 0.081, "Warm synchronized API wall time includes allocations; startup, generation, transfers and correctness checks excluded.",
             fontsize=9, color="#475569")
    fig.text(0.055, 0.049, "CPU codec forbidden inside measured calls. CUDA-event kernel profiles are separate and are not plotted.",
             fontsize=9, color="#475569")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results" / "rerun-before.json")
    parser.add_argument("--candidate", type=Path, default=ROOT / "results" / "rerun-after.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    comparison = compare(read_profile(args.baseline), read_profile(args.candidate))
    comparison["profiles"] = {"before": args.baseline.name, "after": args.candidate.name}
    comparison["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    comparison["renderer"] = {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "backend": "Agg"}
    comparison["plotted_input_bytes"] = SIZE
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison["exports_sha256"] = save_figure(render(comparison), args.output_dir, "optimisation-throughput")
    (args.output_dir / "optimisation-manifest.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"Validated {comparison['validated_cases']} paired cases; saved optimisation-throughput.png/.svg/.pdf and optimisation-manifest.json")


if __name__ == "__main__":
    main()

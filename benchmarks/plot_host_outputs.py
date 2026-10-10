#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Plot completed host-output decode latency from a source-bound report."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_results import configure_style, report_identifier, save_figure

ROOT = Path(__file__).resolve().parent
SIZES = (65_536, 1_048_576, 67_108_864)
CONTRACTS = ("ndarray", "memoryview", "bytes")
BACKENDS = ("cpu", "cuda")
SERIES = tuple(f"{backend}_{contract}" for contract in CONTRACTS for backend in BACKENDS)
COLORS = {"cpu": "#0072B2", "cuda": "#D55E00"}
LABELS = {"ndarray": "NumPy array", "memoryview": "Memoryview", "bytes": "Bytes"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid {name}")
    require(value > 0 if positive else value >= 0, f"Invalid {name}")
    return value


def digest(value, name, length=64):
    lengths = (length,) if isinstance(length, int) else length
    require(isinstance(value, str) and len(value) in lengths and all(char in "0123456789abcdef" for char in value), f"Invalid {name}")


def case_statistics(case):
    rows = {series["name"]: series for series in case["series"]}
    return {
        "output_bytes": case["output_bytes"],
        "series": [{"name": name, "samples": len(rows[name]["samples"]),
                    "median_wall_ns": statistics.median(sample["wall_ns"] for sample in rows[name]["samples"]),
                    "min_wall_ns": min(sample["wall_ns"] for sample in rows[name]["samples"]),
                    "max_wall_ns": max(sample["wall_ns"] for sample in rows[name]["samples"]),
                    "backing_storage": rows[name]["backing_storage"]} for name in SERIES],
    }


def load_report(path):
    report = json.loads(path.read_text())
    require(report.get("schema_version", report.get("schema")) == 1 and report.get("complete") is True, "Expected complete host-output schema 1")
    require(report["kind"] == "host_outputs", "Wrong report kind")
    digest(report["source_revision"], "source revision", (40, 64))
    digest(report["harness_sha256"], "harness hash")
    for field in ("source_sha256", "dependencies_sha256"):
        require(isinstance(report[field], dict) and report[field], f"Missing {field}")
        for name, value in report[field].items():
            digest(value, name)
    for field in ("cache_key", "library_sha256", "build_sha256"):
        digest(report["native_build"][field], field)
    require(report["environment"]["gpu_uuid"], "Missing measured GPU identity")
    args = report["arguments"]
    require(tuple(sorted(args["sizes"])) == SIZES and args["workload"] == "float32", "Unexpected workload matrix")
    require(type(args["samples"]) is int and args["samples"] > 0, "Invalid sample count")
    require(type(args["warmups"]) is int and args["warmups"] > 0, "Invalid warmup count")
    cases = report["cases"]
    require(len(cases) == len(SIZES) and sorted(case["output_bytes"] for case in cases) == list(SIZES), "Missing or duplicate sizes")
    for case in cases:
        require(case["workload"] == "float32" and case["input_bytes"] > 0, "Invalid case input")
        require(case["fixture"]["seed"] == args["seed"] and case["order_seed"] == args["order_seed"] + case["output_bytes"], "Case seeds differ from arguments")
        require(case["fixture"]["stdlib_level"] == 6, "Expected identical stdlib level-6 input streams")
        digest(case["fixture"]["raw_sha256"], "payload hash")
        digest(case["fixture"]["stdlib_stream_sha256"], "stream hash")
        require(case["fixture"]["stdlib_stream_bytes"] == case["input_bytes"], "Encoded input size differs")
        require(len(case["series"]) == len(SERIES) and {item["name"] for item in case["series"]} == set(SERIES), "Expected all six output series")
        order = case["order"]
        require(len(order) == len(SERIES) * (args["samples"] + args["warmups"]), "Incomplete interleaved schedule")
        observations = {}
        for series in case["series"]:
            require(series["backend"] in BACKENDS and series["contract"] in CONTRACTS, "Unknown output contract")
            require(series["name"] == f"{series['backend']}_{series['contract']}", "Series identity differs")
            require(isinstance(series["backing_storage"], str) and series["backing_storage"], "Missing backing-storage description")
            for group, warmup in (("samples", False), ("warmups", True)):
                require(len(series[group]) == args["warmups" if warmup else "samples"], "Wrong timing count")
                require([sample["round"] for sample in series[group]] == list(range(len(series[group]))), "Missing or duplicate rounds")
                for sample in series[group]:
                    require(sample["complete"] is True and sample["byte_exact"] is True and sample["readonly"] is True, "Incomplete or unvalidated output")
                    require(sample["warmup"] is warmup and sample["output_bytes"] == case["output_bytes"], "Output sample identity differs")
                    index = sample["order_index"]
                    require(type(index) is int and 0 <= index < len(order) and index not in observations, "Invalid or duplicate observation index")
                    require(order[index] == {"warmup": warmup, "round": sample["round"], "series": series["name"]}, "Sample differs from interleaved schedule")
                    observations[index] = sample
                    require(sample["output_type"] == series["contract"], "Output type differs from contract")
                    if series["contract"] == "ndarray":
                        require(sample["dtype"] == "uint8" and sample["shape"] == [case["output_bytes"]], "Invalid ndarray contract")
                    elif series["contract"] == "memoryview":
                        require(sample["format"] == "B" and sample["shape"] == [case["output_bytes"]], "Invalid memoryview contract")
                        require(sample["backing_type"] == ("bytes" if series["backend"] == "cpu" else "ndarray"), "Unexpected memoryview backing storage")
                    start = number(sample["start_ns"], "start time")
                    end = number(sample["end_ns"], "end time")
                    wall = number(sample["wall_ns"], "latency", positive=True)
                    require(end - start == wall, "Latency differs from timestamp interval")
                    for field in ("thread_cpu_ns", "minor_faults", "major_faults"):
                        number(sample[field], field)
            timings = [sample["wall_ns"] for sample in series["samples"]]
            for field, expected in (("median_wall_ns", statistics.median(timings)), ("min_wall_ns", min(timings)), ("max_wall_ns", max(timings))):
                require(series["summary"][field] == expected, f"Incorrect {field}")
            for field in ("thread_cpu_ns", "minor_faults", "major_faults"):
                require(series["summary"]["median_" + field] == statistics.median(sample[field] for sample in series["samples"]), f"Incorrect median {field}")
        require(len(observations) == len(order), "Missing scheduled observations")
        require(all(observations[index]["end_ns"] <= observations[index + 1]["start_ns"] for index in range(len(order) - 1)), "Synchronous observations overlap or are reordered")
    return report, sorted(cases, key=lambda case: case["output_bytes"])


def render(report, cases):
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 7.0), sharey=True)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.74, bottom=0.28, wspace=0.16)
    fig.suptitle("Decode latency · host input → completed host output", x=0.16, y=0.96, ha="left", fontsize=19, weight="bold")
    environment = report["environment"]
    device = environment.get("device_kind", "CUDA GPU")
    fig.text(0.16, 0.90, f"{device} · float32 payloads · identical stdlib level-6 streams · source {report['source_revision'][:12]}", fontsize=10.5)
    fig.text(0.16, 0.855, "Transfers, public API work, status checks and output conversion included; byte oracle and file I/O excluded.", fontsize=10)
    labels = [f"{LABELS[contract]} · {backend.upper()}" for contract in CONTRACTS for backend in BACKENDS]
    for ax, case, size_label in zip(axes, cases, ("64 KiB", "1 MiB", "64 MiB")):
        ax.set_title(size_label + " raw payload", pad=13)
        rows = {series["name"]: series for series in case["series"]}
        maximum = max(sample["wall_ns"] / 1e6 for series in rows.values() for sample in series["samples"])
        for row, name in enumerate(SERIES):
            series = rows[name]
            values = np.array([sample["wall_ns"] / 1e6 for sample in series["samples"]])
            median = statistics.median(values)
            color = COLORS[series["backend"]]
            ax.barh(row, median, height=0.54, color=color, alpha=0.20, linewidth=0)
            jitter = np.linspace(-0.17, 0.17, len(values))
            ax.scatter(values, row + jitter, s=22, color=color, marker="o" if series["backend"] == "cpu" else "s", edgecolors="white", linewidths=0.4, zorder=3)
            ax.text(0.985, row, f"{median:.3f}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, weight="bold", color=color)
        for separator in (1.5, 3.5):
            ax.axhline(separator, color="#CBD5E1", linewidth=0.6)
        ax.set_xlim(0, maximum * 1.28)
        ax.set_ylim(len(SERIES) - 0.45, -0.65)
        ax.set_yticks(range(len(SERIES)), labels, fontsize=10)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Completed latency (ms)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    args = report["arguments"]
    fig.text(0.16, 0.19, f"Dots: all {args['samples']} samples per series. Bars and labels: median latency in ms. {args['warmups']} warmups excluded; series interleaved in randomized rounds.", fontsize=9.5)
    fig.text(0.16, 0.14, "Memoryviews have different backing storage: CPU views decoded bytes; CUDA views a completed host NumPy array.", fontsize=9.5)
    fig.text(0.16, 0.09, "CPU: stdlib zlib. CUDA: synchronous host API. All outputs reside on the host. Each panel has its own latency scale.", fontsize=9.5)
    fig.text(0.16, 0.04, "Shared-workstation measurements; sample spread is displayed directly. These timings do not establish isolated kernel costs.", fontsize=9.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/host-outputs/rtx4090-float32.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/host-outputs")
    parser.add_argument("--prefix", default="rtx4090", help="hardware identifier for the manifest filename")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        parser.error("prefix must be a filename component")
    report, cases = load_report(args.input)
    if args.validate_only:
        print("Validated three sizes and all six host-output contracts")
        return
    configure_style()
    plt.rcParams["svg.hashsalt"] = "cuda-zlib-host-outputs-v1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = "host-output-latency"
    exports = save_figure(render(report, cases), args.output_dir, name)
    evidence_fields = ("dependencies_sha256", "benchmark_sha256", "profile_resident_sha256", "native_build", "environment", "arguments", "gpu_snapshot_before", "gpu_snapshot_after", "methodology", "validation", "warnings")
    manifest = {
        "schema": 1, "kind": "host_outputs", "source_report": report_identifier(args.input), "source_report_sha256": sha256(args.input),
        "plotter_sha256": sha256(Path(__file__)),
        "measurement_source": {"revision": report["source_revision"], "harness_sha256": report["harness_sha256"], "sha256": report["source_sha256"]},
        "report_evidence": {field: report[field] for field in evidence_fields if field in report},
        "style_helpers_sha256": {"benchmarks/plot_results.py": sha256(ROOT / "plot_results.py")},
        "exports": exports,
        "figures": {name: {"exports": exports, "sizes": list(SIZES), "series_order": list(SERIES), "statistic": "median of individual wall_ns", "units": "ms", "case_statistics": [case_statistics(case) for case in cases]}},
        "renderer": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "backend": matplotlib.get_backend()},
        "scope": "host compressed bytes through completed consumer output, including transfers/status checks/conversion; byte oracle and file I/O excluded",
        "limitations": ["Independent linear latency scale for each size", "All individual samples shown; jitter only separates points", "CPU and CUDA memoryviews have different backing storage", "Shared workstation; no isolated kernel cost claim"],
    }
    (args.output_dir / f"{args.prefix}-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("Rendered one host-output figure, three exports and provenance manifest")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Plot validated small-file measurements as PNG, SVG and PDF figures."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

REPORT_TYPE = "cuda-zlib-small-batch"
LABELS = {"zeros": "Zeros", "text": "Synthetic text", "random": "Random bytes"}
COLORS = {"cpu": "#009E73", "single": "#0072B2", "batch": "#D55E00"}
MARKERS = {"cpu": "^", "single": "o", "batch": "s"}
LINESTYLES = {"cpu": "-.", "single": "--", "batch": "-"}
CPU_KEYS = {"cpu_compress_level1", "cpu_decompress_level6"}
CUDA_KEYS = {f"cuda_{operation}_{method}_{scope}"
             for operation in ("compress", "decompress")
             for method in ("single", "batch")
             for scope in ("resident", "host_bytes")}


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}: expected a positive finite number")
    return value


def positive_int(value, label):
    positive(value, label)
    if not isinstance(value, int):
        raise ValueError(f"{label}: expected an integer")
    return value


def equal(actual, expected, label):
    positive(actual, label)
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{label}: stored summary disagrees with its samples")


def hash_value(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label}: expected a SHA-256 digest")


def load_results(path):
    """Reject partial matrices, invalid provenance, and inconsistent timings."""
    source = path.read_bytes()
    report = json.loads(source)
    if report.get("schema_version") != 1 or report.get("report_type") != REPORT_TYPE or report.get("complete") is not True:
        raise ValueError("Expected a complete schema-1 small-batch report")
    args = report["arguments"]
    samples = positive_int(args["samples"], "samples")
    for field in ("sizes", "counts"):
        if not isinstance(args[field], list) or not args[field] or len(set(args[field])) != len(args[field]):
            raise ValueError(f"{field}: expected distinct nonempty integers")
        for value in args[field]:
            positive_int(value, field)
    if not isinstance(args["workloads"], list) or not args["workloads"] or len(set(args["workloads"])) != len(args["workloads"]) or not set(args["workloads"]) <= set(LABELS):
        raise ValueError("workloads: expected distinct known workloads")
    if type(args["cpu_only"]) is not bool or type(args["roundtrip"]) is not bool:
        raise ValueError("cpu_only and roundtrip must be booleans")
    expected_keys = set(CPU_KEYS)
    encoders = {"zlib1", "zlib6"}
    if not args["cpu_only"]:
        expected_keys |= CUDA_KEYS
        encoders.add("cuda")
        if args["roundtrip"]:
            expected_keys.add("cuda_roundtrip_batch_resident")
        for field in ("gpu", "cuda_platform", "jax", "jaxlib"):
            if not report["environment"].get(field):
                raise ValueError(f"Missing CUDA environment field: {field}")
        if not any(path.endswith(".cuh") for path in report["source"]["sha256"]):
            raise ValueError("Missing native header source hashes")
    if not report["environment"].get("cpu") or not report["source"]["sha256"]:
        raise ValueError("Missing CPU/source provenance")
    for label, digest in report["source"]["sha256"].items():
        hash_value(digest, label)
    expected = {(workload, size, count) for workload in args["workloads"]
                for size in args["sizes"] for count in args["counts"]}
    cases = {}
    for case in report["cases"]:
        identity = (case["workload"], case["file_bytes"], case["file_count"])
        if identity not in expected or identity in cases:
            raise ValueError(f"Unexpected or duplicate case: {identity}")
        workload, size, count = identity
        if case["total_input_bytes"] != size * count:
            raise ValueError(f"Wrong byte denominator: {identity}")
        if not case["validation"].startswith(("all timed workflow outputs byte-exact", "last returned output from each timed workflow byte-exact")):
            raise ValueError(f"Missing byte-exact validation: {identity}")
        if set(case["timings"]) != expected_keys:
            raise ValueError(f"Incomplete or unknown timing keys: {identity}")
        for key, timing in case["timings"].items():
            values = timing["seconds"]
            if not isinstance(values, list) or len(values) != samples:
                raise ValueError(f"Wrong sample count: {identity} {key}")
            for value in values:
                positive(value, f"{identity} {key} sample")
            median = statistics.median(values)
            equal(timing["median_seconds"], median, f"{identity} {key} median")
            equal(timing["min_seconds"], min(values), f"{identity} {key} minimum")
            equal(timing["max_seconds"], max(values), f"{identity} {key} maximum")
            equal(timing["median_seconds_per_file"], median / count, f"{identity} {key} per-file time")
            equal(timing["mib_per_second"], size * count / 2**20 / median, f"{identity} {key} throughput")
            positive(timing["warmup_seconds"], f"{identity} {key} warmup")
        if len(case["input_sha256"]) != count:
            raise ValueError(f"Wrong input digest count: {identity}")
        for digest in case["input_sha256"]:
            hash_value(digest, str(identity))
        if set(case["encoded_bytes"]) != encoders or set(case["encoded_sha256"]) != encoders:
            raise ValueError(f"Wrong encoded provenance keys: {identity}")
        for encoder in encoders:
            if len(case["encoded_bytes"][encoder]) != count or len(case["encoded_sha256"][encoder]) != count:
                raise ValueError(f"Wrong encoded file count: {identity} {encoder}")
            for value in case["encoded_bytes"][encoder]:
                positive_int(value, f"{identity} {encoder} encoded extent")
                if value < 8:
                    raise ValueError("Encoded stream is smaller than a zlib frame")
            for digest in case["encoded_sha256"][encoder]:
                hash_value(digest, f"{identity} {encoder}")
        cases[identity] = case
    if set(cases) != expected:
        raise ValueError("Incomplete benchmark matrix")
    return report, cases, hashlib.sha256(source).hexdigest()


def format_bytes(size):
    return f"{size // 1024} KiB" if size >= 1024 and size % 1024 == 0 else f"{size} B"


def series(operation, scope, cpu_only):
    level = "level1" if operation == "compress" else "level6"
    cpu_label = "CPU zlib level 1" if operation == "compress" else "CPU zlib"
    values = [(cpu_label, f"cpu_{operation}_{level}", "cpu")]
    if not cpu_only:
        values += [("CUDA single-file loop", f"cuda_{operation}_single_{scope}", "single"),
                   ("CUDA packed batch", f"cuda_{operation}_batch_{scope}", "batch")]
    return values


def draw(report, cases, workload, scope):
    sizes, counts = sorted(report["arguments"]["sizes"]), sorted(report["arguments"]["counts"])
    cpu_only = report["arguments"]["cpu_only"]
    figure, axes = plt.subplots(len(sizes), 2, figsize=(11, 2.9 * len(sizes) + 2), squeeze=False)
    scope_label = "Resident CUDA buffers" if scope == "resident" else "Host bytes → host bytes"
    if cpu_only:
        scope_label = "Host bytes → host bytes · CPU measurements"
    figure.suptitle(f"Independent small zlib streams · {LABELS[workload]}", fontsize=16, weight="bold", y=0.99)
    device = report["environment"].get("gpu", "")
    figure.text(0.5, 0.93, f"{scope_label}" + (f" · {device}" if device else ""), ha="center", fontsize=11)
    for row, size in enumerate(sizes):
        for column, operation in enumerate(("compress", "decompress")):
            axis = axes[row, column]
            for label, key, method in series(operation, scope, cpu_only):
                timings = [cases[(workload, size, count)]["timings"][key] for count in counts]
                medians = np.asarray([timing["median_seconds_per_file"] * 1e6 for timing in timings])
                low = np.asarray([timing["min_seconds"] / count * 1e6 for timing, count in zip(timings, counts)])
                high = np.asarray([timing["max_seconds"] / count * 1e6 for timing, count in zip(timings, counts)])
                axis.errorbar(counts, medians, yerr=np.vstack((medians - low, high - medians)),
                              color=COLORS[method], marker=MARKERS[method], linewidth=1.7,
                              linestyle=LINESTYLES[method],
                              markersize=5, capsize=3, label=label)
            title = "Compression" if operation == "compress" else "Decompression · stdlib level 6 streams"
            axis.set_title(f"{title}\n{format_bytes(size)} per file", fontsize=11)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.xaxis.set_major_locator(FixedLocator(counts))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
            axis.xaxis.set_minor_locator(NullLocator())
            axis.set_xlabel("Files per call")
            axis.set_ylabel("Amortized time per file (µs)")
            axis.grid(True, which="major", alpha=0.35)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.legend(fontsize=8, loc="best")
    samples = report["arguments"]["samples"]
    notes = f"Median of {samples} completed calls; whiskers show sample minimum–maximum. Total call time divided by file count.\n"
    if cpu_only:
        notes += "CPU: single-threaded stdlib zlib level 1 compression; identical level 6 streams for decompression."
    elif scope == "resident":
        notes += "CUDA: warmed jax.jit, device outputs; status checked outside timing. CPU: single thread, host bytes outputs."
    else:
        notes += "Includes transfers, status checks and bytes copies for CUDA; CPU uses one thread."
    notes += "\nPanel-specific logarithmic y ranges; lines connect measured file counts only."
    figure.text(0.06, 0.025, notes, fontsize=8.5, color="#475569", va="bottom")
    figure.tight_layout(rect=(0.015, 0.085, 0.995, 0.90))
    return figure


def report_identifier(path):
    try:
        return path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        return path.name


def save_figure(figure, output):
    """Remove export timestamps and SVG trailing whitespace before hashing."""
    metadata = {
        ".png": {"Software": "cuda-zlib benchmark plotting"},
        ".svg": {"Date": None, "Creator": "cuda-zlib benchmark plotting"},
        ".pdf": {"CreationDate": None, "ModDate": None, "Creator": "cuda-zlib benchmark plotting"},
    }
    figure.savefig(output, dpi=180, metadata=metadata[output.suffix])
    if output.suffix == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("small-batch-figures"))
    parser.add_argument("--prefix", default="small-batch")
    parser.add_argument("--validate-only", action="store_true", help="validate the report without writing figures")
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix:
        parser.error("prefix must be a filename component")
    report, cases, digest = load_results(args.report)
    if args.validate_only:
        print(f"Validated {len(cases)} cases; source SHA-256 {digest}")
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "svg.hashsalt": "cuda-zlib-small-batch-v1",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    scopes = ("host_bytes",) if report["arguments"]["cpu_only"] else ("resident", "host_bytes")
    for workload in report["arguments"]["workloads"]:
        for scope in scopes:
            figure = draw(report, cases, workload, scope)
            for extension in ("png", "svg", "pdf"):
                output = args.output_dir / f"{args.prefix}-{workload}-{scope.replace('_', '-')}.{extension}"
                exports.append({"file": output.name, "sha256": save_figure(figure, output),
                                "workload": workload, "scope": scope,
                                "metric": "median total call seconds / file count, in microseconds"})
            plt.close(figure)
    manifest = {
        "schema_version": 1, "report_type": REPORT_TYPE,
        "source_report": report_identifier(args.report), "source_report_sha256": digest,
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "renderer": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "backend": "Agg"},
        "source_revision": report.get("source_revision", report["source"].get("git", {}).get("commit")),
        "harness_sha256": report.get("harness_sha256"),
        "measurement_source": report["source"], "environment": report["environment"],
        "arguments": report["arguments"], "methodology": report["methodology"],
        "plotted_operations": ["compression", "decompression"],
        "roundtrip_plotted": False,
        "point_statistic": "median completed call seconds / file count * 1e6",
        "error_bars": "sample minimum and maximum seconds / file count * 1e6; not confidence intervals",
        "series_styles": {method: {"color": COLORS[method], "marker": MARKERS[method], "linestyle": LINESTYLES[method]}
                          for method in COLORS},
        "exports": exports,
    }
    output = args.output_dir / f"{args.prefix}-manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(exports)} figure exports and {output}", flush=True)


if __name__ == "__main__":
    main()

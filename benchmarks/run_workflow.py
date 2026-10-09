#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Capture, normalize and plot checked compression/decompression workflows.

Use a fresh output directory. --compare-instrumentation also runs plain,
sampler-only and both Nsight trace configurations to expose observer overhead.
All subprocesses use this interpreter; existing CUDA/cache environment is kept.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "plain": (None, "none"),
    "telemetry": (None, "process-tree"),
    "cuda-nvtx": ("cuda,nvtx", "process-tree"),
    "cuda-nvtx-osrt": ("cuda,nvtx,osrt", "process-tree"),
    "plain-repeat": (None, "none"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stop_owned_tree(process):
    """Stop only this launch and its discovered descendants, including sessions."""
    import psutil
    try:
        owned = psutil.Process(process.pid)
        descendants = owned.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    _, alive = psutil.wait_procs(descendants, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run_logged(command, cwd, environment, log_path, timeout):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        launch = {"pid": process.pid, "cwd": str(cwd), "command": command,
                  "log": str(log_path), "stop": f"kill -TERM -- -{process.pid}"}
        write_json(log_path.with_suffix(".launch.json"), launch)
        print(json.dumps(launch), flush=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            stop_owned_tree(process)
            raise
        require(code == 0, f"command exited {code}; see {log_path}")


def harness_command(args, operation, directory, telemetry):
    command = [sys.executable, str(ROOT / "benchmarks/profile_workflow.py"),
               "--operation", operation, "--output", str(directory / "telemetry.json"),
               "--source-revision", args.source_revision, "--gpu-uuid", args.gpu_uuid,
               "--telemetry", telemetry, "--size", str(args.size), "--workload", args.workload,
               "--seed", str(args.seed), "--chunk-bytes", str(args.chunk_bytes),
               "--warmups", str(args.warmups), "--iterations", str(args.iterations),
               "--device", str(args.device), "--sample-ms", str(args.sample_ms),
               "--timeout", str(args.timeout)]
    return command


def nsight_command(args, directory, trace, command):
    return [args.nsys, "profile", "--trace=" + trace, "--nvtx-domain-exclude=TSL",
            "--cuda-graph-trace=node", "--cuda-event-trace=false", "--sample=none",
            "--cpuctxsw=none", "--wait=all", "--discard-environment=true", "--output",
            str(directory / "capture"), *command, "--nvtx-library", str(args.nvtx_library)]


def host_summary(telemetry):
    require(telemetry["complete"] is True and telemetry["worker"]["exit_code"] == 0,
            "incomplete workflow")
    phases = telemetry["phase_intervals"]
    loop = [row for row in phases if row["name"] == "measured_loop"]
    require(len(loop) == 1, "missing measured loop")
    values = {}
    for phase in phases:
        if phase.get("warmup") is False:
            values.setdefault(phase["name"], []).append((phase["end_ns"] - phase["start_ns"]) / 1e6)
    return {"measured_loop_ms": (loop[0]["end_ns"] - loop[0]["start_ns"]) / 1e6,
            "phases": {key: {"count": len(v), "median_ms": statistics.median(v),
                             "minimum_ms": min(v), "maximum_ms": max(v), "sum_ms": sum(v)}
                       for key, v in sorted(values.items())}}


def capture(args, operation, variant, environment):
    trace, sampling = VARIANTS[variant]
    directory = args.output_dir / operation / variant
    directory.mkdir(parents=True)
    command = harness_command(args, operation, directory, sampling)
    if trace:
        command = nsight_command(args, directory, trace, command)
    command_path, log_path = directory / "command.json", directory / "capture.log"
    write_json(command_path, command)
    started = datetime.now(timezone.utc).isoformat()
    run_logged(command, ROOT, environment, log_path, args.timeout + 180)
    telemetry_path = directory / "telemetry.json"
    telemetry = json.loads(telemetry_path.read_text())
    summary = host_summary(telemetry)
    record = {"operation": operation, "variant": variant, "trace": trace, "telemetry": sampling,
              "source_revision": args.source_revision, "harness_sha256": telemetry["harness_sha256"],
              "runner_sha256": digest(Path(__file__)), "capture_started_utc": started,
              "host_statistics": summary, "directory": str(directory),
              "telemetry_sha256": digest(telemetry_path)}
    if trace:
        sqlite_path, export_log = directory / "capture.sqlite", directory / "export.log"
        export_command = [args.nsys, "export", "--type", "sqlite", "--output", str(sqlite_path),
                          str(directory / "capture.nsys-rep")]
        run_logged(export_command, ROOT, environment, export_log, 120)
        artifacts = {}
        filenames = {"nsys_report": "capture.nsys-rep", "telemetry": "telemetry.json",
                     "command": "command.json", "log": "capture.log", "export_log": "export.log",
                     "worker_log": "telemetry-worker.log", "worker_result": "telemetry-worker.json",
                     "sqlite": "capture.sqlite"}
        for key, filename in filenames.items():
            item = {"path": filename, "sha256": digest(directory / filename)}
            if key == "sqlite":
                item["private"] = True
            else:
                item["published_path"] = f"benchmarks/results/workflow/captures/{operation}/{variant}/{filename}"
            artifacts[key] = item
        version = subprocess.check_output([args.nsys, "--version"], text=True).strip()
        receipt = {"source_revision": args.source_revision, "capture_started_utc": started,
                   "toolchain": {"nsys": version, "driver": telemetry["selected_gpu"]["driver_version"]},
                   "runner_sha256": record["runner_sha256"], "artifacts": artifacts}
        receipt_path, normalized = directory / "capture.json", directory / "normalized.json"
        write_json(receipt_path, receipt)
        extract = [sys.executable, str(ROOT / "benchmarks/extract_workflow.py"),
                   "--sqlite", str(sqlite_path), "--telemetry", str(telemetry_path),
                   "--receipt", str(receipt_path), "--output", str(normalized)]
        run_logged(extract, ROOT, environment, directory / "extract.log", 180)
        record.update(normalized_path=str(normalized), normalized_sha256=digest(normalized),
                      capture_receipt_sha256=digest(receipt_path))
    write_json(directory / "run.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--operation", choices=("both", "compress", "decompress"), default="both")
    parser.add_argument("--source-revision")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--nvtx-library", type=Path, required=True)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--trace", choices=("cuda,nvtx", "cuda,nvtx,osrt"), default="cuda,nvtx")
    parser.add_argument("--compare-instrumentation", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--size", type=int, default=64 * 1024**2)
    parser.add_argument("--workload", choices=("zeros", "text", "uint32", "float32", "random"), default="float32")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--chunk-bytes", type=int, default=32768)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sample-ms", type=float, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.nvtx_library = args.nvtx_library.resolve()
    require(not args.output_dir.exists(), "output directory must be fresh")
    require(0 < args.size < 256 * 1024**2 and args.iterations > 0 and args.warmups > 0,
            "positive bounded size and iteration counts required")
    require(256 <= args.chunk_bytes <= 65535 and args.device >= 0,
            "invalid compression chunk size or CUDA ordinal")
    require(math.isfinite(args.timeout) and args.timeout > 0 and
            math.isfinite(args.sample_ms) and args.sample_ms > 0, "positive finite timing parameters required")
    if args.source_revision is None:
        args.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", args.source_revision), "full source revision required")
    require(args.nvtx_library.is_file(), "NVTX library does not exist")
    operations = ("compress", "decompress") if args.operation == "both" else (args.operation,)
    variants = list(VARIANTS) if args.compare_instrumentation else ["cuda-nvtx-osrt" if "osrt" in args.trace else "cuda-nvtx"]
    args.output_dir.mkdir(parents=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    records = []
    def interrupted(signum, frame):
        raise InterruptedError(f"runner received signal {signum}")
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        for operation in operations:
            for variant in variants:
                records.append(capture(args, operation, variant, environment))
                write_json(args.output_dir / "comparison.json", {"complete": False, "runs": records})
        if not args.no_plots:
            chosen = "cuda-nvtx-osrt" if "osrt" in args.trace else "cuda-nvtx"
            plot = [sys.executable, str(ROOT / "benchmarks/plot_workflow.py")]
            for operation in operations:
                plot.extend(["--" + operation, str(args.output_dir / operation / chosen / "normalized.json")])
            plot.extend(["--output-dir", str(args.output_dir / "figures")])
            run_logged(plot, ROOT, environment, args.output_dir / "plot.log", 180)
        write_json(args.output_dir / "comparison.json", {"complete": True, "runs": records})
    except BaseException as error:
        write_json(args.output_dir / "failure.json", {"type": type(error).__name__, "message": str(error), "runs": records})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    print(f"Completed {len(records)} checked workflow runs: {args.output_dir}")


if __name__ == "__main__":
    main()

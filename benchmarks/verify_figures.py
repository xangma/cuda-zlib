#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Check published figure and measurement identities without JAX or Matplotlib.

Works in a checkout or source export. Numerical report validation belongs to
the plotters; this check detects stale artifacts.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "benchmarks/figures"
EXTENSIONS = {".png", ".svg", ".pdf"}
MANIFESTS = (
    ("manifest.json", "general", "benchmarks/plot_results.py", 15),
    ("small-batch/rtx4090-manifest.json", "batch", "benchmarks/plot_small_batch.py", 18),
    ("resident-checked-manifest.json", "resident", "benchmarks/plot_resident.py", 3),
    ("nsight/rtx4090-manifest.json", "nsight", "benchmarks/plot_nsight.py", 6),
    ("timeline/rtx4090-manifest.json", "timeline", "benchmarks/plot_timeline.py", 6),
    ("workflow/rtx4090-manifest.json", "workflow", "benchmarks/plot_workflow.py", 12),
    ("host-outputs/rtx4090-manifest.json", "host_outputs", "benchmarks/plot_host_outputs.py", 3),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def repo_path(name):
    require(isinstance(name, str) and bool(name), "missing repository-relative path")
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
            f"nonportable path: {name}; use a repository-relative path")
    result = (ROOT / path).resolve()
    require(result.is_relative_to(ROOT), f"path escapes the checkout: {name}")
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hash(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"invalid SHA-256 for {path.relative_to(ROOT)}")
    require(digest(path) == expected,
            f"stale {path.relative_to(ROOT)}: SHA-256 differs from recorded identity")


def read_json(path):
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"expected a JSON object: {path.relative_to(ROOT)}")
    return value


def check_sources(report, kind):
    revision = report["source_revision"]
    require(isinstance(revision, str) and
            re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision),
            "source_revision must identify the measured Git commit")
    package = {path.relative_to(ROOT).as_posix()
               for path in (ROOT / "src/cuda_zlib").rglob("*")
               if path.is_file() and path.suffix in {".py", ".cu", ".cuh"}}
    if kind == "batch":
        recorded = dict(report["source"]["sha256"])
        required = package | {"benchmarks/benchmark.py", "benchmarks/small_batch.py",
                              "benchmarks/plot_small_batch.py"}
        harness = "benchmarks/small_batch.py"
    else:
        hashes = (report["environment"]["codec_sha256"] if kind == "general"
                  else report["source_sha256"])
        recorded = {"src/cuda_zlib/" + name: value for name, value in hashes.items()}
        harness = ("benchmarks/benchmark.py" if kind == "general" else
                   "benchmarks/profile_workflow.py" if kind == "workflow" else
                   "benchmarks/profile_timeline.py" if kind == "timeline" else
                   "benchmarks/profile_resident.py")
        required = package | {harness}
    require(recorded.get(harness, report["harness_sha256"]) == report["harness_sha256"],
            f"conflicting harness identities for {harness}")
    recorded[harness] = report["harness_sha256"]
    require(set(recorded) == required,
            f"stale source inventory: missing {sorted(required - set(recorded))}; "
            f"unexpected {sorted(set(recorded) - required)}")
    for name, expected in recorded.items():
        check_hash(repo_path(name), expected)


def verify_exports(path, exports, count):
    name = path.relative_to(FIGURES)
    require(len(exports) == count, f"{name}: expected {count} exports, found {len(exports)}")
    paths, families = set(), {}
    for filename, expected in exports:
        require(isinstance(filename, str) and Path(filename).name == filename,
                f"{name}: export must be a filename: {filename}")
        export = repo_path(str(path.parent.relative_to(ROOT) / filename))
        require(export.suffix in EXTENSIONS, f"unexpected export format: {filename}")
        require(export not in paths, f"duplicate export: {filename}")
        check_hash(export, expected)
        paths.add(export)
        families.setdefault(export.stem, set()).add(export.suffix)
    require(all(formats == EXTENSIONS for formats in families.values()),
            f"{name}: every figure must have PNG, SVG and PDF exports")
    return paths


def verify_workflow_artifacts(report, operation):
    artifacts, hashes = report["artifacts"], report["artifact_sha256"]
    required = {"nsys_report", "telemetry", "command", "log", "export_log",
                "worker_log", "worker_result", "capture_receipt", "sqlite"}
    require(required <= set(artifacts), "missing workflow capture evidence")
    published = {item["path"]: item["sha256"] for key, item in artifacts.items() if key != "sqlite"}
    require(len(published) == len(artifacts) - 1 and hashes == published,
            "workflow artifact inventory differs from report records")
    capture_root = ROOT / "benchmarks/results/workflow/captures" / operation
    for filename, expected in hashes.items():
        artifact = repo_path(filename)
        require(artifact.is_relative_to(capture_root),
                f"workflow artifact outside {operation} capture directory: {filename}")
        check_hash(artifact, expected)
    require(artifacts["nsys_report"]["path"].endswith(".nsys-rep"), "missing raw workflow Nsight report")
    sqlite = artifacts["sqlite"]
    require(sqlite.get("private") is True and "path" not in sqlite and "published_path" not in sqlite,
            "workflow SQLite must remain private")
    require(isinstance(sqlite["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", sqlite["sha256"]),
            "invalid private workflow SQLite identity")
    receipt = read_json(repo_path(artifacts["capture_receipt"]["path"]))
    require(receipt["source_revision"] == report["source_revision"] and
            receipt["runner_sha256"] == report["runner_sha256"], "workflow receipt source differs")
    require(all(receipt[field] == report[field] for field in ("capture_started_utc", "toolchain")),
            "workflow capture metadata differs from receipt")
    command = json.loads(repo_path(artifacts["command"]["path"]).read_bytes())
    require(command == report["command"], "workflow capture command differs from report")
    require(set(receipt["artifacts"]) == set(artifacts) - {"capture_receipt"},
            "workflow receipt artifact inventory differs")
    for key, item in receipt["artifacts"].items():
        recorded = artifacts[key]
        relative = Path(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in item["path"],
                "workflow receipt artifact path escapes capture directory")
        require(item["sha256"] == recorded["sha256"], f"workflow receipt {key} hash differs")
        if key == "sqlite":
            require(item.get("private") is True and "published_path" not in item and
                    item["path"] == recorded["filename"], "workflow private SQLite identity differs")
        else:
            require(item["published_path"] == recorded["path"] and
                    relative.name == Path(recorded["path"]).name,
                    f"workflow receipt {key} path differs")
    telemetry = read_json(repo_path(artifacts["telemetry"]["path"]))
    worker = read_json(repo_path(artifacts["worker_result"]["path"]))
    require(telemetry["complete"] is True and worker["complete"] is True and
            telemetry["worker"]["exit_code"] == 0 and telemetry["worker"]["result"] == worker,
            "workflow worker artifact differs from successful collector result")
    require(telemetry["worker"]["pid"] == worker["pid"] == report["worker"]["pid"] and
            report["worker"]["exit_code"] == 0, "workflow worker identity differs from report")
    require(telemetry["operation"] == telemetry["arguments"]["operation"] == worker["operation"] == operation,
            "workflow raw capture operation differs")
    require(telemetry["metadata_layout"] == report["metadata_layout"],
            "workflow raw capture metadata_layout differs from report")
    for field in ("source_revision", "harness_sha256", "source_sha256", "dependencies_sha256",
                  "native_build", "fixture", "environment"):
        require(telemetry[field] == worker[field] == report[field],
                f"workflow raw capture {field} differs from report")
    require(telemetry["worker"]["result_sha256"] == artifacts["worker_result"]["sha256"] and
            telemetry["worker"]["log_sha256"] == artifacts["worker_log"]["sha256"],
            "workflow collector worker hashes differ")


def verify_workflow_manifest(path, manifest, plotter, count):
    operations = {"compress", "decompress"}
    require(type(manifest["schema"]) is int and manifest["schema"] == 1 and manifest["kind"] == "workflow",
            "expected workflow schema-1 manifest")
    require(isinstance(manifest["operations"], list) and len(manifest["operations"]) == 2 and
            set(manifest["operations"]) == operations, "workflow requires both operations exactly once")
    for field in ("source_reports", "measurement_sources", "report_evidence", "warnings", "time_origins", "phase_semantics"):
        require(set(manifest[field]) == operations, f"workflow {field} requires both operations")
    check_hash(repo_path(plotter), manifest["plotter_sha256"])
    helpers = manifest["style_helpers_sha256"]
    require(set(helpers) == {"benchmarks/plot_results.py", "benchmarks/plot_nsight.py", "benchmarks/plot_timeline.py"},
            "missing workflow style helper identities")
    for filename, expected in helpers.items():
        check_hash(repo_path(filename), expected)
    reports = {}
    for operation in sorted(operations):
        item = manifest["source_reports"][operation]
        report_path = repo_path(item["path"])
        require(report_path.is_relative_to(ROOT / "benchmarks/results/workflow"),
                "workflow report must live under benchmarks/results/workflow")
        check_hash(report_path, item["sha256"])
        report = reports[operation] = read_json(report_path)
        require(report.get("complete") is True and type(report.get("schema_version")) is int and
                report["schema_version"] == 1 and report["operation"] == operation and
                report["kind"] == "workflow_timeline" and report["profiled"] is True,
                "expected complete profiled schema-1 workflow for " + operation)
        check_sources(report, "workflow")
        measured = {"revision": report["source_revision"], "harness_sha256": report["harness_sha256"],
                    "sha256": report["source_sha256"]}
        require(manifest["measurement_sources"][operation] == measured, "workflow measured source differs")
        for field, required in (("dependencies_sha256", {"benchmarks/profile_timeline.py", "benchmarks/profile_resident.py", "benchmarks/benchmark.py"}),
                                ("extractor_dependencies_sha256", {"benchmarks/extract_timeline.py", "benchmarks/extract_nsight.py"})):
            require(set(report[field]) == required, f"workflow {field} inventory differs")
            for filename, expected in report[field].items():
                check_hash(repo_path(filename), expected)
        check_hash(repo_path("benchmarks/extract_workflow.py"), report["extractor_sha256"])
        check_hash(repo_path("benchmarks/run_workflow.py"), report["runner_sha256"])
        require(report["benchmark_sha256"] == report["dependencies_sha256"]["benchmarks/benchmark.py"],
                "workflow benchmark identity differs")
        # Reuse the source proof without importing JAX, NumPy or Matplotlib.
        spec = importlib.util.spec_from_file_location("workflow_figure_source_proof", repo_path("benchmarks/extract_workflow.py"))
        extractor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extractor)
        extractor.source_binding(report, ROOT)
        evidence = {field: report[field] for field in ("native_build", "extractor_sha256", "artifact_sha256",
                                                       "dependencies_sha256", "extractor_dependencies_sha256")}
        require(manifest["report_evidence"][operation] == evidence, "workflow report evidence differs")
        for manifest_field, report_field in (("warnings", "warnings"), ("time_origins", "time_origin"),
                                              ("phase_semantics", "phase_semantics")):
            require(manifest[manifest_field][operation] == report[report_field], f"workflow {report_field} differs")
        require(report["fixture_oracle_verified"] is True, "workflow fixture oracle was skipped")
        require(report["metadata_layout"] == (["encoded_length", "status"] if operation == "compress" else ["status", "reserved_zero"]),
                "workflow metadata layout differs from operation")
        require(set(report["views"]) == {"whole_process", "measured_loop"}, "workflow requires both views")
        verify_workflow_artifacts(report, operation)
    require(manifest["measurement_sources"]["compress"] == manifest["measurement_sources"]["decompress"],
            "workflow operations have different measured sources")
    expected_figures = {f"{operation}-{suffix}": (operation, view) for operation in operations
                        for suffix, view in (("whole-process", "whole_process"), ("warmed-loop", "measured_loop"))}
    require(set(manifest["figures"]) == set(expected_figures), "workflow figure matrix is incomplete or duplicated")
    exports = {}
    for name, (operation, view) in expected_figures.items():
        figure = manifest["figures"][name]
        require((figure["operation"], figure["view"]) == (operation, view), "workflow figure identity differs")
        require(set(figure["exports"]) == {name + extension for extension in EXTENSIONS}, "workflow figure exports differ")
        exports.update(figure["exports"])
    require(exports == manifest["exports"], "workflow flat exports differ from figure records")
    return verify_exports(path, list(exports.items()), count)


def host_output_statistics(report, series_order):
    args = report["arguments"]
    sizes = [65536, 1048576, 67108864]
    names = {f"{backend}_{contract}" for backend in ("cpu", "cuda")
             for contract in ("ndarray", "memoryview", "bytes")}
    require(sorted(args["sizes"]) == sizes and args["workload"] == "float32", "unexpected host-output matrix")
    require(len(series_order) == 6 and set(series_order) == names, "host-output figure requires six series")
    require(all(type(args[field]) is int and args[field] > 0 for field in ("samples", "warmups")),
            "invalid host-output sample counts")
    require(len(report["cases"]) == 3 and sorted(case["output_bytes"] for case in report["cases"]) == sizes,
            "missing or duplicate host-output cases")
    statistics_rows = []
    for case in sorted(report["cases"], key=lambda item: item["output_bytes"]):
        size, fixture = case["output_bytes"], case["fixture"]
        require(case["workload"] == args["workload"] and fixture["seed"] == args["seed"] and
                fixture["stdlib_level"] == 6 and type(case["input_bytes"]) is int and
                8 <= case["input_bytes"] <= 2**28 and fixture["stdlib_stream_bytes"] == case["input_bytes"],
                "host-output fixture identity differs")
        require(all(isinstance(fixture[field], str) and re.fullmatch(r"[0-9a-f]{64}", fixture[field])
                    for field in ("raw_sha256", "stdlib_stream_sha256")), "invalid host-output fixture hashes")
        require(case["order_seed"] == args["order_seed"] + size, "host-output ordering seed differs")
        order = case["order"]
        expected = {(warmup, index, name) for warmup, count in ((True, args["warmups"]), (False, args["samples"]))
                    for index in range(count) for name in names}
        require(len(order) == len(expected) and all(type(row["warmup"]) is bool and type(row["round"]) is int for row in order)
                and {(row["warmup"], row["round"], row["series"]) for row in order} == expected,
                "incomplete or duplicate host-output round order")
        require([(not row["warmup"], row["round"]) for row in order] ==
                sorted((not row["warmup"], row["round"]) for row in order), "host-output rounds are out of order")
        require(len(case["series"]) == 6 and {item["name"] for item in case["series"]} == names,
                "host-output case requires all six series")
        indexed, observed = {}, {}
        for series in case["series"]:
            backend, contract = series["backend"], series["contract"]
            require(backend in ("cpu", "cuda") and contract in ("ndarray", "memoryview", "bytes") and
                    series["name"] == f"{backend}_{contract}" and isinstance(series["backing_storage"], str) and
                    bool(series["backing_storage"]), "host-output consumer identity differs")
            indexed[series["name"]] = series
            for group, warmup in (("warmups", True), ("samples", False)):
                rows = series[group]
                require(len(rows) == args[group] and {row["round"] for row in rows} == set(range(args[group])),
                        "host-output sample rounds differ")
                for row in rows:
                    require(row["complete"] is True and row["byte_exact"] is True and row["readonly"] is True and
                            row["warmup"] is warmup and type(row["round"]) is int and
                            row["output_bytes"] == size and row["output_type"] == contract,
                            "incomplete or invalid host-output consumer result")
                    if contract == "ndarray":
                        require(row["dtype"] == "uint8" and row["shape"] == [size], "host-output ndarray contract differs")
                    elif contract == "memoryview":
                        require(row["format"] == "B" and row["shape"] == [size] and
                                row["backing_type"] == ("bytes" if backend == "cpu" else "ndarray"),
                                "host-output memoryview contract differs")
                    metrics = ("start_ns", "end_ns", "wall_ns", "thread_cpu_ns", "minor_faults", "major_faults", "order_index")
                    require(all(type(row[field]) is int and row[field] >= 0 for field in metrics) and
                            row["wall_ns"] > 0 and row["end_ns"] - row["start_ns"] == row["wall_ns"],
                            "invalid host-output timing or counter")
                    index = row["order_index"]
                    require(index < len(order) and index not in observed and
                            order[index] == {"warmup": warmup, "round": row["round"], "series": series["name"]},
                            "host-output sample differs from round order")
                    observed[index] = row
            timings = [row["wall_ns"] for row in series["samples"]]
            summary = {"median_wall_ns": statistics.median(timings), "min_wall_ns": min(timings), "max_wall_ns": max(timings),
                       **{f"median_{field}": statistics.median(row[field] for row in series["samples"])
                          for field in ("thread_cpu_ns", "minor_faults", "major_faults")}}
            require(series["summary"] == summary, "host-output summary differs from raw samples")
        require(set(observed) == set(range(len(order))) and
                all(observed[index]["end_ns"] <= observed[index + 1]["start_ns"] for index in range(len(order) - 1)),
                "host-output samples are missing or overlap")
        statistics_rows.append({"output_bytes": size, "series": [
            {"name": name, "samples": len(indexed[name]["samples"]),
             **{field: indexed[name]["summary"][field] for field in ("median_wall_ns", "min_wall_ns", "max_wall_ns")},
             "backing_storage": indexed[name]["backing_storage"]} for name in series_order]})
    return statistics_rows


def verify_host_outputs_manifest(path, manifest, plotter, count):
    require(type(manifest["schema"]) is int and manifest["schema"] == 1 and manifest["kind"] == "host_outputs",
            "expected host-output schema-1 manifest")
    report_path = repo_path(manifest["source_report"])
    require(report_path.is_relative_to(ROOT / "benchmarks/results/host-outputs"), "host-output report outside results directory")
    check_hash(report_path, manifest["source_report_sha256"])
    check_hash(repo_path(plotter), manifest["plotter_sha256"])
    require(set(manifest["style_helpers_sha256"]) == {"benchmarks/plot_results.py"}, "host-output style helper inventory differs")
    check_hash(repo_path("benchmarks/plot_results.py"), manifest["style_helpers_sha256"]["benchmarks/plot_results.py"])
    report = read_json(report_path)
    require(type(report["schema_version"]) is int and report["schema_version"] == 1 and
            report["kind"] == "host_outputs" and report["complete"] is True and not report.get("error"),
            "expected complete schema-1 host-output report")
    check_hash(repo_path("benchmarks/host_outputs.py"), report["harness_sha256"])
    dependencies = report["dependencies_sha256"]
    require(set(dependencies) == {"benchmarks/profile_workflow.py", "benchmarks/profile_timeline.py",
                                  "benchmarks/profile_resident.py", "benchmarks/benchmark.py"},
            "host-output dependency inventory differs")
    for filename, expected in dependencies.items():
        check_hash(repo_path(filename), expected)
    require(report["benchmark_sha256"] == dependencies["benchmarks/benchmark.py"] and
            report["profile_resident_sha256"] == dependencies["benchmarks/profile_resident.py"],
            "host-output dependency aliases differ")
    proof = {**report, "harness_sha256": dependencies["benchmarks/profile_timeline.py"]}
    check_sources(proof, "timeline")
    # The frozen native-source proof uses only the standard library.
    spec = importlib.util.spec_from_file_location("host_output_source_proof", repo_path("benchmarks/extract_timeline.py"))
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    extractor.source_binding(proof, ROOT)
    measured = {"revision": report["source_revision"], "harness_sha256": report["harness_sha256"], "sha256": report["source_sha256"]}
    require(manifest["measurement_source"] == measured, "host-output measured source differs")
    fields = ("dependencies_sha256", "benchmark_sha256", "profile_resident_sha256", "native_build", "environment", "arguments",
              "gpu_snapshot_before", "gpu_snapshot_after", "methodology", "validation", "warnings")
    require(manifest["report_evidence"] == {field: report[field] for field in fields if field in report},
            "host-output report evidence differs")
    require(isinstance(report["environment"]["gpu_uuid"], str) and re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", report["environment"]["gpu_uuid"]),
        "missing host-output CUDA GPU UUID")
    require(all(isinstance(report[field], dict) for field in ("gpu_snapshot_before", "gpu_snapshot_after", "methodology")),
            "missing host-output measurement context")
    require(report["methodology"] and all(isinstance(value, str) and value for value in report["methodology"].values()),
            "invalid host-output methodology")
    require(set(manifest["figures"]) == {"host-output-latency"}, "host-output figure identity differs")
    figure = manifest["figures"]["host-output-latency"]
    require(figure["sizes"] == [65536, 1048576, 67108864] and figure["statistic"] == "median of individual wall_ns" and
            figure["units"] == "ms", "host-output figure statistic differs")
    require(figure["case_statistics"] == host_output_statistics(report, figure["series_order"]),
            "host-output figure statistics differ from raw samples")
    require(figure["exports"] == manifest["exports"] and set(manifest["exports"]) ==
            {"host-output-latency" + extension for extension in EXTENSIONS}, "host-output figure exports differ")
    return verify_exports(path, list(manifest["exports"].items()), count)


def verify_manifest(name, kind, plotter, count):
    path = FIGURES / name
    manifest = read_json(path)
    if kind == "workflow":
        return verify_workflow_manifest(path, manifest, plotter, count)
    if kind == "host_outputs":
        return verify_host_outputs_manifest(path, manifest, plotter, count)
    if kind == "general":
        report_name = manifest["source"]
        raw_hash, plotter_hash = manifest["source_sha256"], manifest["script_sha256"]
        exports = [(filename, sha) for figure in manifest["figures"].values()
                   for filename, sha in figure["exports_sha256"].items()]
    else:
        report_name = manifest["source_report"]
        raw_hash, plotter_hash = manifest["source_report_sha256"], manifest["plotter_sha256"]
        exports = ([(item["file"], item["sha256"]) for item in manifest["exports"]]
                   if kind == "batch" else list(manifest["exports"].items()))
    report_path = repo_path(report_name)
    require(report_path.is_relative_to(ROOT / "benchmarks/results"),
            f"report must live under benchmarks/results: {report_name}")
    check_hash(report_path, raw_hash)
    check_hash(repo_path(plotter), plotter_hash)
    report = read_json(report_path)
    require(report.get("complete") is True and type(report.get("schema_version")) is int
            and report["schema_version"] == 1,
            f"{report_name}: expected a complete schema-1 benchmark report")
    check_sources(report, kind)
    if kind == "timeline":
        for filename, field in (("benchmarks/benchmark.py", "benchmark_sha256"),
                                ("benchmarks/profile_resident.py", "profile_resident_sha256"),
                                ("benchmarks/extract_timeline.py", "extractor_sha256"),
                                ("benchmarks/extract_nsight.py", "helper_sha256")):
            check_hash(repo_path(filename), report[field])
        check_hash(repo_path("benchmarks/plot_results.py"), manifest["style_helper_sha256"])
        style_helpers = manifest["style_helpers_sha256"]
        require(set(style_helpers) == {"benchmarks/plot_results.py", "benchmarks/plot_nsight.py"},
                "missing timeline style helper identities")
        for filename, expected in style_helpers.items():
            check_hash(repo_path(filename), expected)
        require(manifest["warnings"] == report["warnings"], "Timeline diagnostic warnings differ")
        for field in ("native_build", "extractor_sha256", "artifact_sha256", "time_origin"):
            require(manifest[field] == report[field], f"Timeline {field} differs from report")
        require(report["fixture_oracle_verified"] is True, "Timeline fixture oracle was skipped")
        artifacts = report["artifact_sha256"]
        require(isinstance(artifacts, dict) and bool(artifacts), "missing timeline artifacts")
        required = {"nsys_report", "telemetry", "command", "log", "export_log",
                    "worker_log", "worker_result", "capture_receipt"}
        require(required <= set(report["artifacts"]), "missing timeline capture evidence")
        for key in required:
            item = report["artifacts"][key]
            require(artifacts.get(item["path"]) == item["sha256"],
                    f"Timeline {key} missing from artifact inventory")
        for filename, expected in artifacts.items():
            artifact = repo_path(filename)
            require(artifact.is_relative_to(ROOT / "benchmarks/results/timeline/captures"),
                    f"timeline artifact outside capture directory: {filename}")
            check_hash(artifact, expected)
        require(any(filename.endswith(".nsys-rep") for filename in artifacts),
                "missing raw timeline Nsight report")
        require(set(report["views"]) == {"whole_process", "measured_loop"},
                "timeline requires whole-process and measured-loop views")
    if kind == "nsight":
        check_hash(repo_path("benchmarks/plot_results.py"), manifest["style_helper_sha256"])
        require(manifest["warnings"] == {case["name"]: case["warnings"]
                                        for case in report["cases"] if case["warnings"]},
                "Nsight diagnostic warnings differ from the normalized report")
        check_hash(repo_path("benchmarks/benchmark.py"), report["benchmark_sha256"])
        check_hash(repo_path("benchmarks/extract_nsight.py"), report["extractor_sha256"])
        artifacts = report["artifact_sha256"]
        require(isinstance(artifacts, dict) and bool(artifacts),
                "missing Nsight trace artifacts")
        for filename, expected in artifacts.items():
            artifact = repo_path(filename)
            require(artifact.is_relative_to(ROOT / "benchmarks/results/nsight/captures"),
                    f"Nsight artifact outside capture directory: {filename}")
            check_hash(artifact, expected)
        workloads = {"zeros", "text", "uint32", "float32", "random"}
        expected_cases = {(workload, size) for workload in workloads
                          for size in (65536, 1048576, 67108864)} | {("uint32", 131072)}
        cases = report["cases"]
        require(len(cases) == 16 and
                {(case["workload"], case["output_bytes"]) for case in cases} == expected_cases,
                "Nsight capture matrix is incomplete or duplicated")
        for case in cases:
            prefix = f"benchmarks/results/nsight/captures/{case['workload']}-{case['output_bytes']}"
            required = {prefix + suffix for suffix in
                        (".nsys-rep", ".log", "-profile.json", "-command.json", "-capture.json", "-export.log")}
            require(required <= set(artifacts), f"missing raw Nsight artifacts: {prefix}")
    if kind == "batch":
        measured = report["source"]
        require(manifest["source_revision"] == report["source_revision"] and
                manifest["harness_sha256"] == report["harness_sha256"],
                f"{name}: stale measured revision or harness identity")
    else:
        measured = {"revision": report["source_revision"],
                    "harness_sha256": report["harness_sha256"],
                    "sha256": (report["environment"]["codec_sha256"] if kind == "general"
                               else report["source_sha256"])}
    require(manifest["measurement_source"] == measured,
            f"{name}: measured source metadata differs from the raw report")
    return verify_exports(path, exports, count)


def main():
    try:
        # Exactly one active hardware snapshot. Old figures belong outside FIGURES.
        selection_path = FIGURES / "publication.json"
        selection = read_json(selection_path) if selection_path.is_file() else None
        gpu_slug = "rtx4090"
        if selection is not None:
            require(set(selection) == {"schema_version", "kind", "gpu_slug", "source_revision"}
                    and selection["schema_version"] == 1 and selection["kind"] == "active_publication",
                    "invalid active publication selection")
            gpu_slug = selection["gpu_slug"]
            require(gpu_slug in {"rtx3090", "rtx4090"}, "unsupported active publication hardware")
            require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", selection["source_revision"]),
                    "invalid active publication revision")
        specifications = tuple((name.replace("rtx4090", gpu_slug), kind, plotter, count)
                               for name, kind, plotter, count in MANIFESTS)
        expected = set()
        for specification in specifications:
            if selection is not None:
                manifest = read_json(FIGURES / specification[0])
                if specification[1] == "batch":
                    revisions = {manifest["source_revision"]}
                elif specification[1] == "workflow":
                    revisions = {value["revision"] for value in manifest["measurement_sources"].values()}
                else:
                    revisions = {manifest["measurement_source"]["revision"]}
                require(revisions == {selection["source_revision"]}, "active publication revision mismatch")
            exports = verify_manifest(*specification)
            require(not expected & exports, "exports occur in more than one manifest")
            expected.update(exports)
        actual = {path.resolve() for path in FIGURES.rglob("*")
                  if path.is_file() and path.suffix.lower() in EXTENSIONS}
        require(actual == expected,
                "unmanifested or missing exports: " + ", ".join(
                    str(path.relative_to(ROOT)) for path in sorted(actual ^ expected)))
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        print(f"Benchmark figure verification failed: {error}", file=sys.stderr)
        return 1
    print(f"Verified {len(MANIFESTS)} manifests, measured source identities and {len(expected)} figure exports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Host-output publication checks; synthetic timings never leave tmp_path."""

import builtins
import copy
import json
import statistics

import pytest

from test_verify_workflow_figures import ROOT, publication, sha, verifier


@pytest.fixture
def host_publication(publication):
    root, _, workflow_reports, _, _ = publication
    for name in ("host_outputs.py", "plot_host_outputs.py"):
        (root / "benchmarks" / name).write_bytes((ROOT / "benchmarks" / name).read_bytes())
    base = workflow_reports["compress"]
    dependencies = copy.deepcopy(base["dependencies_sha256"])
    dependencies["benchmarks/profile_workflow.py"] = sha((root / "benchmarks/profile_workflow.py").read_bytes())
    report = {field: copy.deepcopy(base[field]) for field in ("source_revision", "source_sha256", "native_build")}
    report.update(schema_version=1, kind="host_outputs", complete=True,
        harness_sha256=sha((root / "benchmarks/host_outputs.py").read_bytes()), dependencies_sha256=dependencies,
        benchmark_sha256=dependencies["benchmarks/benchmark.py"], profile_resident_sha256=dependencies["benchmarks/profile_resident.py"],
        environment={"gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef"},
        gpu_snapshot_before={"errors": []}, gpu_snapshot_after={"errors": []},
        methodology={"scope": "synthetic full synchronous host-input contracts"},
        arguments={"sizes": [65536, 1048576, 67108864], "workload": "float32", "samples": 2, "warmups": 1,
                   "seed": 20261008, "order_seed": 20261009}, cases=[])
    names = [f"{backend}_{contract}" for contract in ("ndarray", "memoryview", "bytes") for backend in ("cpu", "cuda")]
    for size in report["arguments"]["sizes"]:
        case = {"workload": "float32", "output_bytes": size, "input_bytes": 100,
                "fixture": {"seed": 20261008, "stdlib_level": 6, "stdlib_stream_bytes": 100,
                            "raw_sha256": sha(b"test raw"), "stdlib_stream_sha256": sha(b"test stream")},
                "order_seed": 20261009 + size, "order": [], "series": []}
        indexed = {}
        for name in names:
            backend, contract = name.split("_")
            item = {"name": name, "backend": backend, "contract": contract,
                    "backing_storage": "fresh test " + backend + " storage", "samples": [], "warmups": []}
            indexed[name] = item
            case["series"].append(item)
        for warmup, count in ((True, 1), (False, 2)):
            for round_index in range(count):
                for name in names if round_index == 0 else reversed(names):
                    index = len(case["order"])
                    case["order"].append({"warmup": warmup, "round": round_index, "series": name})
                    wall = 100 + index
                    row = {"complete": True, "byte_exact": True, "readonly": True, "warmup": warmup, "round": round_index,
                           "order_index": index, "output_bytes": size, "output_type": indexed[name]["contract"],
                           "start_ns": index * 1000, "end_ns": index * 1000 + wall, "wall_ns": wall,
                           "thread_cpu_ns": 10, "minor_faults": 0, "major_faults": 0}
                    if indexed[name]["contract"] == "ndarray":
                        row.update(dtype="uint8", shape=[size])
                    elif indexed[name]["contract"] == "memoryview":
                        row.update(format="B", shape=[size],
                                   backing_type="bytes" if indexed[name]["backend"] == "cpu" else "ndarray")
                    indexed[name]["warmups" if warmup else "samples"].append(row)
        for item in case["series"]:
            walls = [row["wall_ns"] for row in item["samples"]]
            item["summary"] = {"median_wall_ns": statistics.median(walls), "min_wall_ns": min(walls), "max_wall_ns": max(walls),
                               "median_thread_cpu_ns": 10, "median_minor_faults": 0, "median_major_faults": 0}
        report["cases"].append(case)
    exports = {}
    for extension in verifier.EXTENSIONS:
        filename = "host-output-latency" + extension
        path = root / "benchmarks/figures/host-outputs" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test figure")
        exports[filename] = sha(path.read_bytes())
    figure = {"exports": exports, "sizes": report["arguments"]["sizes"], "series_order": names,
              "statistic": "median of individual wall_ns", "units": "ms", "case_statistics": []}
    for case in report["cases"]:
        figure["case_statistics"].append({"output_bytes": case["output_bytes"], "series": [
            {"name": item["name"], "samples": 2, "backing_storage": item["backing_storage"],
             **{field: item["summary"][field] for field in ("median_wall_ns", "min_wall_ns", "max_wall_ns")}}
            for item in case["series"]]})
    manifest = {"schema": 1, "kind": "host_outputs", "source_report": "benchmarks/results/host-outputs/rtx4090-float32.json",
                "plotter_sha256": sha((root / "benchmarks/plot_host_outputs.py").read_bytes()),
                "style_helpers_sha256": {"benchmarks/plot_results.py": sha((root / "benchmarks/plot_results.py").read_bytes())},
                "exports": exports, "figures": {"host-output-latency": figure}}

    def save():
        path = root / manifest["source_report"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, sort_keys=True))
        manifest["source_report_sha256"] = sha(path.read_bytes())
        manifest["measurement_source"] = {"revision": report["source_revision"], "harness_sha256": report["harness_sha256"], "sha256": report["source_sha256"]}
        fields = ("dependencies_sha256", "benchmark_sha256", "profile_resident_sha256", "native_build", "environment", "arguments",
                  "gpu_snapshot_before", "gpu_snapshot_after", "methodology", "validation", "warnings")
        manifest["report_evidence"] = copy.deepcopy({field: report[field] for field in fields if field in report})

    def verify():
        path = root / "benchmarks/figures/host-outputs/rtx4090-manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True))
        return verifier.verify_manifest("host-outputs/rtx4090-manifest.json", "host_outputs", "benchmarks/plot_host_outputs.py", 3)

    save()
    return root, report, manifest, save, verify


def test_host_output_guard_uses_only_standard_library(host_publication, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"jax", "numpy", "matplotlib"}:
            raise AssertionError("host-output guard imported " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert len(host_publication[-1]()) == 3


@pytest.mark.parametrize("filename", ["src/cuda_zlib/_codec.py", "benchmarks/host_outputs.py",
    "benchmarks/profile_workflow.py", "benchmarks/plot_host_outputs.py", "benchmarks/plot_results.py"])
def test_host_output_source_or_style_changes_are_rejected(host_publication, filename):
    root, _, _, _, verify = host_publication
    with (root / filename).open("ab") as output:
        output.write(b"\n# changed\n")
    with pytest.raises(ValueError, match="stale"):
        verify()


@pytest.mark.parametrize("mutation", ["missing-series", "warmup-oracle", "order", "contract", "clock", "summary",
                                     "native-build", "gpu-identity", "figure-statistic", "evidence", "view-backing"])
def test_host_output_validation_and_provenance_tampering_is_rejected(host_publication, mutation):
    _, report, manifest, save, verify = host_publication
    case = report["cases"][0]
    row = case["series"][0]["samples"][0]
    if mutation == "missing-series":
        case["series"].pop()
    elif mutation == "warmup-oracle":
        case["series"][0]["warmups"][0]["byte_exact"] = False
    elif mutation == "order":
        row["order_index"] = 0
    elif mutation == "contract":
        row["dtype"] = "uint32"
    elif mutation == "clock":
        row["wall_ns"] += 1
    elif mutation == "view-backing":
        next(item for item in case["series"] if item["name"] == "cuda_memoryview")["samples"][0]["backing_type"] = "bytes"
    elif mutation == "summary":
        case["series"][0]["summary"]["median_wall_ns"] += 1
    elif mutation == "native-build":
        report["native_build"]["cache_key"] = "b" * 64
    elif mutation == "gpu-identity":
        report["environment"]["gpu_uuid"] = "unknown"
    elif mutation == "figure-statistic":
        manifest["figures"]["host-output-latency"]["case_statistics"][0]["series"][0]["median_wall_ns"] += 1
    save()
    if mutation == "evidence":
        manifest["report_evidence"]["methodology"]["scope"] = "stale scope"
    with pytest.raises(ValueError):
        verify()

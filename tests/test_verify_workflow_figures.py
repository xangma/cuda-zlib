# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""CPU-only publication identity checks using temporary synthetic artifacts."""

import ast
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_workflow_figures", ROOT / "benchmarks/verify_figures.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def publication(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "FIGURES", tmp_path / "benchmarks/figures")

    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(value, sort_keys=True).encode() if isinstance(value, dict) else value
        path.write_bytes(content)
        return sha(content)

    source = {}
    for path in (ROOT / "src/cuda_zlib").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".cu", ".cuh"}:
            relative = path.relative_to(ROOT).as_posix()
            source[path.relative_to(ROOT / "src/cuda_zlib").as_posix()] = write(relative, path.read_bytes())
    helpers = ("profile_workflow.py", "profile_timeline.py", "profile_resident.py", "benchmark.py",
               "extract_workflow.py", "extract_timeline.py", "extract_nsight.py", "run_workflow.py",
               "plot_workflow.py", "plot_results.py", "plot_nsight.py", "plot_timeline.py")
    helper_hashes = {"benchmarks/" + name: write("benchmarks/" + name, (ROOT / "benchmarks" / name).read_bytes())
                     for name in helpers}
    generated = {}
    for output, module in (("encoder.cuh", "_encode_kernels.py"), ("decoder.cuh", "_decode_kernels.py"),
                           ("postprocess.cuh", "_postprocess.py")):
        tree = ast.parse((tmp_path / "src/cuda_zlib" / module).read_text())
        literal = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "CUDA_SOURCE" for target in node.targets))
        generated[output] = sha(ast.literal_eval(literal).encode())
    generated.update({name: source["native/" + name] for name in ("codec_ffi.cu", "batch_encode.cuh", "batch_decode.cuh")})
    identity = {"architecture": "sm_89", "sources": generated}
    native = {"identity": identity, "cache_key": sha(json.dumps(identity, sort_keys=True).encode()),
              "library_sha256": sha(b"test library"), "build_sha256": sha(b"test build"),
              "ffi_targets": [f"cuda_zlib_{action}_sm_89" for action in
                              ("compress", "decompress", "compress_batch", "decompress_batch")]}
    dependencies = {name: helper_hashes[name] for name in
                    ("benchmarks/profile_timeline.py", "benchmarks/profile_resident.py", "benchmarks/benchmark.py")}
    extractor_dependencies = {name: helper_hashes[name] for name in
                              ("benchmarks/extract_timeline.py", "benchmarks/extract_nsight.py")}
    manifest = {"schema": 1, "kind": "workflow", "operations": ["compress", "decompress"],
                "plotter_sha256": helper_hashes["benchmarks/plot_workflow.py"],
                "source_reports": {}, "measurement_sources": {}, "report_evidence": {},
                "warnings": {}, "time_origins": {}, "phase_semantics": {}, "figures": {}, "exports": {},
                "style_helpers_sha256": {name: helper_hashes[name] for name in
                                         ("benchmarks/plot_results.py", "benchmarks/plot_nsight.py", "benchmarks/plot_timeline.py")}}
    reports = {}
    evidence_fields = ("native_build", "extractor_sha256", "artifact_sha256", "dependencies_sha256", "extractor_dependencies_sha256")

    def save(operation):
        report = reports[operation]
        filename = f"benchmarks/results/workflow/rtx4090-{operation}-float32.json"
        manifest["source_reports"][operation] = {"path": filename, "sha256": write(filename, report)}
        manifest["measurement_sources"][operation] = {"revision": report["source_revision"], "harness_sha256": report["harness_sha256"], "sha256": copy.deepcopy(report["source_sha256"])}
        manifest["report_evidence"][operation] = {field: copy.deepcopy(report[field]) for field in evidence_fields}

    for operation in manifest["operations"]:
        common = {"source_revision": "a" * 40, "source_sha256": source,
                  "harness_sha256": helper_hashes["benchmarks/profile_workflow.py"],
                  "dependencies_sha256": dependencies, "native_build": native,
                  "metadata_layout": ["encoded_length", "status"] if operation == "compress" else ["status", "reserved_zero"],
                  "fixture": {"raw_sha256": sha(b"raw fixture")}, "environment": {"backend": "test fixture"}}
        worker = {**{key: value for key, value in common.items() if key != "metadata_layout"},
                  "operation": operation, "complete": True, "pid": 17}
        directory = f"benchmarks/results/workflow/captures/{operation}/profiled"
        artifacts = {}
        for key, filename, content in (("nsys_report", "capture.nsys-rep", b"synthetic trace"),
                                       ("command", "command.json", b"[]"), ("log", "capture.log", b"capture"),
                                       ("export_log", "export.log", b"export"), ("worker_log", "worker.log", b"worker"),
                                       ("worker_result", "worker.json", worker)):
            name = directory + "/" + filename
            artifacts[key] = {"path": name, "sha256": write(name, content)}
        telemetry = {**common, "operation": operation, "arguments": {"operation": operation}, "complete": True,
                     "worker": {"pid": 17, "exit_code": 0, "result": worker,
                                "result_sha256": artifacts["worker_result"]["sha256"],
                                "log_sha256": artifacts["worker_log"]["sha256"]}}
        telemetry_name = directory + "/telemetry.json"
        artifacts["telemetry"] = {"path": telemetry_name, "sha256": write(telemetry_name, telemetry)}
        receipt_artifacts = {key: {"path": Path(item["path"]).name, "published_path": item["path"], "sha256": item["sha256"]}
                             for key, item in artifacts.items()}
        private = {"filename": "capture.sqlite", "sha256": sha(b"private SQLite"), "private": True}
        artifacts["sqlite"] = private
        receipt_artifacts["sqlite"] = {"path": private["filename"], "sha256": private["sha256"], "private": True}
        receipt = {"source_revision": common["source_revision"], "runner_sha256": helper_hashes["benchmarks/run_workflow.py"],
                   "capture_started_utc": "2026-10-09T10:00:00+00:00", "toolchain": {"nsys": "test"}, "artifacts": receipt_artifacts}
        receipt_name = directory + "/capture.json"
        artifacts["capture_receipt"] = {"path": receipt_name, "sha256": write(receipt_name, receipt)}
        reports[operation] = {**common, "schema_version": 1, "kind": "workflow_timeline", "operation": operation,
            "complete": True, "profiled": True, "fixture_oracle_verified": True,
            "extractor_sha256": helper_hashes["benchmarks/extract_workflow.py"],
            "extractor_dependencies_sha256": extractor_dependencies,
            "runner_sha256": helper_hashes["benchmarks/run_workflow.py"], "benchmark_sha256": dependencies["benchmarks/benchmark.py"],
            "capture_started_utc": receipt["capture_started_utc"], "toolchain": receipt["toolchain"], "command": [],
            "worker": {"pid": 17, "exit_code": 0},
            "artifacts": artifacts, "artifact_sha256": {item["path"]: item["sha256"] for key, item in artifacts.items() if key != "sqlite"},
            "warnings": [], "time_origin": {"method": "test origin"}, "phase_semantics": {},
            "views": {"whole_process": {}, "measured_loop": {}}}
        save(operation)
        for field, target in (("warnings", "warnings"), ("time_origins", "time_origin"), ("phase_semantics", "phase_semantics")):
            manifest[field][operation] = reports[operation][target]
        for suffix, view in (("whole-process", "whole_process"), ("warmed-loop", "measured_loop")):
            name = operation + "-" + suffix
            exports = {name + extension: write("benchmarks/figures/workflow/" + name + extension, b"synthetic figure") for extension in verifier.EXTENSIONS}
            manifest["figures"][name] = {"operation": operation, "view": view, "exports": exports}
            manifest["exports"].update(exports)

    def verify():
        write("benchmarks/figures/workflow/rtx4090-manifest.json", manifest)
        return verifier.verify_manifest("workflow/rtx4090-manifest.json", "workflow", "benchmarks/plot_workflow.py", 12)

    return tmp_path, manifest, reports, save, verify


def test_complete_workflow_publication_needs_no_gpu_or_plotting_imports(publication, monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"jax", "numpy", "matplotlib"}:
            raise AssertionError("identity verification imported " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert len(publication[-1]()) == 12


@pytest.mark.parametrize("filename", ["src/cuda_zlib/_codec.py", "benchmarks/profile_workflow.py",
    "benchmarks/run_workflow.py", "benchmarks/benchmark.py", "benchmarks/profile_timeline.py",
    "benchmarks/extract_workflow.py", "benchmarks/extract_nsight.py", "benchmarks/plot_workflow.py",
    "benchmarks/plot_timeline.py"])
def test_changed_measured_or_rendering_dependency_is_rejected(publication, filename):
    root, _, _, _, verify = publication
    with (root / filename).open("ab") as output:
        output.write(b"\n# changed after capture\n")
    with pytest.raises(ValueError, match="stale"):
        verify()


@pytest.mark.parametrize("mutation", ["single-operation", "swapped-report", "native-build", "raw-trace", "command",
                                     "receipt", "worker", "published-sqlite", "missing-runner", "figure-operation"])
def test_workflow_provenance_tampering_is_rejected(publication, mutation):
    root, manifest, reports, save, verify = publication
    report = reports["compress"]
    if mutation == "single-operation":
        manifest["operations"] = ["compress"]
    elif mutation == "swapped-report":
        manifest["source_reports"]["compress"] = manifest["source_reports"]["decompress"]
    elif mutation == "native-build":
        report["native_build"]["cache_key"] = "b" * 64
        save("compress")
    elif mutation == "raw-trace":
        (root / report["artifacts"]["nsys_report"]["path"]).write_bytes(b"different trace")
    elif mutation == "command":
        report["command"] = ["different", "capture"]
        save("compress")
    elif mutation in {"receipt", "worker"}:
        key = "capture_receipt" if mutation == "receipt" else "worker_result"
        path = root / report["artifacts"][key]["path"]
        content = json.loads(path.read_bytes())
        content["source_revision"] = "b" * 40
        path.write_text(json.dumps(content, sort_keys=True))
        report["artifacts"][key]["sha256"] = sha(path.read_bytes())
        report["artifact_sha256"][path.relative_to(root).as_posix()] = sha(path.read_bytes())
        if mutation == "worker":
            receipt_path = root / report["artifacts"]["capture_receipt"]["path"]
            receipt = json.loads(receipt_path.read_bytes())
            receipt["artifacts"]["worker_result"]["sha256"] = report["artifacts"][key]["sha256"]
            receipt_path.write_text(json.dumps(receipt, sort_keys=True))
            report["artifacts"]["capture_receipt"]["sha256"] = sha(receipt_path.read_bytes())
            report["artifact_sha256"][receipt_path.relative_to(root).as_posix()] = sha(receipt_path.read_bytes())
        save("compress")
    elif mutation == "published-sqlite":
        report["artifacts"]["sqlite"]["path"] = "benchmarks/results/workflow/captures/compress/private.sqlite"
        save("compress")
    elif mutation == "missing-runner":
        del report["runner_sha256"]
        save("compress")
    elif mutation == "figure-operation":
        manifest["figures"]["compress-whole-process"]["operation"] = "decompress"
    with pytest.raises((ValueError, KeyError)):
        verify()

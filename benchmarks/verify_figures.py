#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Check published figure and measurement identities without JAX or Matplotlib.

Works in a checkout or source export. Numerical report validation belongs to
the plotters; this check detects stale artifacts.
"""

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "benchmarks/figures"
EXTENSIONS = {".png", ".svg", ".pdf"}
MANIFESTS = (
    ("manifest.json", "general", "benchmarks/plot_results.py", 15),
    ("small-batch/rtx4090-manifest.json", "batch", "benchmarks/plot_small_batch.py", 18),
    ("resident-checked-manifest.json", "resident", "benchmarks/plot_resident.py", 3),
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
        harness = ("benchmarks/benchmark.py" if kind == "general"
                   else "benchmarks/profile_resident.py")
        required = package | {harness}
    require(recorded.get(harness, report["harness_sha256"]) == report["harness_sha256"],
            f"conflicting harness identities for {harness}")
    recorded[harness] = report["harness_sha256"]
    require(set(recorded) == required,
            f"stale source inventory: missing {sorted(required - set(recorded))}; "
            f"unexpected {sorted(set(recorded) - required)}")
    for name, expected in recorded.items():
        check_hash(repo_path(name), expected)


def verify_manifest(name, kind, plotter, count):
    path = FIGURES / name
    manifest = read_json(path)
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


def main():
    try:
        expected = set()
        for specification in MANIFESTS:
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
    print(f"Verified 3 manifests, measured source identities and {len(expected)} figure exports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Validate four ABBA profiles and concatenate all wall samples per version."""

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path

from plot_optimisation import OPERATIONS, ROOT, compare, read_profile

RUN_ORDER = ("baseline-a", "candidate-a", "candidate-b", "baseline-b")
PATH_ARGUMENTS = {"output", "save_streams", "streams_from"}
IDENTITY_FIELDS = ("input_sha256", "encoded_sha256", "encoded_bytes",
                   "frozen_stream_sha256", "stdlib_stream_sha256")


def merge_profiles(paths, outputs):
    """Return two validated aggregates without changing the four raw reports."""
    if len({p.resolve() for p in paths.values()}) != len(RUN_ORDER):
        raise ValueError("Expected four distinct raw profile files")
    if len({p.name for p in paths.values()}) != len(RUN_ORDER):
        raise ValueError("Raw profile basenames must identify each run uniquely")
    loaded = {run_id: read_profile(paths[run_id]) for run_id in RUN_ORDER}
    reference = loaded[RUN_ORDER[0]]
    reference_args = {k: v for k, v in reference[0]["arguments"].items()
                      if k not in PATH_ARGUMENTS}
    for run_id, profile in loaded.items():
        if "aggregation" in profile[0]:
            raise ValueError(f"Expected raw measurements, not an aggregate: {run_id}")
        # Reuse the plot validator for matching input/stream/environment evidence.
        compare(reference, profile)
        measured_args = {k: v for k, v in profile[0]["arguments"].items()
                         if k not in PATH_ARGUMENTS}
        if measured_args != reference_args:
            raise ValueError(f"Different measured arguments: {run_id}")
        for identity, case in profile[1].items():
            for field in IDENTITY_FIELDS:
                if case[field] != reference[1][identity][field]:
                    raise ValueError(f"Different {field}: {run_id} {identity}")
    for version in ("baseline", "candidate"):
        if loaded[f"{version}-a"][0]["source_sha256"] != loaded[f"{version}-b"][0]["source_sha256"]:
            raise ValueError(f"Codec source changed between {version} runs")

    run_records = []
    for position, run_id in enumerate(RUN_ORDER, 1):
        report, _, _, sha256 = loaded[run_id]
        run_records.append({
            "position": position, "run_id": run_id, "version": run_id.split("-")[0],
            "profile": paths[run_id].name, "profile_sha256": sha256,
            "source_sha256": report["source_sha256"], "arguments": report["arguments"],
            "environment": report["environment"],
            "samples_per_measurement": report["arguments"]["samples"],
        })
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    aggregates = {}
    for version in ("baseline", "candidate"):
        included = [run_id for run_id in RUN_ORDER if run_id.startswith(version + "-")]
        first = included[0]
        aggregate = copy.deepcopy(loaded[first][0])
        sample_slices, count = [], 0
        for run_id in included:
            samples = loaded[run_id][0]["arguments"]["samples"]
            sample_slices.append({"run_id": run_id, "start": count, "stop": count + samples})
            count += samples
        aggregate["arguments"].update(samples=count, output=outputs[version].name,
                                       save_streams=None, streams_from=None)
        aggregate["environment"]["gpu_after"] = loaded[included[-1]][0]["environment"]["gpu_after"]
        aggregate["aggregation"] = {
            "schema_version": 1, "version": version,
            "method": "Concatenate every raw wall sample in run order; recompute summaries from the combined samples.",
            "run_order": list(RUN_ORDER), "runs": copy.deepcopy(run_records),
            "run_order_basis": "Run identities supplied to the CLI; raw profiles do not record execution timestamps.",
            "included_run_ids": included, "wall_sample_slices": sample_slices,
            "script_sha256": script_sha256,
            "kernel_diagnostics": {
                "copied_from_run_id": first, "profile_sha256": loaded[first][3],
                "scope": "One separately instrumented call per case/operation from the first included run; not merged or summed.",
            },
        }
        for case in aggregate["cases"]:
            identity = (case["workload"], case["input_bytes"])
            for operation in OPERATIONS:
                # Leave the first run's separate one-call diagnostics intact.
                seconds = [value for run_id in included
                           for value in loaded[run_id][1][identity]["timings"][operation]["wall"]["seconds"]]
                median = statistics.median(seconds)
                case["timings"][operation]["wall"] = {
                    "seconds": seconds, "median_seconds": median,
                    "min_seconds": min(seconds), "max_seconds": max(seconds),
                    "mib_per_second": case["input_bytes"] / 2**20 / median,
                }
        aggregates[version] = aggregate
    return aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for run_id in RUN_ORDER:
        parser.add_argument(f"--{run_id}", type=Path,
                            default=ROOT / "results" / f"rerun-{run_id}.json")
    parser.add_argument("--output-before", type=Path, default=ROOT / "results" / "rerun-before.json")
    parser.add_argument("--output-after", type=Path, default=ROOT / "results" / "rerun-after.json")
    args = parser.parse_args()
    paths = {run_id: getattr(args, run_id.replace("-", "_")) for run_id in RUN_ORDER}
    outputs = {"baseline": args.output_before, "candidate": args.output_after}
    if len({p.resolve() for p in outputs.values()}) != 2:
        parser.error("Aggregate outputs must be distinct")
    if {p.resolve() for p in paths.values()} & {p.resolve() for p in outputs.values()}:
        parser.error("Aggregate outputs must not overwrite raw profiles")
    aggregates = merge_profiles(paths, outputs)
    for version, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregates[version], indent=2) + "\n")
    # Verify exactly the files that the plotting CLI will subsequently consume.
    result = compare(read_profile(outputs["baseline"]), read_profile(outputs["candidate"]))
    print(f"Validated four profiles in supplied ABBA order and {result['validated_cases']} cases; "
          f"merged all {result['samples_per_measurement']} wall samples per version/operation. "
          "Kernel diagnostics identify each version's first run only.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Export measurement data without private captures or operational metadata.

Keep the original report outside the public checkout. This export preserves
timings, fixture/source/build hashes and diagnostic warnings. Its private report
digest records provenance; it is not independent verification of the capture.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

PRIVATE_KEYS = {
    "cwd", "source_root", "library", "build", "log_path", "result_path",
    "path", "capture_receipt", "export_log", "log", "nsys_report",
    "raw_profile", "sqlite", "create_time", "queries", "stdout", "stderr",
    "exportorigin", "exporttime", "hostname", "host_name", "username", "user",
}
ARTIFACT_KEYS = {"artifact_sha256", "artifacts"}
ANONYMIZED_ID_KEYS = {"pid", "ppid", "pids", "captured_pid", "captured_processes",
                      "globalpid", "global_pid", "globaltid", "global_tid", "tid",
                      "thread_id", "process_id", "process_ids", "process_identities",
                      "collector_pid", "owned_pid"}
PRIVATE_ID_KEYS = {"uuid", "gpu_uuid", "device_uuid", "hostname", "host_name",
                   "username", "user"}
PRIVATE_PATH = re.compile(r"/(?:Users|home|dev/shm|tmp)/[^\s\"'<>;,)\]}]+")
DEVICE_UUID = re.compile(r"(?:GPU-)?[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)


def public_value(value):
    aliases = {}

    def normalize(item):
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                lowered = key.lower()
                if lowered in PRIVATE_KEYS:
                    continue
                if lowered in ARTIFACT_KEYS:
                    result[key] = {}
                elif lowered == "command":
                    result[key] = []
                elif lowered in PRIVATE_ID_KEYS:
                    result[key] = "redacted"
                elif lowered in ANONYMIZED_ID_KEYS:
                    if lowered in {"captured_processes", "process_identities"}:
                        result[key] = normalize(child)
                    elif lowered == "owned_pid" and isinstance(child, bool):
                        result[key] = child
                    elif isinstance(child, list):
                        result[key] = [anonymous_id(identifier) for identifier in child]
                    elif isinstance(child, dict):
                        result[key] = {
                            anonymous_id(identifier): normalize(details)
                            for identifier, details in child.items()
                        }
                    elif child is None:
                        result[key] = None
                    else:
                        result[key] = anonymous_id(child)
                elif lowered.startswith("export_") and any(
                    word in lowered for word in ("host", "user", "path", "time_local", "time_utc")
                ):
                    continue
                else:
                    result[key] = normalize(child)
            return result
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, str):
            item = PRIVATE_PATH.sub("<private-path>", item)
            item = DEVICE_UUID.sub("<device-id>", item)
            item = re.sub(r"\broni1\b", "<capture-host>", item)
            return re.sub(r"\bxangma\b", "<user>", item)
        return item

    def anonymous_id(value):
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            return value
        if value not in aliases:
            aliases[value] = len(aliases) + 1
        return aliases[value]

    return normalize(value)


def export_report(raw):
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get("complete") is not True:
        raise ValueError("expected a complete measurement report")
    if "publication" in report:
        raise ValueError("export the original private report, not an existing public export")
    public = public_value(report)
    public["publication"] = {
        "schema_version": 1,
        "kind": "portable_measurement",
        "private_evidence_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("retain the private original; output must be a separate file")
    public = export_report(args.input.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    print(f"Exported portable measurement: {args.output.name}")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""CPU tests for timeline ownership, null metrics, phases and clock bounds."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("profile_timeline", Path(__file__).resolve().parents[1] /
                                             "benchmarks/profile_timeline.py")
timeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timeline)


def event(event_id, name, kind, timestamp, parent=None, success=True):
    return {"id": event_id, "name": name, "event": kind, "timestamp_ns": timestamp,
            "nvtx_before_ns": timestamp - 1, "nvtx_after_ns": timestamp + 1,
            "parent_id": parent, "pid": 17, "iteration": None, "iteration_id": None,
            "warmup": None, "nvtx_label": "nvtx:" + event_id, "success": success}


def test_phases_preserve_nested_ids_pid_and_brackets():
    events = [event("outer", "measured_loop", "start", 10),
              event("decode1", "decode", "start", 20, "outer"),
              event("decode1", "decode", "end", 30, "outer"),
              event("outer", "measured_loop", "end", 40)]
    rows = timeline.phase_intervals(events)
    assert [row["id"] for row in rows] == ["outer", "decode1"]
    assert rows[1]["pid"] == 17 and rows[1]["parent_id"] == "outer"
    assert rows[1]["start_ns"] == 20 and rows[1]["end_ns"] == 30
    assert rows[1]["nvtx_start_bracket_ns"] == [19, 21]


@pytest.mark.parametrize("events,match", [
    ([event("a", "decode", "start", 10)], "unfinished"),
    ([event("a", "decode", "end", 10)], "without start"),
    ([event("a", "decode", "start", 10), event("a", "decode", "start", 11)], "duplicate"),
    ([event("a", "decode", "start", 20), event("a", "decode", "end", 10)], "reversed"),
    ([event("a", "decode", "start", 10), event("a", "upload", "end", 20)], "identity differs"),
    ([event("p", "warmup", "start", 20), event("a", "decode", "start", 10, "p"),
      event("a", "decode", "end", 30, "p"), event("p", "warmup", "end", 40)], "outside parent"),
])
def test_malformed_stage_pairs_are_rejected(events, match):
    with pytest.raises(ValueError, match=match):
        timeline.phase_intervals(events)


def test_parser_ignores_logs_but_rejects_other_process_events():
    assert timeline.parse_stage_line("startup warning", 17) is None
    line = timeline.EVENT_PREFIX + json.dumps(event("a", "decode", "start", 10))
    assert timeline.parse_stage_line(line, 17)["pid"] == 17
    with pytest.raises(ValueError, match="another PID"):
        timeline.parse_stage_line(line, 18)


class FakeNVML:
    NVML_VALUE_NOT_AVAILABLE = -1

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=1000, used=600, free=400)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=10, memory=20)

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [SimpleNamespace(pid=17, usedGpuMemory=100), SimpleNamespace(pid=99, usedGpuMemory=400)]


def test_whole_device_metrics_keep_foreign_processes_and_scope_owned_memory():
    row = timeline.sample_gpu(FakeNVML(), object(), {17})
    assert row["used_memory_bytes"] == 600
    assert row["owned_compute_memory_bytes"] == 100
    assert row["compute_processes"] == [{"pid": 17, "used_memory_bytes": 100, "owned": True},
                                        {"pid": 99, "used_memory_bytes": 400, "owned": False}]
    for query in row["queries"].values():
        assert query["start_ns"] <= query["end_ns"] and query["error"] is None


def test_missing_nvml_queries_remain_null_with_errors():
    class Missing(FakeNVML):
        def nvmlDeviceGetMemoryInfo(self, handle):
            raise RuntimeError("unavailable")

        def nvmlDeviceGetComputeRunningProcesses(self, handle):
            raise RuntimeError("unsupported")

    row = timeline.sample_gpu(Missing(), object(), {17})
    assert row["used_memory_bytes"] is None and row["compute_processes"] is None
    assert row["owned_compute_memory_bytes"] is None
    assert row["queries"]["memory"]["error"]["message"] == "unavailable"
    assert row["gpu_utilization_percent"] == 10


def test_unavailable_owned_memory_is_not_treated_as_zero():
    class Unknown(FakeNVML):
        def nvmlDeviceGetComputeRunningProcesses(self, handle):
            return [SimpleNamespace(pid=17, usedGpuMemory=-1)]

    row = timeline.sample_gpu(Unknown(), object(), {17})
    assert row["compute_processes"][0]["used_memory_bytes"] is None
    assert row["owned_compute_memory_bytes"] is None


def test_process_samples_preserve_cumulative_cpu_rss_and_lifetime():
    class Process:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive):
            assert recursive is True
            return [Process(18)]

        def create_time(self):
            return 1000 + self.pid

        def cpu_times(self):
            return SimpleNamespace(user=1.25, system=0.5)

        def memory_info(self):
            return SimpleNamespace(rss=4096)

    fake = SimpleNamespace(Process=Process, NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError)
    identities = {}
    rows, query = timeline.sample_processes(fake, 17, identities)
    assert {row["pid"] for row in rows} == {17, 18}
    assert set(identities) == {(17, 1017), (18, 1018)}
    assert all(row["cpu_user_seconds"] == 1.25 and row["cpu_system_seconds"] == 0.5 and
               row["rss_bytes"] == 4096 and row["missing"] is False for row in rows)
    assert query["start_ns"] <= query["end_ns"]


def test_process_tree_race_is_recorded_as_missing_not_zero_cpu():
    def missing(pid):
        raise ProcessLookupError("worker exited")

    fake = SimpleNamespace(Process=missing, NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError)
    rows, query = timeline.sample_processes(fake, 17, {})
    assert rows == [] and query["error"]["type"] == "ProcessLookupError"


def test_partial_process_query_retains_discovered_identity_and_missing_cpu():
    class Process:
        pid = 17

        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive):
            return []

        def create_time(self):
            return 1000

        def cpu_times(self):
            raise PermissionError("CPU query unavailable")

    fake = SimpleNamespace(Process=Process, NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError)
    identities = {}
    rows, _ = timeline.sample_processes(fake, 17, identities)
    assert rows[0]["pid"] == 17 and rows[0]["missing"] is True
    assert rows[0]["cpu_user_seconds"] is None and rows[0]["rss_bytes"] is None
    assert rows[0]["error"]["type"] == "PermissionError"
    assert (17, 1000) in identities

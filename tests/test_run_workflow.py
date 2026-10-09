# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Check capture boundaries and nested timing without CUDA or Nsight."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "run_workflow", Path(__file__).resolve().parents[1] / "benchmarks/run_workflow.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_nsight_application_options_follow_profiler_output(tmp_path):
    args = SimpleNamespace(source_revision="a" * 40, gpu_uuid="GPU-test", size=67108864,
                           workload="float32", seed=20261008, chunk_bytes=32768,
                           warmups=2, iterations=20, device=0, sample_ms=10, timeout=600,
                           nsys="nsys", nvtx_library=Path("/tmp/libnvToolsExt.so"))
    application = runner.harness_command(args, "compress", tmp_path, "process-tree")
    command = runner.nsight_command(args, tmp_path, "cuda,nvtx", application)
    boundary = command.index(str(runner.ROOT / "benchmarks/profile_workflow.py"))
    assert command[:boundary].count("--output") == 1
    assert command[boundary:].count("--output") == 1
    assert "--trace=cuda,nvtx" in command[:boundary]
    assert "--nvtx-domain-exclude=TSL" in command[:boundary]
    assert command[boundary + 1:boundary + 3] == ["--operation", "compress"]
    assert "--nvtx-library" in command[boundary:]
    assert "--nvtx-library" not in application


def test_host_statistics_excludes_warmup_and_keeps_nested_phases_separate():
    def phase(name, start, end, warmup=None):
        return dict(name=name, start_ns=start * 1000000, end_ns=end * 1000000, warmup=warmup)
    telemetry = dict(complete=True, worker=dict(exit_code=0), phase_intervals=[
        phase("measured_loop", 10, 30), phase("codec", 0, 100, True),
        phase("download_check", 10, 30, False), phase("output_download", 10, 15, False),
        phase("host_bytes", 15, 25, False), phase("validate", 25, 30, False)])
    result = runner.host_summary(telemetry)
    assert result["measured_loop_ms"] == 20
    assert "codec" not in result["phases"]
    assert result["phases"]["download_check"]["sum_ms"] == 20
    assert result["phases"]["host_bytes"]["median_ms"] == 10
    telemetry["worker"]["exit_code"] = 1
    with pytest.raises(ValueError, match="incomplete"):
        runner.host_summary(telemetry)


def test_failed_child_records_exact_launch_and_log(tmp_path):
    log = tmp_path / "child.log"
    with pytest.raises(ValueError, match="exited 7"):
        runner.run_logged([runner.sys.executable, "-c", "raise SystemExit(7)"],
                          tmp_path, dict(runner.os.environ), log, 10)
    launch = runner.json.loads(log.with_suffix(".launch.json").read_text())
    assert launch["cwd"] == str(tmp_path) and launch["log"] == str(log)
    assert launch["pid"] > 0 and launch["stop"].endswith(str(launch["pid"]))

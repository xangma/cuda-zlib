"""A successful CLI exit is insufficient evidence of a usable CUDA trace."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('trace_benchmark', Path(__file__).parents[1] / 'benchmarks/trace.py')
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def database(path, tables):
    with sqlite3.connect(path) as connection:
        for name, count in tables.items():
            connection.execute('CREATE TABLE ' + name + ' (id INTEGER)')
            connection.executemany('INSERT INTO ' + name + ' VALUES (?)', ((i,) for i in range(count)))
    return path


def test_trace_rejects_partial_import(tmp_path):
    db = database(tmp_path / 'partial.sqlite', {'CUPTI_ACTIVITY_KIND_RUNTIME': 417})
    with pytest.raises(RuntimeError, match='failed to import'):
        trace.validate_trace(db, 'Status: TargetProfilingFailed\nCannot find string for an exterior index')


def test_trace_requires_kernel_records(tmp_path):
    db = database(tmp_path / 'empty.sqlite', {'CUPTI_ACTIVITY_KIND_RUNTIME': 417})
    with pytest.raises(RuntimeError, match='no CUDA kernel'):
        trace.validate_trace(db, '')


def test_trace_requires_api_records(tmp_path):
    db = database(tmp_path / 'kernel-only.sqlite', {'CUPTI_ACTIVITY_KIND_KERNEL': 12})
    with pytest.raises(RuntimeError, match='no CUDA API'):
        trace.validate_trace(db, '')


def test_trace_accepts_imported_cuda(tmp_path):
    counts = {'CUPTI_ACTIVITY_KIND_KERNEL': 12, 'CUPTI_ACTIVITY_KIND_RUNTIME': 86}
    db = database(tmp_path / 'complete.sqlite', counts)
    assert trace.validate_trace(db, 'Generated report') == counts


@pytest.mark.parametrize('message', ['Importation succeeded with non-fatal errors',
                                    'Connection to Agent lost'])
def test_trace_rejects_diagnostics_with_existing_records(tmp_path, message):
    db = database(tmp_path / 'partial.sqlite', {'CUPTI_ACTIVITY_KIND_KERNEL': 12,
                                               'CUPTI_ACTIVITY_KIND_RUNTIME': 86})
    with pytest.raises(RuntimeError):
        trace.validate_trace(db, message)


@pytest.mark.parametrize('excluded', [None, ''])
@pytest.mark.parametrize('mode, ending', [('range', 'repeat'), ('single-range', 'stop')])
def test_trace_cli_domain_filter_and_manifest(tmp_path, monkeypatch, excluded, mode, ending):
    prefix = tmp_path / 'codec'
    args = ['trace.py', '--output', str(prefix), '--cuda-profiler-' + mode]
    if excluded is not None:
        args += ['--nvtx-domain-exclude', excluded]
    command = ['python', 'workload.py', '--cuda-profiler-range']
    monkeypatch.setattr(sys, 'argv', args + ['--'] + command)
    monkeypatch.setattr(trace.subprocess, 'check_output', lambda *a, **k: 'Nsight test version\n')
    invocations = []
    def run(invocation, *, stdout, stderr):
        invocations.append(invocation)
        (tmp_path / 'codec.1.nsys-rep').touch()
        database(tmp_path / 'codec.1.sqlite', {'CUPTI_ACTIVITY_KIND_KERNEL': 12,
                                             'CUPTI_ACTIVITY_KIND_RUNTIME': 86})
        stdout.write('Generated report\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(trace.subprocess, 'run', run)
    trace.main()
    invocation = invocations[0]
    assert invocation[-len(command):] == command
    assert '--capture-range=cudaProfilerApi' in invocation
    assert '--capture-range-end=' + ending in invocation
    assert '--kill=none' in invocation
    assert ('--nvtx-domain-exclude=TSL' in invocation) == (excluded is None)
    assert not any(flag == '--nvtx-domain-exclude=' for flag in invocation)
    manifest = json.loads((tmp_path / 'codec.trace.json').read_text())
    assert manifest['validated'] and len(manifest['records']) == 1
    assert manifest['nvtx_domain_exclude'] == ('TSL' if excluded is None else '')
    assert manifest['capture_mode'] == ('repeat' if mode == 'range' else 'single')


def test_trace_cli_capture_modes_mutually_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['trace.py', '--output', str(tmp_path / 'codec'),
                                    '--cuda-profiler-range', '--cuda-profiler-single-range',
                                    '--', 'workload'])
    monkeypatch.setattr(trace.subprocess, 'check_output',
                        lambda *a, **k: pytest.fail('invalid CLI started Nsight'))
    with pytest.raises(SystemExit) as error:
        trace.main()
    assert error.value.code == 2


def test_trace_cli_rejects_lost_agent_without_report(tmp_path, monkeypatch):
    prefix = tmp_path / 'codec'
    monkeypatch.setattr(sys, 'argv', ['trace.py', '--output', str(prefix), '--', 'workload'])
    monkeypatch.setattr(trace.subprocess, 'check_output', lambda *a, **k: 'Nsight test version\n')
    def run(invocation, *, stdout, stderr):
        stdout.write('Connection to Agent lost. Internal reason: End of file\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(trace.subprocess, 'run', run)
    with pytest.raises(SystemExit) as error:
        trace.main()
    assert error.value.code == 1
    manifest = json.loads((tmp_path / 'codec.trace.json').read_text())
    assert not manifest['validated'] and 'disconnected' in manifest['error']

"""A successful CLI exit is insufficient evidence of a usable CUDA trace."""
import importlib.util
from pathlib import Path
import sqlite3

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

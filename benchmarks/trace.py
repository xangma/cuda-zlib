#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Capture a CUDA workload with Nsight Systems and verify its imported records."""

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess


def validate_log(log):
    # Nsight can return zero and create a partial report after an import error.
    if any(message in log for message in
           ('TargetProfilingFailed', 'ProcessEventsError', 'Cannot find string for an exterior index',
            'Importation succeeded with non-fatal errors')):
        raise RuntimeError('Nsight failed to import CUDA records; see the trace log')
    if 'Connection to Agent lost' in log:
        raise RuntimeError('Nsight capture agent disconnected; see the trace log')


def validate_trace(database, log):
    validate_log(log)
    if not database.is_file():
        raise RuntimeError('Nsight did not export the SQLite trace')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {name: connection.execute('SELECT COUNT(*) FROM "' + name + '"').fetchone()[0]
                  for name in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_RUNTIME',
                               'CUPTI_ACTIVITY_KIND_DRIVER', 'NVTX_EVENTS') if name in tables}
    if not counts.get('CUPTI_ACTIVITY_KIND_KERNEL', 0):
        raise RuntimeError('Nsight report contains no CUDA kernel records')
    if not counts.get('CUPTI_ACTIVITY_KIND_RUNTIME', 0) + counts.get('CUPTI_ACTIVITY_KIND_DRIVER', 0):
        raise RuntimeError('Nsight report contains no CUDA API records')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nsys', default='nsys', help='Nsight Systems CLI executable')
    parser.add_argument('--output', type=Path, required=True, help='new report prefix')
    parser.add_argument('--cuda-profiler-range', action='store_true',
                        help='capture each cudaProfilerStart/Stop range, excluding startup')
    parser.add_argument('--nvtx-domain-exclude', default='TSL',
                        help='excluded NVTX domains (default: TSL, for JAX import compatibility; empty includes all)')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='workload command after --')
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a workload command is required after --')
    prefix = args.output.resolve()
    if any(path.suffix in ('.nsys-rep', '.sqlite', '.qdstrm', '.log', '.json')
           for path in prefix.parent.glob(prefix.name + '*')):
        parser.error('output prefix already exists; choose a new prefix')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([args.nsys, '--version'], text=True).strip()
    flags = ['--capture-range=cudaProfilerApi', '--capture-range-end=repeat', '--kill=none'] \
        if args.cuda_profiler_range else []
    if args.nvtx_domain_exclude:
        flags.append('--nvtx-domain-exclude=' + args.nvtx_domain_exclude)
    invocation = [args.nsys, 'profile', '--trace=cuda,nvtx', '--sample=none',
                  '--cpuctxsw=none', '--discard-environment=true', '--export=sqlite',
                  '--output=' + str(prefix), *flags, *command]
    log_path = Path(str(prefix) + '.log')
    with log_path.open('w') as log:
        result = subprocess.run(invocation, stdout=log, stderr=subprocess.STDOUT)
    report = dict(nsys_version=version, command=invocation, exit_code=result.returncode,
                  log=str(log_path), nvtx_domain_exclude=args.nvtx_domain_exclude, validated=False)
    try:
        if result.returncode:
            raise RuntimeError(f'Nsight or the workload failed (exit {result.returncode})')
        validate_log(log_path.read_text())
        reports = sorted(prefix.parent.glob(prefix.name + '*.nsys-rep'))
        if not reports:
            raise RuntimeError('Nsight did not produce a report')
        report['records'] = {path.name: validate_trace(path.with_suffix('.sqlite'), log_path.read_text())
                             for path in reports}
        report['validated'] = True
    except (RuntimeError, sqlite3.Error) as exc:
        report['error'] = str(exc)
    Path(str(prefix) + '.trace.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['validated']:
        parser.exit(1, report['error'] + '; log: ' + str(log_path) + '\n')
    print(json.dumps(report['records']))


if __name__ == '__main__':
    main()

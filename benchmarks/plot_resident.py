#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Plot completed normal profile_resident.py samples (never capture timings)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'zeros': 'Zeros', 'text': 'Synthetic text', 'uint32': 'Ascending uint32',
          'float32': 'Normal float32', 'random': 'Random bytes'}
COLORS = {'zeros': '#0072B2', 'text': '#D55E00', 'uint32': '#009E73',
          'float32': '#CC79A7', 'random': '#475569'}
MARKERS = {'zeros': 'o', 'text': 'D', 'uint32': '^', 'float32': 's', 'random': 'X'}


def report_identifier(path):
    try:
        return path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        return path.name


def save_figure(figure, path):
    metadata = {
        '.png': {'Software': 'cuda-zlib benchmark plotting'},
        '.svg': {'Date': None, 'Creator': 'cuda-zlib benchmark plotting'},
        '.pdf': {'CreationDate': None, 'ModDate': None, 'Creator': 'cuda-zlib benchmark plotting'},
    }
    figure.savefig(path, dpi=180, metadata=metadata[path.suffix])
    if path.suffix == '.svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true', help='validate the report without writing figures')
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    if report.get('schema_version') != 1 or report.get('complete') is not True or report['arguments'].get('cuda_profiler_range'):
        parser.error('a complete normal timing report is required')
    rows = report['cases']
    sizes = sorted(report['arguments']['sizes'])
    workloads = report['arguments']['workloads']
    sample_count = report['arguments']['samples']
    if not sizes or len(set(sizes)) != len(sizes) or not all(type(size) is int and size > 0 for size in sizes) or \
            not workloads or len(set(workloads)) != len(workloads) or not set(workloads) <= set(LABELS) or \
            type(sample_count) is not int or sample_count < 1:
        parser.error('distinct known workloads, positive sizes and sample count required')
    cases = {(row['input_bytes'], row['workload']): row for row in rows}
    if len(cases) != len(rows) or set(cases) != {(size, workload) for size in sizes for workload in workloads}:
        parser.error('unique complete size/workload matrix required')
    for row in rows:
        samples = row['seconds']
        if row['byte_exact'] is not True or row['cpu_codec_forbidden'] is not True or len(samples) != sample_count or \
                not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples) or \
                not math.isclose(statistics.median(samples), row['median_seconds'], rel_tol=1e-10):
            parser.error('invalid case status, samples or median')
    if args.validate_only:
        print(f"Validated {len(rows)} cases; source SHA-256 {hashlib.sha256(args.input.read_bytes()).hexdigest()}")
        return
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'cuda-zlib-resident-v1', 'pdf.fonttype': 42,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), layout='constrained')
    x = np.arange(len(sizes)); width = .8 / len(workloads)
    for i, workload in enumerate(workloads):
        selected = [cases[size, workload] for size in sizes]
        latency = np.array([row['median_seconds'] * 1000 for row in selected])
        low = np.array([min(row['seconds']) * 1000 for row in selected])
        high = np.array([max(row['seconds']) * 1000 for row in selected])
        position = x + (i - (len(workloads) - 1) / 2) * width
        style = {'label': LABELS[workload], 'color': COLORS[workload],
                 'marker': MARKERS[workload], 'linestyle': 'none',
                 'markersize': 5, 'capsize': 3, 'elinewidth': 1}
        axes[0].errorbar(position, latency, yerr=np.array([latency-low, high-latency]), **style)
        throughput = np.array(sizes) / 1024**2 / (latency / 1000)
        slow = np.array(sizes) / 1024**2 / (high / 1000)
        fast = np.array(sizes) / 1024**2 / (low / 1000)
        axes[1].errorbar(position, throughput, yerr=np.array([throughput-slow, fast-throughput]), **style)
    for axis in axes:
        axis.set_xticks(x, [f'{size/1024:g} KiB' if size < 1024**2 else
                           f'{size/1024**2:g} MiB' for size in sizes])
        axis.set_xlabel('Uncompressed stream size')
        axis.set_axisbelow(True); axis.grid(axis='y', alpha=.2)
        axis.set_yscale('log')
    axes[0].set_ylabel('Completed latency (ms, logarithmic)')
    axes[1].set_ylabel('Throughput (MiB/s, logarithmic)')
    axes[0].set_title('Lower latency is better'); axes[1].set_title('Higher throughput is better')
    snapshots = report['environment'].get('gpu_before', [])
    device_index = report['arguments']['device']
    gpu = snapshots[device_index].split(',')[0] if isinstance(snapshots, list) and \
        0 <= device_index < len(snapshots) else 'CUDA GPU'
    fig.suptitle('Resident checked JIT decompression — ' + gpu, fontsize=15)
    axes[1].legend(loc='upper left', fontsize=9)
    fig.supxlabel(f'Stdlib level-6 streams · medians of {sample_count} completed calls · whiskers: all sample min/max, not confidence intervals\n'
                  'Device inputs/outputs; uploads, compilation and host status/byte checks excluded; no CPU comparison', fontsize=9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exports = {}
    for extension in ('png', 'svg', 'pdf'):
        path = args.output_dir / ('resident-checked.' + extension)
        exports[path.name] = save_figure(fig, path)
    plt.close(fig)
    (args.output_dir/'resident-checked-manifest.json').write_text(json.dumps(
        {'schema_version': 1, 'source_report': report_identifier(args.input),
         'source_report_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
         'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'renderer': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'backend': 'Agg'},
         'measurement_source': {'revision': report.get('source_revision'),
                                'sha256': report['source_sha256'], 'harness_sha256': report['harness_sha256']},
         'environment': report['environment'], 'native_build': report['native_build'],
         'arguments': {key: value for key, value in report['arguments'].items() if key != 'output'},
         'methodology': report['methodology'], 'validated_cases': len(rows), 'samples_per_case': sample_count,
         'latency_units': 'median completed seconds * 1000, milliseconds',
         'throughput_units': 'uncompressed bytes / 2**20 / median completed seconds, MiB/s',
         'error_bars': 'all sample minimum and maximum latency; throughput extrema are inverse individual sample times; not confidence intervals',
         'series_styles': {workload: {'label': LABELS[workload], 'color': COLORS[workload], 'marker': MARKERS[workload]}
                           for workload in workloads}, 'exports': exports}, indent=2)+'\n')


if __name__ == '__main__':
    main()

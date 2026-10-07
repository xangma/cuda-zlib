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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    if not report.get('complete') or report['arguments'].get('cuda_profiler_range'):
        parser.error('a complete normal timing report is required')
    rows = report['cases']
    sizes = sorted(set(row['input_bytes'] for row in rows))
    workloads = list(dict.fromkeys(row['workload'] for row in rows))
    cases = {(row['input_bytes'], row['workload']): row for row in rows}
    if not rows or len(cases) != len(rows) or len(cases) != len(sizes) * len(workloads):
        parser.error('unique complete size/workload matrix required')
    for row in rows:
        samples = row['seconds']
        if not row['byte_exact'] or not row['cpu_codec_forbidden'] or not samples or \
                not all(math.isfinite(x) and x > 0 for x in samples) or \
                not math.isclose(statistics.median(samples), row['median_seconds'], rel_tol=1e-10):
            parser.error('invalid case status, samples or median')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), layout='constrained')
    x = np.arange(len(sizes)); width = .8 / len(workloads)
    for i, workload in enumerate(workloads):
        selected = [cases[size, workload] for size in sizes]
        latency = np.array([row['median_seconds'] * 1000 for row in selected])
        low = np.array([min(row['seconds']) * 1000 for row in selected])
        high = np.array([max(row['seconds']) * 1000 for row in selected])
        position = x + (i - (len(workloads) - 1) / 2) * width
        label = {'zeros': 'Zero bytes', 'text': 'Generated text', 'uint32': 'Integer counters', 'float32': 'Gaussian floats', 'random': 'Random bytes'}.get(workload, workload)
        axes[0].bar(position, latency, width, label=label,
                    yerr=np.array([latency-low, high-latency]), capsize=3)
        throughput = np.array(sizes) / 1024**2 / (latency / 1000)
        slow = np.array(sizes) / 1024**2 / (high / 1000)
        fast = np.array(sizes) / 1024**2 / (low / 1000)
        axes[1].bar(position, throughput, width, label=label,
                    yerr=np.array([throughput-slow, fast-throughput]), capsize=3)
    for axis in axes:
        axis.set_xticks(x, [f'{size/1024**2:g} MiB' for size in sizes])
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
    fig.supxlabel('Stdlib level-6 streams · median completed calls · whiskers: sample min/max\nUploads, compilation and host status/byte checks excluded; no CPU comparison', fontsize=9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exports = {}
    for extension in ('png', 'svg', 'pdf'):
        path = args.output_dir / ('resident-checked.' + extension)
        fig.savefig(path, dpi=180)
        exports[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    (args.output_dir/'resident-checked-manifest.json').write_text(json.dumps(
        {'source_report_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
         'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'exports': exports}, indent=2)+'\n')


if __name__ == '__main__':
    main()

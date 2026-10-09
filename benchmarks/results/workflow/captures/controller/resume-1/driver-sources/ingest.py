#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Validate a downloaded refresh bundle and stage all seven figure sets.

Never launches CUDA or Nsight. Default writes only a fresh private staging tree.
--apply copies validated reports/artifacts/exports into the explicit source root;
it does not edit documentation, source, historical private data, or Git.
"""
import argparse
from collections import Counter
import itertools
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

from common import check_native, freeze, read, require, run, sha, utc, write

WORKLOADS = ['zeros', 'text', 'uint32', 'float32', 'random']
SIZES = [65536, 1048576, 67108864]


class Ingest:
    def __init__(self, args):
        self.a = args
        self.bundle, self.root, self.out = (p.resolve() for p in (args.bundle, args.root, args.output_dir))
        require(not self.out.exists(), 'private output directory must be fresh')
        self.receipt = read(self.bundle / 'RUN.json')
        require(self.receipt['complete'] is True and not self.receipt.get('prepared_only'), 'measurement run incomplete')
        require(self.receipt['source_revision'] == args.revision, 'wrong measurement revision')
        self.frozen = read(self.bundle / 'FROZEN.json')
        require(sha(self.bundle / 'FROZEN.json') == self.receipt['frozen_sha256'], 'freeze receipt changed')
        freeze(self.root, args.revision, self.frozen)
        self.native = read(self.bundle / 'expected-native.json')
        check_native(read(self.bundle / 'loaded-native.json'), self.native)
        require(self.native['source_sha256'] == self.frozen['source_sha256'], 'native source mismatch')
        self.out.mkdir(parents=True)
        self.stage = self.out / 'staged-repository'
        (self.stage / 'benchmarks').mkdir(parents=True)
        for name, digest in self.frozen['files_sha256'].items():
            source, target = self.root / name, self.stage / name
            require(sha(source) == digest, 'source changed during ingest')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        self.checks = []
        self.publish = set()
        self.hash_index = {}

    def call(self, name, command, timeout=300):
        run(command, self.stage, self.env, self.out / 'logs' / (name + '.log'), timeout)

    def copy(self, source, destination, digest=None):
        destination = Path(destination)
        require(not destination.is_absolute() and '..' not in destination.parts, 'unsafe publication path')
        if digest is not None:
            require(sha(source) == digest, f'artifact checksum mismatch: {source}')
        target = self.stage / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(sha(target) == sha(source), f'conflicting artifact: {destination}')
        else:
            shutil.copyfile(source, target)
        self.publish.add(destination.as_posix())
        return target

    def identity(self, report, kind, harness):
        require(report.get('complete') is True, f'{kind}: incomplete')
        if kind != 'resident':
            require(report['source_revision'] == self.a.revision, f'{kind}: wrong source revision')
        sources = (report['environment']['codec_sha256'] if kind == 'general' else
                   {k.removeprefix('src/cuda_zlib/'): v for k, v in report['source']['sha256'].items()
                    if k.startswith('src/cuda_zlib/')} if kind == 'batch' else report['source_sha256'])
        require(sources == self.frozen['source_sha256'], f'{kind}: runtime source mismatch')
        require(report['harness_sha256'] == self.frozen['files_sha256']['benchmarks/' + harness],
                f'{kind}: harness hash mismatch')
        actual = report['environment']['native_build'] if kind in ('general', 'batch') else report['native_build']
        check_native(actual, self.native)
        uuid = report.get('gpu_uuid') or report.get('environment', {}).get('gpu_uuid')
        if uuid is not None:
            require(uuid == self.receipt['arguments']['gpu_uuid'], f'{kind}: different GPU UUID')
        for field in ('dependencies_sha256', 'extractor_dependencies_sha256'):
            for path, digest in report.get(field, {}).items():
                require(self.frozen['files_sha256'].get(path) == digest, f'{kind}: stale dependency {path}')
        for field, path in (('extractor_sha256', {'nsight': 'extract_nsight.py', 'timeline': 'extract_timeline.py',
                                               'workflow': 'extract_workflow.py'}.get(kind)),
                            ('benchmark_sha256', 'benchmark.py'), ('profile_resident_sha256', 'profile_resident.py')):
            if field in report and path:
                require(report[field] == self.frozen['files_sha256']['benchmarks/' + path], f'{kind}: stale {field}')

    def measurements(self):
        date, seed = self.receipt['arguments']['date'], self.receipt['arguments']['seed']
        specifications = [('general', f'rtx4090-{date}.json', 'benchmark.py', SIZES, WORKLOADS),
            ('batch', f'small-batch-rtx4090-{date}.json', 'small_batch.py', [256,4096,65536], ['zeros','text','random']),
            ('resident', f'resident-checked-rtx4090-{date}.json', 'profile_resident.py',
             [65536,131072,262144,1048576,8388608,67108864], WORKLOADS)]
        for kind, filename, script, sizes, workloads in specifications:
            raw = self.bundle / 'results' / filename
            report = read(raw)
            self.identity(report, kind, script)
            args = report['arguments']
            require(args['sizes'] == sizes and args['workloads'] == workloads and args['seed'] == seed
                    and args['device'] == 0, f'{kind}: wrong matrix/seed/device')
            expected = set(itertools.product(workloads, sizes, [1,8,32,128])) if kind == 'batch' else set(itertools.product(workloads, sizes))
            rows = report['cases']
            ids = [(c['workload'], c['file_bytes'], c['file_count']) if kind == 'batch' else
                   (c['workload'], c['input_bytes']) for c in rows]
            require(len(ids) == len(expected) and set(ids) == expected, f'{kind}: incomplete/duplicate cases')
            if kind == 'batch':
                require(args['counts'] == [1,8,32,128] and args['roundtrip'] is True and args['samples'] == 7,
                        'wrong batch counts/sample/roundtrip scope')
            count = 0
            for case in rows:
                if kind == 'resident':
                    require(case['byte_exact'] is True and case['cpu_codec_forbidden'] is True, 'resident oracle failed')
                    timings = {'resident': case}
                else:
                    require(case['validation'] and len(case['timings']) == 11, 'timing workflow matrix incomplete')
                    timings = case['timings']
                for name, timing in timings.items():
                    values = timing['seconds']
                    n = 31 if kind == 'resident' else 7 if kind == 'batch' else 5 if name.startswith('cpu_') else 15
                    require(len(values) == n and all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in values), 'invalid timing samples')
                    require(timing['median_seconds'] == statistics.median(values), 'median differs from raw samples')
                    if kind != 'resident':
                        require(timing['min_seconds'] == min(values) and timing['max_seconds'] == max(values), 'wrong extrema')
                        size = case['input_bytes'] if kind == 'general' else case['total_input_bytes']
                        require(math.isclose(timing['mib_per_second'], size / (1 << 20) / timing['median_seconds'], rel_tol=1e-12), 'wrong throughput')
                        if kind == 'batch':
                            require(timing['median_seconds_per_file'] == timing['median_seconds'] / case['file_count'], 'wrong per-file median')
                    count += n
            destination = Path('benchmarks/results') / filename
            target = self.copy(raw, destination)
            if kind == 'resident':
                require('source_revision' not in report or report['source_revision'] == self.a.revision, 'wrong resident annotation')
                original = dict(report)
                report['source_revision'] = self.a.revision
                write(target, report)
                changed = read(target); changed.pop('source_revision')
                original.pop('source_revision', None)
                require(changed == original, 'resident changed beyond revision annotation')
            self.checks.append(dict(kind=kind, cases=len(rows), samples=count, raw_sha256=sha(raw),
                                    published_sha256=sha(target), resident_revision_annotation=kind == 'resident'))

    def artifacts(self, report):
        # Search only known raw-artifact directories, excluding caches and private SQLite.
        if not self.hash_index:
            for directory in ('stages', 'timeline', 'workflows'):
                for path in (self.bundle / directory).rglob('*'):
                    if path.is_file() and path.suffix != '.sqlite':
                        self.hash_index.setdefault(sha(path), []).append(path)
        for name, digest in report['artifact_sha256'].items():
            matches = [p for p in self.hash_index.get(digest, []) if p.name == Path(name).name]
            require(matches, f'missing raw artifact: {name}')
            self.copy(matches[0], name, digest)

    def extract_again(self, name, script, args, original):
        result = self.out / 'reextracted' / (name + '.json')
        self.call('reextract-' + name, [self.a.python, 'benchmarks/' + script, *map(str,args), '--output', str(result)])
        expected, actual = read(original), read(result)
        for value in (expected, actual):
            value.pop('created_utc', None)
        require(expected == actual, f'{name}: current extractor differs from captured normalization')

    def diagnostics(self):
        raw, destination = self.bundle / 'results/nsight.json', 'benchmarks/results/nsight/rtx4090-decode.json'
        report = read(raw); self.identity(report, 'nsight', 'profile_resident.py')
        self.extract_again('nsight', 'extract_nsight.py', ['--capture-dir', self.bundle / 'stages',
            '--stage-receipt', self.bundle / 'stages/STAGE.json', '--native-receipt', self.bundle / 'stages/native.json',
            '--source-revision', self.a.revision], raw)
        self.copy(raw, destination); self.artifacts(report)
        raw = self.bundle / 'results/timeline.json'
        report = read(raw); self.identity(report, 'timeline', 'profile_timeline.py')
        self.extract_again('timeline', 'extract_timeline.py', ['--sqlite', self.bundle / 'timeline/live.sqlite',
            '--telemetry', self.bundle / 'timeline/live-telemetry.json', '--receipt', self.bundle / 'timeline/live-capture.json', '--root', self.stage], raw)
        self.copy(raw, 'benchmarks/results/timeline/rtx4090-float32.json'); self.artifacts(report)
        comparison = read(self.bundle / 'workflows/comparison.json')
        require(comparison['complete'] is True and len(comparison['runs']) == 10, 'all ten instrumentation conditions required')
        variants = ['plain', 'telemetry', 'cuda-nvtx', 'cuda-nvtx-osrt', 'plain-repeat']
        require([(r['operation'], r['variant']) for r in comparison['runs']] == list(itertools.product(('compress','decompress'),variants)), 'wrong workflow condition order/matrix')
        records, inventory = [], {}
        for row in comparison['runs']:
            operation, variant = row['operation'], row['variant']
            d = self.bundle / 'workflows' / operation / variant
            telemetry = read(d / 'telemetry.json')
            self.identity(telemetry, 'control', 'profile_workflow.py')
            require(sha(d / 'telemetry.json') == row['telemetry_sha256'], 'workflow raw telemetry changed')
            normalized = self.out / 'controls' / f'{operation}-{variant}.json'
            cmd = [self.a.python, 'benchmarks/extract_workflow.py', '--telemetry', str(d / 'telemetry.json'), '--root', str(self.stage)]
            if row['trace']:
                raw = d / 'normalized.json'
                report = read(raw); self.identity(report, 'workflow', 'profile_workflow.py')
                require(sha(raw) == row['normalized_sha256'], 'normalized workflow changed')
                self.extract_again(operation + '-' + variant, 'extract_workflow.py',
                    ['--telemetry', d / 'telemetry.json', '--sqlite', d / 'capture.sqlite', '--receipt', d / 'capture.json', '--root', self.stage], raw)
                suffix = '-osrt' if variant.endswith('-osrt') else ''
                public = f'benchmarks/results/workflow/rtx4090-{operation}-float32{suffix}.json'
                self.copy(raw, public); self.artifacts(report)
                inventory.update(report['artifact_sha256'])
            else:
                self.call('control-' + operation + '-' + variant, cmd + ['--control', '--output', str(normalized)])
                for filename in ('capture.log','command.json','telemetry-worker.json','telemetry-worker.log','telemetry.json'):
                    public = f'benchmarks/results/workflow/captures/{operation}/{variant}/{filename}'
                    self.copy(d / filename, public)
                    inventory[public] = sha(d / filename)
            # Preserve measured summaries; replace private absolute origin paths with
            # portable public identifiers, leaving the raw run.json unchanged.
            record = {k:v for k,v in row.items() if k not in ('directory','normalized_path')}
            record['telemetry_path'] = f'benchmarks/results/workflow/captures/{operation}/{variant}/telemetry.json'
            if row['trace']:
                record['normalized_path'] = f'benchmarks/results/workflow/rtx4090-{operation}-float32{suffix}.json'
            records.append(record)
        require(len(inventory) == 62, 'wrong instrumentation artifact count')
        path = 'benchmarks/results/workflow/instrumentation.json'
        write(self.stage / path, dict(schema_version=1, kind='workflow_instrumentation_controls', complete=True,
            source_revision=self.a.revision, methodology='Sequential plain, telemetry, CUDA/NVTX, CUDA/NVTX/OSRT and repeated plain conditions; instrumented diagnostics, shared workstation.',
            runs=records, artifact_sha256=dict(sorted(inventory.items()))))
        self.publish.add(path)

    def host_and_download(self):
        raw = self.bundle / 'results/host-outputs.json'; report = read(raw)
        self.identity(report, 'host_outputs', 'host_outputs.py')
        require(report['arguments']['samples'] == 12 and report['arguments']['warmups'] == 2, 'wrong host-output sampling')
        self.copy(raw, 'benchmarks/results/host-outputs/rtx4090-float32.json')
        raw = self.bundle / 'download/report.json'; report = read(raw)
        self.identity(report, 'download', 'profile_download.py')
        args = report['arguments']
        require(args['iterations'] == 12 and args['warmups'] == 2 and args['size'] == 67108864 and args['seed'] == self.receipt['arguments']['seed'], 'wrong download fixture')
        require([r['operation'] for r in report['operations']] == ['compress','decompress'], 'missing download operation')
        checks = 0
        for operation in report['operations']:
            require([m['name'] for m in operation['modes']] == ['numpy','npinned','depth2pinned'], 'missing download modes')
            for mode in operation['modes']:
                require(len(mode['cases']) == 12 and len(mode['warmups']) == 2, 'incomplete download samples')
                for row in mode['cases'] + mode['warmups']:
                    require(row['byte_exact'] is True and row['status_code'] == 0 and row['codec_transfer_cpu_codec_forbidden'] is True and row['validation_outside_cpu_codec_guard'] is True, 'download output validation failed')
                    require(row['end_ns'] > row['start_ns'], 'invalid download interval')
                    checks += 1
                summary = mode['summary']; cases = mode['cases']
                require(summary['case_latency_median_ns'] == statistics.median(r['end_ns']-r['start_ns'] for r in cases), 'wrong download latency median')
                require(summary['batch_wall_per_case_ns'] == summary['batch_wall_ns']/len(cases), 'wrong download batch arithmetic')
                for name, metrics in summary['phase_metrics'].items():
                    selected = [p for r in cases for p in r['phases'] if p['name'] == name]
                    for key, aggregates in metrics.items():
                        values = [p[key] for p in selected if p[key] is not None]
                        require(aggregates['median'] == statistics.median(values) and aggregates['sum'] == sum(values), 'wrong download phase statistics')
            require(operation['cache_probe']['byte_exact'] is True and operation['cache_probe']['excluded_from_workflows'] is True, 'cache probe failed')
            checks += 1
        require(checks == 86, 'wrong download output count')
        public = 'benchmarks/results/workflow/rtx4090-download-float32.json'
        self.copy(raw, public)
        prefix = 'benchmarks/results/workflow/captures/download/'
        for name in ('command.json','capture.log'):
            self.copy(self.bundle / 'download' / name, prefix + name)
        evidence = {'command': {'path': prefix+'command.json', 'sha256':sha(self.bundle/'download/command.json')},
                    'log': {'path':prefix+'capture.log','sha256':sha(self.bundle/'download/capture.log')},
                    'report': {'path':public,'sha256':sha(raw)}}
        write(self.stage / (prefix+'capture.json'), dict(schema_version=1,complete=True,source_revision=self.a.revision,
              harness_sha256=report['harness_sha256'],checked_outputs=checks,artifacts=evidence))
        self.publish.add(prefix+'capture.json')

    def figures(self):
        b, f = 'benchmarks/results/', 'benchmarks/figures'
        date = self.receipt['arguments']['date']
        commands = [
            ['plot_results.py','--input',b+f'rtx4090-{date}.json','--output-dir',f],
            ['plot_small_batch.py',b+f'small-batch-rtx4090-{date}.json','--output-dir',f+'/small-batch','--prefix','rtx4090'],
            ['plot_resident.py',b+f'resident-checked-rtx4090-{date}.json','--output-dir',f],
            ['plot_nsight.py'], ['plot_timeline.py'],
            ['plot_workflow.py','--compress',b+'workflow/rtx4090-compress-float32.json','--decompress',b+'workflow/rtx4090-decompress-float32.json'],
            ['plot_host_outputs.py']]
        for command in commands:
            base = [self.a.python, 'benchmarks/' + command[0], *command[1:]]
            self.call(command[0]+'-validate',base+['--validate-only'])
            self.call(command[0]+'-render',base)
        self.call('verify-figures',[self.a.python,'benchmarks/verify_figures.py'])
        exports = [p for p in (self.stage/'benchmarks/figures').rglob('*') if p.suffix in ('.png','.svg','.pdf')]
        require(len(exports) == 63, 'expected exactly 63 exports')
        for path in (self.stage/'benchmarks/figures').rglob('*'):
            if path.is_file(): self.publish.add(path.relative_to(self.stage).as_posix())

    def collector_provenance(self):
        """Keep controller/wrapper evidence separate from the 62 workflow artifacts."""
        import json
        prefix = 'benchmarks/results/workflow/captures/controller/'
        commands = [json.loads(line) for line in (self.bundle/'nsys-effective-argv.jsonl').read_text().splitlines() if line]
        profiles = [r for r in commands if r['command'][1] == 'profile']
        require(len(profiles) == 4, 'expected four effective workflow profiler launches')
        traces = Counter()
        for row in profiles:
            command = row['command']
            require('--cuda-flush-interval=0' in command and '--wait=all' in command,
                    'effective workflow profiler command lacks flush/wait flags')
            traces.update(x.removeprefix('--trace=') for x in command if x.startswith('--trace='))
        require(traces == {'cuda,nvtx':2,'cuda,nvtx,osrt':2}, 'wrong effective workflow trace matrix')
        drivers = self.bundle/'driver-sources'
        require((drivers/'common.py').is_file() and (drivers/'refresh_controller.py').is_file(),
                'captured actual driver sources are required; do not substitute local repaired files')
        files = [self.bundle/name for name in ('nsys-forward','nsys-effective-argv.jsonl','RUN.json',
                 'FROZEN.json','expected-native.json','loaded-native.json')]
        files += [p for p in drivers.rglob('*') if p.is_file()]
        hashes = {}
        for source in files:
            relative = source.relative_to(self.bundle).as_posix()
            destination = prefix+relative
            self.copy(source,destination)
            hashes[destination]=sha(source)
        path = prefix+'collector-provenance.json'
        write(self.stage/path,dict(schema_version=1,kind='publication_collector_provenance',complete=True,
            source_revision=self.a.revision,artifact_sha256=dict(sorted(hashes.items())),
            captured_driver_sha256={p.relative_to(drivers).as_posix():sha(p) for p in drivers.rglob('*') if p.is_file()},
            effective_workflow_profiles=len(profiles),
            local_ingest_sha256=sha(Path(__file__)),local_ingest_helper_sha256=sha(Path(__file__).with_name('common.py')),
            methodology='Original captured driver sources, forwarding executable and effective profiler argv are preserved byte-for-byte. Local ingest helper identity is recorded separately; it does not replace the live controller source. The 62 workflow measurement artifacts retain their original independent inventory.'))
        self.publish.add(path)

    def execute(self):
        self.measurements(); self.diagnostics(); self.host_and_download(); self.collector_provenance(); self.figures()
        freeze(self.root,self.a.revision,self.frozen)
        receipt = dict(complete=True,source_revision=self.a.revision,created_utc=utc(),
            frozen_sha256=sha(self.bundle/'FROZEN.json'),measurement_checks=self.checks,
            manifests=7,exports=63,published_sha256={p:sha(self.stage/p) for p in sorted(self.publish)},
            apply_requested=self.a.apply,applied=False,
            resident_provenance='Published resident copy adds only source_revision after exact source/harness/native validation; raw report is preserved.',
            remaining='Documentation numerical tables/links and visual QA require primary review; no isolation or causal speedup claim established.')
        write(self.out/'INGEST_REVIEW.json',receipt)
        if self.a.apply:
            for name in sorted(self.publish):
                target=self.root/name; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(self.stage/name,target)
            # Remove obsolete RTX4090 matrix snapshots only, retaining Apple CPU references.
            current_date=self.receipt['arguments']['date']
            for pattern in ('rtx4090-????????.json','small-batch-rtx4090-????????.json','resident-checked-rtx4090-????????.json'):
                for path in (self.root/'benchmarks/results').glob(pattern):
                    if not path.name.endswith(current_date+'.json'): path.unlink()
            self.call('verify-applied',[self.a.python,str(self.root/'benchmarks/verify_figures.py')])
            receipt['applied']=True; write(self.out/'INGEST_REVIEW.json',receipt)
        print(f"Validated seven manifests / 63 exports; review {self.out/'INGEST_REVIEW.json'}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--apply',action='store_true')
    Ingest(p.parse_args()).execute()


if __name__=='__main__':
    main()

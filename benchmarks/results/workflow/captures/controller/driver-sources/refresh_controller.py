#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Sequential, bounded GPU publication refresh. Run only on the chosen GPU host.

--prepare-only verifies a Git checkout and writes FROZEN.json without CUDA work.
An archived source tree must receive that FROZEN.json through --frozen-manifest.
The normal run requires a fresh output directory and a prepared native receipt.
"""
import argparse
import os
from pathlib import Path
import signal
import socket
import sqlite3
import sys
import time

from common import check_native, freeze, native_receipt, read, require, run, sha, utc, write

WORKLOADS = ['zeros', 'text', 'uint32', 'float32', 'random']
SIZES = [65536, 1048576, 67108864]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--native-receipt', type=Path, required=True)
    p.add_argument('--frozen-manifest', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--nsys', default='nsys')
    p.add_argument('--nvcc', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--nvtx-library', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--seed', type=int, default=20261008)
    p.add_argument('--date', default='20261009')
    p.add_argument('--total-timeout', type=float, default=7200)
    p.add_argument('--prepare-only', action='store_true')
    return p


class Controller:
    def __init__(self, args):
        self.a = args
        self.root, self.out = args.root.resolve(), args.output_dir.resolve()
        require(not self.out.exists(), 'output directory must be fresh')
        require(args.device == 0, 'stage extractor requires CUDA ordinal 0')
        require(args.total_timeout > 0, 'positive total timeout required')
        require(len(args.date) == 8 and args.date.isdigit(), 'date must be YYYYMMDD')
        self.frozen = freeze(self.root, args.revision,
                             read(args.frozen_manifest) if args.frozen_manifest else None)
        self.native = native_receipt(read(args.native_receipt), args.prepare_only)
        require(self.native['source_sha256'] == self.frozen['source_sha256'], 'native sources differ')
        self.out.mkdir(parents=True)
        write(self.out / 'FROZEN.json', self.frozen)
        write(self.out / 'expected-native.json', self.native)
        self.env = dict(os.environ, PYTHONPATH=str(self.root / 'src'),
                        PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=str(args.device),
                        CUDA_ZLIB_CACHE_DIR=str(args.cache.resolve()), CUDACXX=args.nvcc,
                        CUDA_ZLIB_SOURCE_REVISION=args.revision, XLA_PYTHON_CLIENT_PREALLOCATE='false',
                        JAX_COMPILATION_CACHE_DIR=str(self.out / 'jax-cache'))
        # Retain explicit additional dependency locations, after this frozen package.
        if os.environ.get('PYTHONPATH'):
            self.env['PYTHONPATH'] += os.pathsep + os.environ['PYTHONPATH']
        self.started = time.monotonic()
        self.steps = []
        self.loaded = None
        self.result = dict(complete=False, source_revision=args.revision, host=socket.gethostname(),
                           controller_pid=os.getpid(), controller_pgid=os.getpgrp(),
                           started_utc=utc(), arguments={k: str(v) if isinstance(v, Path) else v
                                                       for k, v in vars(args).items()},
                           frozen_sha256=sha(self.out / 'FROZEN.json'), steps=self.steps)
        write(self.out / 'RUN.json', self.result)

    def verify(self):
        freeze(self.root, self.a.revision, self.frozen)
        if self.loaded:
            require(sha(self.loaded['library']) == self.native['library_sha256'], 'loaded library changed')
            require(sha(self.loaded['build']) == self.native['build_sha256'], 'loaded build changed')

    def call(self, name, command, timeout=900, log=None):
        self.verify()
        remaining = self.a.total_timeout - (time.monotonic() - self.started)
        require(remaining > 0, 'overall publication timeout reached')
        result = run(command, self.root, self.env, log or self.out / 'logs' / (name + '.log'),
                     min(timeout, remaining))
        self.verify()
        self.steps.append(dict(name=name, **result))
        write(self.out / 'RUN.json', self.result)
        return result

    def py(self, script, *args):
        return [self.a.python, 'benchmarks/' + script, *map(str, args)]

    def profile(self, trace, output, command, capture_range=False):
        flags = [self.a.nsys, 'profile', '--trace=' + trace, '--cuda-graph-trace=node',
                 '--cuda-event-trace=false', '--cuda-flush-interval=0', '--wait=all',
                 '--sample=none', '--cpuctxsw=none', '--discard-environment=true']
        if 'nvtx' in trace:
            flags += ['--nvtx-domain-exclude=TSL']
        if capture_range:
            flags += ['--capture-range=cudaProfilerApi', '--capture-range-end=stop']
        return flags + ['--output', str(output), *command]

    def export(self, name, stem, log):
        command = [self.a.nsys, 'export', '--type', 'sqlite', '--output',
                   str(stem.with_suffix('.sqlite')), str(stem.with_suffix('.nsys-rep'))]
        self.call(name + '-export', command, 180, log)
        with sqlite3.connect(stem.with_suffix('.sqlite')) as db:
            require(db.execute('pragma quick_check').fetchone()[0] == 'ok', 'SQLite integrity failure')
            count = db.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
            require(count > 0, 'no recorded kernels')
        return command, count

    def probe(self):
        code = ('import json,sys; from pathlib import Path; sys.path.insert(0,"benchmarks"); '
                'import benchmark; from cuda_zlib import _codec; '
                'v=benchmark.native_build(_codec._select_device(0)); '
                'Path(sys.argv[1]).write_text(json.dumps(v,indent=2)+"\\n")')
        target = self.out / 'loaded-native.json'
        self.call('probe-native', [self.a.python, '-c', code, str(target)], 180)
        self.loaded = read(target)
        check_native(self.loaded, self.native)
        require(Path(self.loaded['library']).resolve().is_relative_to(self.a.cache.resolve()),
                'loaded native library is outside supplied cache')
        self.verify()

    def measurements(self):
        d = self.out / 'results'; d.mkdir()
        self.call('general', self.py('benchmark.py', '--sizes', *SIZES, '--workloads', *WORKLOADS,
                  '--samples', 15, '--cpu-samples', 5, '--seed', self.a.seed, '--device', 0,
                  '--output', d / f'rtx4090-{self.a.date}.json'), 1800)
        self.call('small-batch', self.py('small_batch.py', '--sizes', 256, 4096, 65536,
                  '--counts', 1, 8, 32, 128, '--workloads', 'zeros', 'text', 'random',
                  '--samples', 7, '--seed', self.a.seed, '--device', 0, '--chunk-bytes', 32768,
                  '--roundtrip', '--output', d / f'small-batch-rtx4090-{self.a.date}.json'), 1800)
        self.call('resident', self.py('profile_resident.py', '--sizes', 65536, 131072, 262144,
                  1048576, 8388608, 67108864, '--workloads', *WORKLOADS, '--samples', 31,
                  '--seed', self.a.seed, '--device', 0,
                  '--output', d / f'resident-checked-rtx4090-{self.a.date}.json'), 1800)
        self.call('host-outputs', self.py('host_outputs.py', '--source-revision', self.a.revision,
                  '--device', 0, '--sizes', *SIZES, '--workload', 'float32', '--seed', self.a.seed,
                  '--order-seed', 20261009, '--samples', 12, '--warmups', 2,
                  '--output', d / 'host-outputs.json'), 1200)

    def stages(self):
        d = self.out / 'stages'; d.mkdir()
        write(d / 'STAGE.json', dict(source_revision=self.a.revision,
              source_sha256=self.frozen['source_sha256'],
              harness_sha256=self.frozen['files_sha256']['benchmarks/profile_resident.py'],
              benchmark_sha256=self.frozen['files_sha256']['benchmarks/benchmark.py']))
        write(d / 'native.json', self.native)
        matrix = [(w, s) for s in SIZES for w in WORKLOADS] + [('uint32', 131072)]
        for workload, size in matrix:
            name = f'{workload}-{size}'; stem = d / name
            command = self.profile('cuda', stem, self.py('profile_resident.py', '--sizes', size,
                      '--workloads', workload, '--samples', 1, '--seed', self.a.seed,
                      '--cuda-profiler-range', '--output', d / (name + '-profile.json')), True)
            write(d / (name + '-command.json'), command)
            started = utc()
            self.call('stage-' + name, command, 300, stem.with_suffix('.log'))
            export, count = self.export('stage-' + name, stem, d / (name + '-export.log'))
            write(d / (name + '-capture.json'), dict(source_revision=self.a.revision,
                  capture_started_utc=started, command=command, export_command=export,
                  kernel_activities=count, sha256={p.name: sha(p) for p in d.glob(name + '.*')
                                                   if p.is_file() and p.suffix != '.json'},
                  profile_sha256=sha(d / (name + '-profile.json'))))
        self.call('extract-stages', self.py('extract_nsight.py', '--capture-dir', d,
                  '--stage-receipt', d / 'STAGE.json', '--native-receipt', d / 'native.json',
                  '--source-revision', self.a.revision, '--output', self.out / 'results/nsight.json'), 300)

    def timeline(self):
        d = self.out / 'timeline'; d.mkdir()
        stem = d / 'live'
        command = self.profile('cuda,nvtx,osrt', stem, self.py('profile_timeline.py',
                  '--source-revision', self.a.revision, '--gpu-uuid', self.a.gpu_uuid,
                  '--nvtx-library', self.a.nvtx_library, '--device', 0, '--size', 67108864,
                  '--workload', 'float32', '--seed', self.a.seed, '--warmups', 2,
                  '--iterations', 20, '--sample-ms', 10, '--timeout', 600,
                  '--output', d / 'live-telemetry.json'))
        write(d / 'live-command.json', command)
        started = utc()
        self.call('timeline', command, 780, d / 'live.log')
        self.export('timeline', stem, d / 'live-export.log')
        filenames = dict(nsys_report='live.nsys-rep', telemetry='live-telemetry.json',
                         command='live-command.json', log='live.log', export_log='live-export.log',
                         worker_log='live-telemetry-worker.log', worker_result='live-telemetry-worker.json',
                         sqlite='live.sqlite')
        artifacts = {}
        for key, name in filenames.items():
            item = dict(path=name, sha256=sha(d / name))
            if key == 'sqlite': item['private'] = True
            else: item['published_path'] = 'benchmarks/results/timeline/captures/' + name
            artifacts[key] = item
        self.call('nsys-version', [self.a.nsys, '--version'], 30)
        version = (self.out / 'logs/nsys-version.log').read_text().strip()
        write(d / 'live-capture.json', dict(source_revision=self.a.revision,
              capture_started_utc=started, toolchain=dict(nsys=version,
              driver=read(d / 'live-telemetry.json')['selected_gpu']['driver_version']), artifacts=artifacts))
        self.call('extract-timeline', self.py('extract_timeline.py', '--sqlite', d / 'live.sqlite',
                  '--telemetry', d / 'live-telemetry.json', '--receipt', d / 'live-capture.json',
                  '--root', self.root, '--output', self.out / 'results/timeline.json'), 300)

    def workflows(self):
        # run_workflow records its argv truthfully; this forwarding executable separately
        # records effective Nsight argv, then execs the actual profiler in the same process.
        wrapper = self.out / 'nsys-forward'
        source = ('#!' + self.a.python + '\nimport json,os,sys,time\n'
                  'args=sys.argv[1:]\n'
                  'if args and args[0]=="profile":\n'
                  ' for flag in ("--cuda-flush-interval=0","--wait=all"):\n'
                  '  if flag not in args: args.insert(1,flag)\n'
                  f'command={[self.a.nsys]!r}+args\n'
                  f'with open({str(self.out / "nsys-effective-argv.jsonl")!r},"a") as f:\n'
                  ' f.write(json.dumps({"pid":os.getpid(),"time_ns":time.time_ns(),"command":command})+"\\n")\n'
                  'os.execvp(command[0],command)\n')
        wrapper.write_text(source); wrapper.chmod(0o755)
        self.call('workflows', self.py('run_workflow.py', '--output-dir', self.out / 'workflows',
                  '--source-revision', self.a.revision, '--operation', 'both', '--gpu-uuid', self.a.gpu_uuid,
                  '--device', 0, '--nvtx-library', self.a.nvtx_library, '--nsys', wrapper,
                  '--trace', 'cuda,nvtx', '--compare-instrumentation', '--no-plots', '--size', 67108864,
                  '--workload', 'float32', '--seed', self.a.seed, '--chunk-bytes', 32768,
                  '--warmups', 2, '--iterations', 20, '--sample-ms', 10, '--timeout', 600), 7200)

    def download(self):
        d = self.out / 'download'; d.mkdir()
        command = self.py('profile_download.py', '--source-revision', self.a.revision,
                  '--operation', 'both', '--modes', 'numpy', 'npinned', 'depth2pinned',
                  '--size', 67108864, '--workload', 'float32', '--seed', self.a.seed,
                  '--iterations', 12, '--warmups', 2, '--device', 0, '--chunk-bytes', 32768,
                  '--output', d / 'report.json')
        write(d / 'command.json', dict(command=command, cwd=str(self.root), host=socket.gethostname()))
        self.call('download', command, 1800, d / 'capture.log')

    def execute(self):
        try:
            if self.a.prepare_only:
                self.result.update(complete=True, prepared_only=True)
            else:
                require(self.a.nvtx_library.is_file(), 'NVTX library missing')
                self.probe(); self.measurements(); self.stages(); self.timeline(); self.workflows(); self.download()
                self.result.update(complete=True, loaded_native=self.loaded)
        except BaseException as error:
            self.result['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            self.result.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - self.started)
            write(self.out / 'RUN.json', self.result)


def main():
    def interrupted(signum, frame):
        raise InterruptedError(f'controller received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    args = parser().parse_args()
    Controller(args).execute()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Resume only timeline/workflows/download after the missing-pynvml failure.

Preserves the original failure, driver sources and completed measurements.
The same frozen revision/root/cache are required; dependencies are explicit.
"""
import argparse
import copy
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
from types import SimpleNamespace

from common import check_native, freeze, read, require, sha, utc, write
from ingest import Ingest
from refresh_controller import Controller, SIZES, WORKLOADS


def validate_completed(controller, destination):
    """CPU-only source/native/arithmetic checks before reusing successful work."""
    v = Ingest.__new__(Ingest)
    v.a, v.bundle, v.stage = controller.a, controller.out, destination
    destination.mkdir()
    v.frozen, v.native, v.receipt = controller.frozen, controller.native, controller.prior
    v.publish, v.checks = set(), []
    v.measurements()
    require(sum(c['cases'] for c in v.checks) == 81 and sum(c['samples'] for c in v.checks) == 5577,
            'completed standard timing matrix differs')
    path = controller.out / 'results/host-outputs.json'
    report = read(path); v.identity(report, 'host_outputs', 'host_outputs.py')
    require(len(report['cases']) == 3 and sorted(c['output_bytes'] for c in report['cases']) == SIZES,
            'completed host-output matrix differs')
    require(report['arguments']['samples'] == 12 and report['arguments']['warmups'] == 2,
            'completed host-output sample counts differ')
    checked = 0
    for case in report['cases']:
        names = {f'{b}_{c}' for b in ('cpu','cuda') for c in ('ndarray','memoryview','bytes')}
        require(len(case['series']) == 6 and {s['name'] for s in case['series']} == names,
                'missing completed host-output series')
        for series in case['series']:
            require(len(series['samples']) == 12 and len(series['warmups']) == 2,
                    'incomplete completed host-output samples')
            for row in series['samples'] + series['warmups']:
                require(row['complete'] is True and row['byte_exact'] is True and row['readonly'] is True
                        and row['end_ns']-row['start_ns'] == row['wall_ns'] > 0,
                        'invalid completed host output')
                checked += 1
    require(checked == 252, 'wrong host-output validation count')
    stage = controller.out / 'results/nsight.json'
    report = read(stage); v.identity(report, 'nsight', 'profile_resident.py')
    expected = {(w,s) for w in WORKLOADS for s in SIZES} | {('uint32',131072)}
    require(len(report['cases']) == 16 and {(c['workload'],c['output_bytes']) for c in report['cases']} == expected,
            'completed stage capture matrix differs')
    require(len(report['artifact_sha256']) == 98, 'incomplete completed stage artifact inventory')
    for name,digest in report['artifact_sha256'].items():
        require(name.startswith('benchmarks/results/nsight/captures/'), 'unexpected stage artifact prefix')
        require(sha(controller.out/'stages'/Path(name).name) == digest, 'completed stage artifact changed')
    return {'standard':v.checks,'host_outputs_checked':checked,'stage_cases':16,
            'reused_report_sha256':{p.relative_to(controller.out).as_posix():sha(p)
                                  for p in (controller.out/'results').glob('*.json')}}


class Resume(Controller):
    def __init__(self, cli):
        self.out = cli.output_dir.resolve()
        self.prior = read(self.out/'RUN.json')
        require(self.prior['complete'] is False and 'pynvml' in self.prior.get('error','') or
                self.prior['complete'] is False and (self.out/'timeline/live.log').is_file()
                and 'pynvml' in (self.out/'timeline/live.log').read_text(errors='replace'),
                'resume is bounded to the recorded missing-pynvml timeline failure')
        require(not (self.out/'RUN.failure.json').exists(), 'failure already archived; choose an explicit reviewed recovery')
        self.resume_dir = self.out/cli.resume_name
        require(not self.resume_dir.exists(), 'resume evidence directory must be fresh')
        require(Path(cli.resume_name).name == cli.resume_name and cli.resume_name not in ('.','..'), 'invalid resume name')
        old = self.prior['arguments']
        for field in ('root','revision','cache'):
            supplied = getattr(cli,field)
            if supplied is not None:
                expected = old[field]
                require(str(supplied.resolve()) == str(Path(expected).resolve()) if field != 'revision'
                        else supplied == expected, f'resume cannot change frozen {field}')
        values = dict(old, python=cli.python, total_timeout=cli.total_timeout, prepare_only=False)
        for field in ('root','cache','native_receipt','frozen_manifest','nvtx_library','output_dir'):
            if values.get(field) is not None: values[field] = Path(values[field])
        self.a = SimpleNamespace(**values)
        self.root = self.a.root.resolve()
        self.frozen = read(self.out/'FROZEN.json')
        require(sha(self.out/'FROZEN.json') == self.prior['frozen_sha256'], 'freeze receipt changed')
        freeze(self.root,self.a.revision,self.frozen)
        self.native = read(self.out/'expected-native.json')
        self.loaded = read(self.out/'loaded-native.json')
        check_native(self.loaded,self.native)
        self.verify()
        expected = ['probe-native','general','small-batch','resident','host-outputs']
        for w,s in [(w,s) for s in SIZES for w in WORKLOADS]+[('uint32',131072)]:
            expected += [f'stage-{w}-{s}', f'stage-{w}-{s}-export']
        expected += ['extract-stages']
        require([r['name'] for r in self.prior['steps']] == expected and
                all(r['complete'] is True and r['exit_code'] == 0 for r in self.prior['steps']),
                'prior completed-step receipt differs from the required prefix')
        for name in ('workflows','download','nsys-forward','nsys-effective-argv.jsonl','results/timeline.json'):
            require(not (self.out/name).exists(), 'remaining output already exists: '+name)
        require((self.out/'timeline').is_dir() and not (self.out/'timeline-failed-missing-pynvml').exists(),
                'expected unique failed timeline directory')
        deps = cli.deps.resolve(); require(deps.is_dir(), 'dependency directory missing')
        self.resume_dir.mkdir()
        validation = validate_completed(self,self.resume_dir/'validation')
        shutil.copyfile(self.out/'RUN.json',self.out/'RUN.failure.json')
        (self.out/'timeline').rename(self.out/'timeline-failed-missing-pynvml')
        snapshots = self.resume_dir/'driver-sources'; snapshots.mkdir()
        for name in ('resume.py','common.py','refresh_controller.py','ingest.py'):
            shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
        self.env = dict(os.environ,PYTHONPATH=os.pathsep.join([str(self.root/'src'),str(deps),os.environ.get('PYTHONPATH','')]),
            PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',CUDA_ZLIB_CACHE_DIR=str(self.a.cache.resolve()),
            CUDACXX=self.a.nvcc,CUDA_ZLIB_SOURCE_REVISION=self.a.revision,XLA_PYTHON_CLIENT_PREALLOCATE='false',
            JAX_COMPILATION_CACHE_DIR=str(self.out/'jax-cache'))
        self.started = time.monotonic()
        self.steps = copy.deepcopy(self.prior['steps'])
        self.result = dict(complete=False,source_revision=self.a.revision,host=socket.gethostname(),
            controller_pid=os.getpid(),controller_pgid=os.getpgrp(),started_utc=utc(),
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in values.items()},
            frozen_sha256=sha(self.out/'FROZEN.json'),steps=self.steps,
            resume=dict(previous_controller_pid=self.prior['controller_pid'],reused_steps=len(expected),
                        previous_result='RUN.failure.json',previous_result_sha256=sha(self.out/'RUN.failure.json'),
                        receipt=cli.resume_name+'/RESUME.json'))
        evidence = dict(schema_version=1,source_revision=self.a.revision,controller_pid=os.getpid(),
            previous_controller_pid=self.prior['controller_pid'],started_utc=utc(),validation=validation,
            previous_failure_sha256=sha(self.out/'RUN.failure.json'),
            failed_timeline_archive='timeline-failed-missing-pynvml',
            failed_timeline_sha256={p.relative_to(self.out).as_posix():sha(p)
                                   for p in (self.out/'timeline-failed-missing-pynvml').rglob('*') if p.is_file()},
            original_driver_sha256={p.relative_to(self.out/'driver-sources').as_posix():sha(p)
                                    for p in (self.out/'driver-sources').rglob('*') if p.is_file()},
            resumed_driver_sha256={p.name:sha(p) for p in snapshots.iterdir()},
            deps_path=str(deps),deps_sha256={p.relative_to(deps).as_posix():sha(p) for p in deps.rglob('*')
                                          if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'},
            pythonpath=self.env['PYTHONPATH'],complete=False)
        write(self.resume_dir/'RESUME.json',evidence)
        self.evidence = evidence
        write(self.out/'RUN.json',self.result)

    def execute(self):
        try:
            target = self.resume_dir/'dependencies.json'
            code = ('import ctypes,hashlib,importlib.metadata as m,json,sys; from pathlib import Path; '
                    'import pynvml,psutil; pynvml.nvmlInit(); '
                    'h=pynvml.nvmlDeviceGetHandleByUUID(sys.argv[2]); '
                    'ctypes.CDLL(sys.argv[3]); '
                    'v={"python":sys.version,"pynvml_origin":pynvml.__file__,"pynvml_sha256":hashlib.sha256(Path(pynvml.__file__).read_bytes()).hexdigest(),'
                    '"nvidia_ml_py":m.version("nvidia-ml-py"),"psutil":psutil.__version__,"psutil_origin":psutil.__file__,'
                    '"gpu_uuid":pynvml.nvmlDeviceGetUUID(h),"driver":pynvml.nvmlSystemGetDriverVersion(),'
                    '"nvtx_library":str(Path(sys.argv[3]).resolve()),"nvtx_sha256":hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()}; '
                    'Path(sys.argv[1]).write_text(json.dumps(v,indent=2)+"\\n"); pynvml.nvmlShutdown()')
            self.call('resume-dependency-preflight',[self.a.python,'-c',code,str(target),self.a.gpu_uuid,str(self.a.nvtx_library)],60,
                      self.resume_dir/'dependency-preflight.log')
            require(read(target)['gpu_uuid'] == self.a.gpu_uuid,'dependency preflight selected different GPU')
            self.timeline(); self.workflows(); self.download()
            self.result.update(complete=True,loaded_native=self.loaded)
            self.evidence.update(complete=True,dependencies_sha256=sha(target),dependencies=read(target))
        except BaseException as error:
            self.result['error'] = f'{type(error).__name__}: {error}'
            self.evidence['error'] = self.result['error']
            raise
        finally:
            self.result.update(ended_utc=utc(),elapsed_seconds=time.monotonic()-self.started)
            self.evidence.update(ended_utc=utc(),elapsed_seconds=self.result['elapsed_seconds'])
            write(self.resume_dir/'RESUME.json',self.evidence)
            write(self.out/'RUN.json',self.result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--deps',type=Path,required=True)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--root',type=Path)
    p.add_argument('--revision')
    p.add_argument('--cache',type=Path)
    p.add_argument('--resume-name',default='resume-1')
    p.add_argument('--total-timeout',type=float,default=7200)
    def interrupted(signum,frame): raise InterruptedError(f'resume received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    Resume(p.parse_args()).execute()


if __name__=='__main__':
    main()

# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Private publication orchestration; no CUDA imports."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from datetime import datetime, timezone


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_inventory(root):
    package = root / 'src/cuda_zlib'
    return {p.relative_to(package).as_posix(): sha(p) for p in sorted(package.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.cu', '.cuh')}


def freeze(root, revision, expected=None):
    require(re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', revision), 'full revision required')
    paths = list((root / 'benchmarks').glob('*.py'))
    paths += [root / 'src/cuda_zlib' / p for p in source_inventory(root)]
    hashes = {p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)}
    value = {'source_revision': revision, 'files_sha256': hashes,
             'source_sha256': source_inventory(root)}
    if expected is not None:
        require(value == expected, 'source/harness/extractor files changed from frozen manifest')
    else:
        # A Git checkout establishes the pin; a Git-less archive must carry that manifest.
        for name, digest in hashes.items():
            blob = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=root)
            require(hashlib.sha256(blob).hexdigest() == digest, f'file differs from revision: {name}')
    return value


NATIVE_FIELDS = ('cache_key', 'library_sha256', 'build_sha256', 'identity')


def native_receipt(value, prepare_only=False):
    """Accept a canonical receipt or the archived candidate's actual build proof."""
    if 'native' not in value:
        return value
    proof = value['native']
    library = Path(proof['path'])
    build = library.with_name('build.json')
    identity = proof['build']
    if not prepare_only:
        require(sha(library) == proof['sha256'], 'candidate library changed')
        require(read(build) == identity, 'candidate build identity changed')
    return {'source_sha256': {k.removeprefix('src/cuda_zlib/'): v
                              for k, v in value['source_sha256'].items()},
            'cache_key': library.parent.name, 'library_path': str(library),
            'library_sha256': proof['sha256'], 'identity': identity,
            'build_sha256': sha(build) if build.exists() else None}


def check_native(actual, expected):
    for field in NATIVE_FIELDS:
        require(actual[field] == expected[field], f'native {field} changed')
    require(hashlib.sha256(json.dumps(actual['identity'], sort_keys=True).encode()).hexdigest()
            == actual['cache_key'], 'invalid native build identity')


def stop_owned(process):
    """Also stop descendants that opened sessions, without touching other jobs."""
    import psutil
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    _, alive = psutil.wait_procs(descendants, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run(command, cwd, env, log, timeout):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {'command': list(map(str, command)), 'cwd': str(cwd), 'log': str(log),
              'started_utc': utc(), 'timeout_seconds': timeout, 'complete': False}
    receipt = log.with_suffix('.launch.json')
    with log.open('w') as output:
        process = None
        try:
            process = subprocess.Popen(record['command'], cwd=cwd, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record.update(pid=process.pid, pgid=process.pid, stop=f'kill -TERM -- -{process.pid}')
            write(receipt, record)
            print(json.dumps(record), flush=True)
            code = process.wait(timeout=timeout)
            record['exit_code'] = code
            require(code == 0, f'command exited {code}: {log}')
            record['complete'] = True
        except BaseException as error:
            if process is not None:
                stop_owned(process)
            record.update(error=f'{type(error).__name__}: {error}', cleanup_complete=process is not None)
            raise
        finally:
            record.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - started)
            write(receipt, record)
    return record

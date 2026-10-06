# Copyright (c) 2026 xangma
# SPDX-License-Identifier: MIT
"""Lazy native build and typed JAX FFI registration."""

import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

from ._errors import BackendUnavailable

_BUILD_LOCK = threading.Lock()


def _nvcc():
    configured = os.environ.get("CUDACXX")
    if configured:
        path = shutil.which(configured)
    else:
        root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        path = shutil.which(str(Path(root) / "bin/nvcc")) if root else None
        path = path or shutil.which("nvcc") or shutil.which("/usr/local/cuda/bin/nvcc")
    if not path:
        raise BackendUnavailable("JAX FFI requires the CUDA toolkit compiler nvcc; set CUDACXX or CUDA_HOME")
    return path


def _compile_library(architecture):
    import jax
    import jaxlib
    from ._encode_kernels import CUDA_SOURCE as encoder
    from ._decode_kernels import CUDA_SOURCE as decoder
    from ._postprocess import CUDA_SOURCE as checksum

    if not architecture.startswith("sm_") or not architecture[3:].isdigit():
        raise BackendUnavailable("invalid CUDA architecture")
    nvcc = _nvcc()
    include = Path(jax.ffi.include_dir())
    native = Path(__file__).parent / "native" / "codec_ffi.cu"
    sources = {"codec_ffi.cu": native.read_text(), "encoder.cuh": encoder,
               "decoder.cuh": decoder, "postprocess.cuh": checksum}
    try:
        version = subprocess.check_output([nvcc, "--version"], text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackendUnavailable(f"could not run CUDA compiler: {exc}") from exc
    identity = {"architecture": architecture, "jaxlib": jaxlib.__version__,
                "compiler": str(Path(nvcc).resolve()), "compiler_version": version,
                "sources": {k: hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},
                "headers": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((include / "xla/ffi/api").glob("*.h"))},
                "flags": ["-O3", "-lineinfo", "--std=c++17", "--cudart=shared"]}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get("CUDA_ZLIB_CACHE_DIR", str(
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "cuda-zlib")))
    destination = cache / key
    library = destination / "codec_ffi.so"
    cache.mkdir(parents=True, exist_ok=True)
    # File locks serialize independent Python/MPI processes, as well as threads.
    import fcntl
    with (cache / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if library.is_file() and (destination / "build.json").is_file():
            return library
        with tempfile.TemporaryDirectory(prefix=key + "-", dir=cache) as directory:
            build = Path(directory)
            for name, source in sources.items():
                (build / name).write_text(source)
            toolkit = Path(nvcc).resolve().parent.parent
            command = [nvcc, "-shared", "-Xcompiler=-fPIC", "-O3", "-lineinfo",
                       "--std=c++17", "--cudart=shared", f"-arch={architecture}",
                       f"-I{include}", str(build / "codec_ffi.cu"), "-o", str(build / "codec_ffi.so"),
                       f"-Xlinker=-rpath={toolkit / 'lib64'}"]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.SubprocessError) as exc:
                raise BackendUnavailable(f"native CUDA build failed: {exc}") from exc
            if result.returncode:
                raise BackendUnavailable("native CUDA build failed:\n" + (result.stdout + result.stderr)[-12000:])
            destination.mkdir(parents=True, exist_ok=True)
            (build / "build.json").write_text(json.dumps(identity, indent=2) + "\n")
            os.replace(build / "codec_ffi.so", library)
            os.replace(build / "build.json", destination / "build.json")
    return library


@functools.lru_cache(maxsize=None)
def _load_library_cached(architecture):
    import jax
    path = _compile_library(architecture)
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise BackendUnavailable(f"could not load native CUDA library: {exc}") from exc
    names = tuple(f"cuda_zlib_{operation}_{architecture}" for operation in ("compress", "decompress"))
    for name, symbol in zip(names, ("CudaZlibCompress", "CudaZlibDecompress")):
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(library, symbol)), platform="CUDA")
    return library, names


def _load_library(architecture):
    with _BUILD_LOCK:
        return _load_library_cached(architecture)


@functools.lru_cache(maxsize=None)
def load_backend(device):
    # Query architecture through the CUDA runtime; no GPU array library needed.
    try:
        import jax
        # The device description is not an architecture identifier. Use CUDA's
        # driver attributes, without creating a private context or stream.
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        ordinal = int(device.local_hardware_id)
        handle, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        code = driver.cuInit(0)
        if not code:
            code = driver.cuDeviceGet(ctypes.byref(handle), ordinal)
        if code:
            raise BackendUnavailable(f"CUDA driver device query failed ({code})")
        for attribute, value in ((75, major), (76, minor)):
            code = driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, handle.value)
            if code:
                raise BackendUnavailable(f"CUDA architecture query failed ({code})")
        return _load_library(f"sm_{major.value}{minor.value}")[1]
    except (ImportError, OSError) as exc:
        raise BackendUnavailable(f"native CUDA FFI unavailable: {exc}") from exc

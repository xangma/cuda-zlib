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


def _workspace_retention():
    value = os.environ.get("CUDA_ZLIB_WORKSPACE_RETENTION_BYTES", str(1 << 30))
    if not value.isascii() or not value.isdecimal() or int(value) >= 1 << 64:
        raise ValueError("CUDA_ZLIB_WORKSPACE_RETENTION_BYTES must be an integer in [0, 2**64-1]")
    return int(value)


def _configure_workspace_pool(library):
    library.CudaZlibSetWorkspaceRetention.argtypes = [ctypes.c_uint64]
    library.CudaZlibSetWorkspaceRetention.restype = ctypes.c_int
    library.CudaZlibWorkspacePoolStats.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    library.CudaZlibWorkspacePoolStats.restype = ctypes.c_int
    library.CudaZlibTrimWorkspacePool.argtypes = [ctypes.c_int]
    library.CudaZlibTrimWorkspacePool.restype = ctypes.c_int
    _check_pool_error(library.CudaZlibSetWorkspaceRetention(_workspace_retention()))


def _check_pool_error(code):
    if code:
        raise BackendUnavailable(f"native CUDA workspace pool operation failed (CUDA error {code})")


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
    for name in ("batch_encode.cuh", "batch_decode.cuh"):
        sources[name] = (native.parent / name).read_text()
    try:
        version = subprocess.check_output([nvcc, "--version"], text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackendUnavailable(f"could not run CUDA compiler: {exc}") from exc
    identity = {"architecture": architecture, "jaxlib": jaxlib.__version__,
                "compiler": str(Path(nvcc).resolve()), "compiler_version": version,
                "sources": {k: hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},
                "headers": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((include / "xla/ffi/api").glob("*.h"))},
                "flags": ["-O3", "-lineinfo", "--std=c++17", "--cudart=shared",
                          "-Xcompiler=-pthread", "-ldl"]}
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
                       f"-Xlinker=-rpath={toolkit / 'lib64'}", "-Xcompiler=-pthread", "-ldl"]
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
    _workspace_retention()  # Reject invalid configuration before compiling.
    path = _compile_library(architecture)
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise BackendUnavailable(f"could not load native CUDA library: {exc}") from exc
    _configure_workspace_pool(library)
    operations = ("compress", "decompress", "compress_batch", "decompress_batch")
    names = tuple(f"cuda_zlib_{operation}_{architecture}" for operation in operations)
    for name, symbol in zip(names, ("CudaZlibCompress", "CudaZlibDecompress",
                                   "CudaZlibCompressBatch", "CudaZlibDecompressBatch")):
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(library, symbol)), platform="CUDA")
    return library, names


def _load_library(architecture):
    with _BUILD_LOCK:
        return _load_library_cached(architecture)


@functools.lru_cache(maxsize=None)
def _backend(device):
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
        return _load_library(f"sm_{major.value}{minor.value}")
    except (ImportError, OSError) as exc:
        raise BackendUnavailable(f"native CUDA FFI unavailable: {exc}") from exc


def load_backend(device):
    return _backend(device)[1][:2]


def load_batch_backend(device):
    return _backend(device)[1][2:]


def workspace_pool_stats(device):
    library = _backend(device)[0]
    values = (ctypes.c_uint64 * 4)()
    _check_pool_error(library.CudaZlibWorkspacePoolStats(int(device.local_hardware_id), values))
    return dict(zip(("retention_bytes", "reserved_bytes", "used_bytes", "pool_count"), values))


def trim_workspace_pool(device):
    library = _backend(device)[0]
    _check_pool_error(library.CudaZlibTrimWorkspacePool(int(device.local_hardware_id)))

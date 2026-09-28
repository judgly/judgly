"""Loading of libjudgly, the native library bundled in the package directory, and its C API."""

import ctypes
import sys
from functools import cache
from importlib.resources import as_file, files

class JudglyError(RuntimeError):
    """The native library could not be loaded, rejected a request or could not answer it."""


_LIBNAME = {"darwin": "libjudgly.dylib", "win32": "judgly.dll"}.get(sys.platform, "libjudgly.so")


@cache
def _lib() -> ctypes.CDLL:
    resource = files("judgly") / _LIBNAME
    if not resource.is_file():
        raise JudglyError(f"judgly: native library {_LIBNAME} not found in the package; "
                          "reinstall judgly from a wheel or build it from source")
    try:
        with as_file(resource) as path:
            lib = ctypes.CDLL(str(path))
    except OSError as e:
        raise JudglyError(f"judgly: cannot load the native library {_LIBNAME}: {e}") from e
    lib.judgly_native_version.restype = ctypes.c_char_p
    lib.judgly_native_version.argtypes = []
    lib.judgly_open.restype = ctypes.c_void_p
    lib.judgly_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.judgly_decide.restype = ctypes.c_void_p  # a malloc'd string, freed by us
    lib.judgly_decide.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.judgly_free_string.restype = None
    lib.judgly_free_string.argtypes = [ctypes.c_void_p]
    lib.judgly_close.restype = None
    lib.judgly_close.argtypes = [ctypes.c_void_p]
    f32, f64 = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)
    lib.judgly_test_head_loss.restype = ctypes.c_double
    lib.judgly_test_head_loss.argtypes = [f64, f64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, f32, f32, f32, f32,
                                          ctypes.POINTER(ctypes.c_uint8), ctypes.c_double]
    lib.judgly_test_n_rotations.restype = ctypes.c_int
    lib.judgly_test_n_rotations.argtypes = [ctypes.c_int] * 4
    lib.judgly_test_theta_clamp.restype = ctypes.c_double
    lib.judgly_test_theta_clamp.argtypes = [ctypes.c_double]
    lib.judgly_test_h2_start_theta.restype = ctypes.c_double
    lib.judgly_test_h2_start_theta.argtypes = [ctypes.c_double]
    lib.judgly_test_head_prob_sd.restype = ctypes.c_double
    lib.judgly_test_head_prob_sd.argtypes = [f64, ctypes.c_int, ctypes.c_int, ctypes.c_int, f32, f32,
                                             f32, ctypes.POINTER(ctypes.c_uint8)]
    lib.judgly_test_head_load.restype = ctypes.c_int
    lib.judgly_test_head_load.argtypes = [ctypes.c_char_p]
    return lib


def native_version() -> str:
    """Build information of the native library as a JSON string.

    Fields: judgly version, the llama.cpp commit and ggml version compiled in, and the ggml
    backends and devices available on this machine.
    """
    raw = _lib().judgly_native_version()
    if raw is None:
        raise JudglyError("judgly: native library could not report its version")
    return raw.decode("utf-8")


def _take_string(ptr: int | None) -> str | None:
    """The text of a string the library allocated, which is then released."""
    if not ptr:
        return None
    try:
        return ctypes.string_at(ptr).decode("utf-8")
    finally:
        _lib().judgly_free_string(ptr)


def open_handle(config_json: str) -> int:
    """judgly_open. Returns the handle; raises JudglyError with the library's message."""
    err = ctypes.c_void_p()
    handle = _lib().judgly_open(config_json.encode("utf-8"), ctypes.byref(err))
    if not handle:
        raise JudglyError(_take_string(err.value) or "judgly_open failed")
    return handle


def decide(handle: int, request_json: str) -> str:
    """judgly_decide: the response JSON, which may be {"error": ...}."""
    text = _take_string(_lib().judgly_decide(handle, request_json.encode("utf-8")))
    if text is None:
        raise MemoryError("judgly_decide: out of memory")
    return text


def close_handle(handle: int) -> None:
    _lib().judgly_close(handle)

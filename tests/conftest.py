"""
conftest.py

Set thread / CPU limits early so test modules don't need to set env vars themselves.
This file is imported by pytest before any test modules, so environment variables
and runtime threadpool limits will be applied before libraries like numpy/MKL/OpenBLAS
are imported by tests.
"""

import os
import warnings

# 1) set environment variables early (only if not already set)
for env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_var, "2")

# 2) attempt to apply runtime threadpool limits where possible
_tpctl_ctx = None

try:
    # threadpoolctl is the most robust runtime API for limiting BLAS/OpenMP pools
    from threadpoolctl import threadpool_limits

    # enter the context so the limits are applied for the lifetime of the test process
    _tpctl_ctx = threadpool_limits(limits=2)
    _tpctl_ctx.__enter__()

except Exception:
    # fallback to specific libraries where Python bindings exist
    try:
        import mkl

        try:
            mkl.set_num_threads(2)
        except Exception:
            pass
    except Exception:
        pass

    try:
        import numexpr

        try:
            numexpr.set_num_threads(2)
        except Exception:
            pass
    except Exception:
        pass


# 3) best-effort: pin process to two logical CPUs (0 and 1) on Linux
def _pin_affinity(cores=(0, 1)):
    try:
        # preferred Linux API
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cores))
            return True
    except Exception:
        pass

    try:
        import psutil

        p = psutil.Process()
        p.cpu_affinity(list(cores))
        return True
    except Exception:
        pass

    return False


# apply affinity but don't fail if not supported
_pin_affinity((0, 1))


def pytest_sessionstart(session):
    """pytest hook: run at session start. Warn if thread limits may not be applied."""
    # simple checks that at least the env var is set
    try:
        if os.environ.get("OMP_NUM_THREADS") != "2":
            warnings.warn("OMP_NUM_THREADS not set to 2 (conftest tried to set it)")
    except Exception:
        pass


def pytest_unconfigure(config):
    """pytest hook: clean up threadpoolctl context if we entered it."""
    global _tpctl_ctx
    if _tpctl_ctx is not None:
        try:
            _tpctl_ctx.__exit__(None, None, None)
        except Exception:
            pass

"""Simple environment sanity checks.

These lightweight tests validate that test-time environment variables and
runtime configuration are suitable for running the small integration
workloads (for example, limiting thread pools to reduce contention in CI).
"""

import os


def test_env_vars():
    """Ensure host environment variables are set to sane defaults.

    This test checks for the presence (or intentional absence) of the
    ``OMP_NUM_THREADS`` variable which the test-suite attempts to set via
    ``conftest.py``.
    """
    assert os.environ.get("OMP_NUM_THREADS") in ("2", None)

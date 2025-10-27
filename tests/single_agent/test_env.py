import os


def test_env_vars():
    assert os.environ.get("OMP_NUM_THREADS") in ("2", None)

from portrisk import core
from portrisk.core import black_scholes, utils


def test_top_level_imports():
    assert utils is not None
    assert core is not None
    assert black_scholes is not None

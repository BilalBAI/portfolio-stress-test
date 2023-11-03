from portrisk import utils, core, black_scholes


def test_top_level_imports():
    assert utils is not None
    assert core is not None
    assert black_scholes is not None

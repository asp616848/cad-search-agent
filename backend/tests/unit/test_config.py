"""Sanity checks for config — no model/IO needed."""


def test_defaults():
    from app.config import DUP_THRESHOLD, GEO_WEIGHT, TEXT_WEIGHT, TOP_K, WEAK_THRESHOLD

    assert abs(GEO_WEIGHT + TEXT_WEIGHT - 1.0) < 1e-6
    assert TOP_K > 0
    assert 0 < DUP_THRESHOLD <= 1.0
    assert 0 < WEAK_THRESHOLD < DUP_THRESHOLD


def test_env_override(monkeypatch):
    monkeypatch.setenv("GEO_WEIGHT", "0.6")
    monkeypatch.setenv("TEXT_WEIGHT", "0.4")
    import importlib

    import app.config as cfg

    importlib.reload(cfg)
    assert cfg.GEO_WEIGHT == 0.6
    assert cfg.TEXT_WEIGHT == 0.4
    importlib.reload(cfg)  # restore defaults for other tests

"""Phase 9 gate — LLM adapter template fallback and cache."""

import os


def _explain(provider="none", key=""):
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_API_KEY"] = key
    # re-import to pick up env changes
    import importlib

    import app.core.llm_adapter as mod

    importlib.reload(mod)
    return mod.explain


def test_template_fallback_no_key():
    explain = _explain(provider="none")
    txt = explain(
        query_meta={},
        result_meta={"name": "bracket-001", "material": "Aluminum", "process": "CNC Milling"},
        scores={"geo": 0.82, "text": 0.41},
    )
    assert "Geometry match 82%" in txt
    assert len(txt) > 0


def test_template_contains_name():
    explain = _explain(provider="none")
    txt = explain(
        query_meta={},
        result_meta={"name": "shaft-007", "material": "Steel", "process": "Turning"},
        scores={"geo": 0.65, "text": 0.30},
    )
    assert "shaft-007" in txt


def test_cache_hit_no_second_call(tmp_path, monkeypatch):
    import app.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "LLM_PROVIDER", "none")
    monkeypatch.setattr(cfg, "LLM_API_KEY", "")

    import importlib

    import app.core.llm_adapter as mod

    mod._conn = None  # force fresh connection
    importlib.reload(mod)

    q = {"name": "query"}
    r = {"name": "result", "material": "Al", "process": "Mill"}
    s = {"geo": 0.9, "text": 0.5}

    txt1 = mod.explain(q, r, s)
    txt2 = mod.explain(q, r, s)
    assert txt1 == txt2  # same text returned from cache


def test_zero_scores_template():
    explain = _explain(provider="none")
    txt = explain(
        query_meta={},
        result_meta={"name": "plate", "material": "", "process": ""},
        scores={"geo": 0.0, "text": 0.0},
    )
    assert "Geometry match 0%" in txt

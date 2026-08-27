"""Unit tests for occ_stats.compute()."""

from pathlib import Path

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def test_occ_stats_ranges() -> None:
    from app.core.occ_stats import compute
    from app.occwl.io import load_shell

    solids = load_shell(str(FIXTURE_DIR / "partA.step"))
    assert solids, "no solids loaded from partA.step"
    shape = solids[0].topods_shape()
    s = compute(shape)

    assert s["face_count"] > 0
    assert s["edge_count"] > 0
    assert s["volume"] > 0
    assert s["surface_area"] > 0
    assert s["bbox_x"] > 0 and s["bbox_y"] > 0 and s["bbox_z"] > 0
    assert 0 < s["solidity"] <= 1.0


def test_occ_stats_keys() -> None:
    from app.core.occ_stats import compute
    from app.occwl.io import load_shell

    solids = load_shell(str(FIXTURE_DIR / "partB.step"))
    shape = solids[0].topods_shape()
    s = compute(shape)
    expected = {
        "face_count",
        "edge_count",
        "volume",
        "surface_area",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "solidity",
    }
    assert expected == set(s.keys())

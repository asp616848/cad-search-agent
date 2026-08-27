"""Select a diverse ~30-part demo library from the MFCAD dataset.

MFCAD filenames encode a feature-type signature: files sharing every segment
except the second-to-last ("instance") digit are random re-instantiations of
the SAME feature combination — i.e. natural near-duplicate candidates.

This script:
  1. Groups MFCAD files by feature signature.
  2. Picks one signature group as the near-duplicate pair.
  3. Samples diverse candidates spread across different signatures/complexity.
  4. Runs each candidate through OUR embedder/occ_stats to verify it loads,
     confirm the near-dup pair actually scores >= 0.90 cosine, and get real
     histograms for metadata generation.
  5. Copies selected files to data/demo_steps/ and writes demo_metadata.json.

Usage:
  python scripts/select_demo_library.py --mfcad_dir /path/to/mfcad/dataset/step \
    --out_dir data/demo_steps --meta_out data/demo_metadata.json --n 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.text_embedder import histogram_to_text  # noqa: E402

_SIG_RE = re.compile(r"^(.*)-(\d+)-(\d+)$")

_MATERIALS = [
    ("Aluminum 6061", "CNC Milling", "Acme Machining"),
    ("Stainless Steel 316", "CNC Milling", "Precision Parts Co"),
    ("Titanium Grade 5", "CNC Milling", "Acme Machining"),
    ("Mild Steel 1018", "CNC Milling", "Midwest Fab"),
    ("Aluminum 7075", "CNC Milling", "Precision Parts Co"),
]

_KNOWN_ISSUE_POOL = [
    "thin wall near pocket caused chatter at high feed rate",
    "tight tolerance on mounting face required secondary op",
    "burr on through-slot edge needed deburring pass",
    "",  # most parts have no issue
    "",
    "",
]


def _signature(stem: str) -> tuple[str, int] | None:
    """Split filename stem into (signature, instance) where signature is
    everything except the instance digit. Returns None if pattern doesn't match."""
    m = _SIG_RE.match(stem)
    if not m:
        return None
    prefix, instance, class_id = m.groups()
    return f"{prefix}-{class_id}", int(instance)


def _det_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def _auto_name(histogram: dict, idx: int) -> str:
    top_features = sorted(histogram.items(), key=lambda kv: -kv[1])[:2]
    if top_features:
        feat_str = " + ".join(f"{v}x {k.replace('_', ' ')}" for k, v in top_features)
        return f"Machined Block {idx:02d} ({feat_str})"
    return f"Machined Block {idx:02d}"


def _auto_metadata(
    name: str, stem: str, histogram: dict, stats: dict, idx: int
) -> dict:
    h = _det_hash(stem)
    material, process, supplier = _MATERIALS[h % len(_MATERIALS)]
    known_issue = _KNOWN_ISSUE_POOL[h % len(_KNOWN_ISSUE_POOL)]
    base_cost = 150 + (stats.get("face_count", 10) * 12) + (h % 300)
    feat_tokens = histogram_to_text(histogram)

    return {
        "filename": f"{stem}.step",
        "name": name,
        "material": material,
        "process": process,
        "cost": round(base_cost, -1),
        "supplier": supplier,
        "notes": (
            f"Machined features: {feat_tokens}"
            if feat_tokens
            else "Simple block geometry"
        ),
        "known_issues": known_issue,
        "ppap_notes": "PPAP Level 2 approved" if idx % 4 == 0 else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfcad_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--meta_out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--candidate_pool", type=int, default=120)
    args = ap.parse_args()

    step_files = sorted(args.mfcad_dir.glob("*.step"))
    print(f"Found {len(step_files)} STEP files in {args.mfcad_dir}")

    # Group by signature
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in step_files:
        sig_info = _signature(f.stem)
        if sig_info is None:
            continue
        sig, _instance = sig_info
        groups[sig].append(f)

    # Near-dup candidate: a signature group with >= 2 members
    dup_groups = [g for g in groups.values() if len(g) >= 2]
    dup_groups.sort(key=lambda g: g[0].stem)  # deterministic order
    if not dup_groups:
        print("ERROR: no near-duplicate signature groups found", file=sys.stderr)
        sys.exit(1)

    # Diverse candidates: one file per distinct signature, spread across the list
    diverse_sigs = sorted(groups.keys())
    step_pick = max(1, len(diverse_sigs) // args.candidate_pool)
    diverse_candidates = [groups[sig][0] for sig in diverse_sigs[::step_pick]]

    print(f"Signature groups: {len(groups)}  Dup-eligible groups: {len(dup_groups)}")
    print(f"Diverse candidate pool: {len(diverse_candidates)}")

    from app.core.graph_builder import step_to_dgl_graph
    from app.core.occ_stats import compute as occ_stats
    from app.core.uvnet_embedder import embed
    from app.occwl.io import load_shell

    def try_embed(path: Path):
        graph = step_to_dgl_graph(path)
        geo_vec, histogram = embed(graph)
        solids = load_shell(str(path))
        stats = occ_stats(solids[0].topods_shape()) if solids else {}
        return geo_vec, histogram, stats

    import torch.nn.functional as F

    # 1. Find a verified near-duplicate pair (cosine >= 0.90)
    near_dup_pair = None
    near_dup_score = 0.0
    for group in dup_groups:
        a, b = group[0], group[1]
        try:
            va, hist_a, stats_a = try_embed(a)
            vb, hist_b, stats_b = try_embed(b)
        except Exception as e:
            print(f"  skip dup candidate {a.stem}: {e}")
            continue
        sim = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
        if sim >= 0.90:
            near_dup_pair = [(a, hist_a, stats_a), (b, hist_b, stats_b)]
            near_dup_score = sim
            print(f"  Near-dup pair found: {a.stem} / {b.stem}  cosine={sim:.4f}")
            break
        print(f"  {a.stem}/{b.stem} cosine={sim:.4f} < 0.90, trying next group")

    if near_dup_pair is None:
        print("ERROR: could not find a verified near-duplicate pair", file=sys.stderr)
        sys.exit(1)

    # 2. Fill remaining slots with diverse candidates, skipping failures and dup files
    used_stems = {p.stem for p, _, _ in near_dup_pair}
    selected: list[tuple[Path, dict, dict]] = list(near_dup_pair)

    for cand in diverse_candidates:
        if len(selected) >= args.n:
            break
        if cand.stem in used_stems:
            continue
        try:
            _vec, hist, stats = try_embed(cand)
        except Exception as e:
            print(f"  skip {cand.stem}: {e}")
            continue
        selected.append((cand, hist, stats))
        used_stems.add(cand.stem)
        print(f"  [{len(selected)}/{args.n}] added {cand.stem}")

    if len(selected) < args.n:
        print(f"WARNING: only found {len(selected)}/{args.n} usable files")

    # 3. Copy files + write metadata
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for idx, (path, hist, stats) in enumerate(selected, start=1):
        dest = args.out_dir / path.name
        shutil.copy2(path, dest)
        name = _auto_name(hist, idx)
        metadata.append(_auto_metadata(name, path.stem, hist, stats, idx))

    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.meta_out, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Demo library selected ===")
    print(f"Total parts: {len(selected)}")
    print(
        f"Near-duplicate pair: {near_dup_pair[0][0].stem} <-> {near_dup_pair[1][0].stem}  (cosine={near_dup_score:.4f})"
    )
    print(f"Copied to: {args.out_dir}")
    print(f"Metadata:  {args.meta_out}")


if __name__ == "__main__":
    main()

# Copyright 2026 blucca
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the GENCO OPF tutorial's checkpoint/split overlap from metadata."""

import argparse
import csv
import json
from pathlib import Path
import random
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS_URL = (
    "https://huggingface.co/gridfm/genco-opf-datakit/resolve/"
    "2d2401572f6a2d0c1be79841b2def99cf45e1639/"
    "mlflow/train/case118_ieee/small/seed0/"
    "465594abc8eb4c96b908363afc84a0e1/"
    "artifacts/stats/case118_ieee_scenario_splits.json"
)


def audit(paper):
    with (ROOT / "scenario-load-map.csv").open() as f:
        rows = list(csv.DictReader(f))
    groups = {int(r["scenario"]): float(r["load_scenario_idx"]) for r in rows}
    assert len(rows) == len(groups) == 1000
    assert sorted(groups) == list(range(1000))
    assert all(np.isfinite(list(groups.values())))
    known = {k: set(paper[k]) for k in ("train", "val", "test")}
    assert not any(known[a] & known[b] for a, b in
                   (("train", "val"), ("train", "test"), ("val", "test")))
    assert set(groups) <= set.union(*known.values())

    # Source transcription: datamodule's shuffle/subset, then utils.py's
    # torch.unique(sorted=True), seeded NumPy permutation, and group membership.
    order = list(range(1000))
    random.Random(0).shuffle(order)
    loads = np.array([groups[s] for s in order])
    unique = np.random.RandomState(0).permutation(np.unique(loads))
    count = int(0.1 * len(unique))
    test_groups = unique[len(unique) - count:]
    test = [order[i] for i in np.flatnonzero(np.isin(loads, test_groups))]
    overlap = {k: sorted(set(test) & ids) for k, ids in known.items()}
    safe = sorted(set(groups) & known["test"])
    return {
        "tutorial_test_count": len(test),
        "overlap_counts": {k: len(v) for k, v in overlap.items()},
        "tutorial_test_ids": test,
        "overlap_ids": overlap,
        "local_official_test_count": len(safe),
        "local_official_test_ids": safe,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path,
                        help="Use an already downloaded official split JSON")
    args = parser.parse_args()
    if args.split_json:
        paper = json.loads(args.split_json.read_text())
    else:
        with urllib.request.urlopen(SPLITS_URL, timeout=30) as response:
            paper = json.load(response)
    result = audit(paper)
    # A single fixed-corpus check makes an accidental change to the split visible.
    assert result["tutorial_test_count"] == 71
    assert result["overlap_counts"] == {"train": 48, "val": 0, "test": 23}
    assert result["local_official_test_count"] == 93
    print(json.dumps(result, indent=2))

# Copyright 2026 blucca
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned GENCO tutorial and saved-holdout evaluations on a CPU."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from audit import ROOT, SPLITS_URL, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphkit", type=Path, required=True,
                        help="Checkout at the revision in provenance.json")
    parser.add_argument("--work", type=Path, required=True,
                        help="Scratch/output directory; use a temp directory")
    args = parser.parse_args()
    work, upstream = args.work.resolve(), args.graphkit.resolve()
    work.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((ROOT / "provenance.json").read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != provenance["graphkit_revision"]:
        raise ValueError(f"Use the pinned graphkit revision, got {revision}")

    import torch
    import yaml
    from huggingface_hub import hf_hub_download, snapshot_download

    torch.set_num_threads(2)
    sys.path.insert(0, str(upstream / "examples" / "notebooks"))
    from tutorial_opf_helpers import write_runtime_hive_from_bus

    with urllib.request.urlopen(SPLITS_URL, timeout=30) as response:
        paper = json.load(response)
    observed = audit(paper)
    assert observed["overlap_counts"] == {"train": 48, "val": 0, "test": 23}
    (work / "audit.json").write_text(json.dumps(observed, indent=2) + "\n")

    # This corpus's raw scenario IDs are contiguous dataset indices 0..999.
    # Preserve saved membership; never re-split a pretrained checkpoint's cases.
    local_ids = set(range(1000))
    safe = {name: sorted(local_ids & set(paper[name]))
            for name in ("train", "val", "test")}
    assert safe["test"] == observed["local_official_test_ids"]
    assert not set(safe["test"]) & (set(paper["train"]) | set(paper["val"]))
    split_dir = work / "saved-subset"
    split_dir.mkdir(exist_ok=True)
    for name, ids in safe.items():
        torch.save(torch.tensor(ids, dtype=torch.long), split_dir / f"{name}.pt")
    (split_dir / "scenario_splits.json").write_text(
        json.dumps(safe, indent=2) + "\n")

    raw = work / "data" / "case118_ieee" / "raw"
    snapshot_download(
        repo_id=provenance["dataset"], repo_type="dataset",
        revision=provenance["dataset_revision"], local_dir=str(raw),
        allow_patterns=[f"{table}_data.parquet/scenario_partition={p}/*"
                        for table in ("bus", "gen", "branch") for p in range(5)],
        max_workers=2,
    )
    write_runtime_hive_from_bus(raw)
    checkpoint = hf_hub_download(
        repo_id=provenance["model"], revision=provenance["model_revision"],
        filename=provenance["checkpoint"], local_dir=str(work / "weights"))
    normalizer = hf_hub_download(
        repo_id=provenance["model"], revision=provenance["model_revision"],
        filename="case118_ieee/small/seed0/normalizer_stats.pt",
        local_dir=str(work / "weights"))

    config_path = upstream / "examples" / "config" / "HGNS_OPF_case118_small.yaml"
    receipt = {"provenance": provenance, "torch": torch.__version__,
               "device": "cpu", "batch_size": 8, "runs": {}}
    for name in ("tutorial", "saved-holdout"):
        config = yaml.safe_load(config_path.read_text())
        config["training"]["accelerator"] = "cpu"
        config["training"]["devices"] = 1
        config["data"]["workers"] = 0
        expected = observed["tutorial_test_ids"]
        if name == "saved-holdout":
            config["data"]["split_by_load_scenario_idx"] = False
            config["data"]["split_from_existing_files"] = str(split_dir)
            expected = safe["test"]
        run_dir = work / name
        run_dir.mkdir(exist_ok=True)
        yaml_path = run_dir / "config.yaml"
        yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
        start = time.monotonic()
        with (run_dir / "evaluation.log").open("w") as log:
            subprocess.run([
                sys.executable, "-m", "gridfm_graphkit", "evaluate",
                "--config", str(yaml_path), "--data_path", str(work / "data"),
                "--model_path", checkpoint, "--normalizer_stats", normalizer,
                "--num_workers", "0", "--batch_size", "8",
                "--log_dir", str(run_dir / "mlruns"), "--exp_name", name,
            ], stdout=log, stderr=subprocess.STDOUT, check=True)
        files = list((run_dir / "mlruns").rglob("case118_ieee_scenario_splits.json"))
        if len(files) != 1:
            raise ValueError(f"Expected one native split receipt, got {files}")
        native = json.loads(files[0].read_text())
        assert sorted(native["test"]) == sorted(expected)
        metrics = {}
        for path in (run_dir / "mlruns").rglob("case118_ieee_*.csv"):
            shutil.copyfile(path, run_dir / path.name)
            metrics[path.name] = path.read_text()
        if "case118_ieee_RMSE.csv" not in metrics:
            raise ValueError("Native evaluation did not produce its RMSE table")
        receipt["runs"][name] = {
            "test_count": len(expected), "test_ids": native["test"],
            "evaluation_seconds": time.monotonic() - start, "metrics_csv": metrics,
        }
        (work / "comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"{name}: {len(expected)} native test cases; metrics saved", flush=True)


if __name__ == "__main__":
    main()

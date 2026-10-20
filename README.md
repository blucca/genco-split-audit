# A small tutorial split can overlap a pretrained model's training set

The GENCO OPF Colab tutorial downloads the first **1,000 IEEE 118 scenarios**,
then loads the paper's pretrained Small checkpoint. Re-splitting that small
slice gives **71 test scenarios: 48 from the checkpoint's original training
split and 23 from its original test split**.

This is a metadata-only reproduction using the original scenario and
load-group IDs. It evaluates the tutorial's sampling contract; model error,
runtime, and the paper's original full-test results have not been remeasured.

## Reproduce

Requires Python and NumPy. No model weights, GPU, Torch, or training are needed.

```sh
python audit.py
# Or use the original split file already on disk:
python audit.py --split-json /path/to/case118_ieee_scenario_splits.json
```

The script downloads one revision-pinned official split JSON (about 2.4 MB),
reproduces the source's seed-0 shuffle and load-group split, and prints every
intersecting ID. A fixed-corpus assertion checks the 71 / 48 / 23 result.

| Tutorial subset | Original train | Original validation | Original test |
|---|---:|---:|---:|
| Train, 837 scenarios | 681 | 110 | 46 |
| Validation, 92 scenarios | 68 | 0 | 24 |
| Test, 71 scenarios | **48** | 0 | **23** |

## What to change

Keep the tutorial's local split for the Tiny model trained on that slice.
For evaluating the pretrained IEEE 118 checkpoint, use the intersection of
the downloaded scenario IDs with **that checkpoint's saved original test
IDs**. There are **93** such scenarios in this slice; the script prints them.
Reuse the checkpoint's saved normalizer. Label that result a small subset of
the official test set. Full reproduction uses the complete saved test split.

The 2000-to-118 transfer experiment has a different pretraining population;
its provenance should be described separately.

## Sources and exact scope

- [Tutorial and helpers at `9d42d823`](https://github.com/gridfm/gridfm-graphkit/tree/9d42d8233268dc92c07f334a643b20b134880f4e/examples/notebooks):
  `prepare_tutorial_opf_raw(n_scenarios=1000)` copies partitions 0–4;
  the Small evaluation uses the same slice through `case118_ieee`.
- [Small config](https://github.com/gridfm/gridfm-graphkit/blob/9d42d8233268dc92c07f334a643b20b134880f4e/examples/config/HGNS_OPF_case118_small.yaml):
  seed 0, 1,000 scenarios, 0.1 validation/test ratios, split by load group.
- [Datamodule](https://github.com/gridfm/gridfm-graphkit/blob/9d42d8233268dc92c07f334a643b20b134880f4e/gridfm_graphkit/datasets/hetero_powergrid_datamodule.py)
  and [group splitter](https://github.com/gridfm/gridfm-graphkit/blob/9d42d8233268dc92c07f334a643b20b134880f4e/gridfm_graphkit/datasets/utils.py).
  `audit.py` transcribes these selection operations into NumPy. It does not
  execute the Torch datamodule. The dataset requires contiguous 0-based
  scenario IDs, so the saved original indices identify the same raw cases here.
- [Official OPF model](https://huggingface.co/gridfm/genco-opf-datakit/tree/2d2401572f6a2d0c1be79841b2def99cf45e1639):
  Small seed 0, run `465594abc8eb4c96b908363afc84a0e1`, the exact run linked by
  the tutorial helper. Its split contains 169,148 / 21,183 / 21,199 scenarios.
- [Original data](https://huggingface.co/datasets/gridfm/opf_small_case118_ieee/tree/5377c37bd0e75980590788ffdcb9100d3bc330ff).
  `scenario-load-map.csv` is the deduplicated `scenario,load_scenario_idx`
  projection of its first five bus partitions. File paths and the extraction
  procedure are in [provenance.json](provenance.json).

Upstream code, checkpoints, and data are by the OpenGridFM/GENCO contributors
and distributed under Apache-2.0. This audit was prepared by blucca's
GPT-6 Astra engineering agent. [Saved output](result.json) includes the full
cross-tabulation and IDs. License: Apache-2.0.

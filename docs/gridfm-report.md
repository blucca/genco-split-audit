# Keep GENCO’s test set attached to its checkpoint

## Finding and contribution

The GENCO OPF tutorial offers a convenient first experiment: download 1,000
IEEE 118 scenarios and evaluate the pretrained Small checkpoint. At the
pinned tutorial revision, its local re-split selects **71 test scenarios,
48 of which belong to that checkpoint’s recorded training partition**.

We built **GENCO Split Audit** to make this relationship visible and repair
the evaluation handoff. It reconstructs the tutorial selection from original
scenario/load-group IDs, intersects the downloaded data with the checkpoint’s
saved split, and runs the native OpenGridFM evaluator on both selections.
The repaired local holdout has **93 scenarios, all from the checkpoint’s
original test partition**.

**[Explore the measured results and every selected ID](https://blucca.github.io/genco-split-audit/)** ·
**[Code and reproduction instructions](https://github.com/blucca/genco-split-audit)**

![The interactive audit shows the tutorial's 71 test scenarios, their original split memberships, and the native CPU evaluation results.](https://blucca.github.io/genco-split-audit/demo.png)

*Figure 1. The tutorial selection contains original training and test cases.
The explorer switches to the 93-case saved holdout and exposes the underlying
membership JSON for independent inspection.*

## Method

We fixed the tutorial, data, and pretrained model revisions, then used the
first five partitions of `gridfm/opf_small_case118_ieee`: 1,000 scenarios
across 45 load groups. `scenario-load-map.csv` contains the original
`scenario,load_scenario_idx` projection from the bus parquet files.

For this corpus, raw scenario IDs are the contiguous dataset indices 0–999;
the upstream dataset processor enforces that correspondence. We replayed
the tutorial’s seed-0 shuffle and load-group split, then compared the selected
IDs with the exact Small checkpoint’s saved scenario partitions:

| Tutorial partition | Original train | Original validation | Original test | Total |
|---|---:|---:|---:|---:|
| Train | 681 | 110 | 46 | 837 |
| Validation | 68 | 0 | 24 | 92 |
| Test | **48** | 0 | **23** | **71** |
| Downloaded slice | **797** | **110** | **93** | **1,000** |

The repair preserves that checkpoint’s membership:

```text
local evaluation IDs = downloaded scenario IDs ∩ checkpoint.saved_test_IDs
```

`compare.py` writes the original local memberships to `train.pt`, `val.pt`,
and `test.pt`, and configures the upstream evaluator through
`split_from_existing_files`. Both evaluations use the same pretrained Small
weights and the checkpoint’s saved normalizer. The workflow also checks the
native evaluator’s actual selected IDs against the intended 71- and 93-case
sets. This last check covers the handoff from the lightweight audit to the
real datamodule.

## Native evaluation results

Both evaluations completed in
[CPU run 39454719421](https://github.com/blucca/genco-split-audit/actions/runs/39454719421),
using Torch 2.11.0+cpu, batch size 8, and two CPU threads. The permanent
[comparison JSON](https://github.com/blucca/genco-split-audit/blob/main/docs/comparison.json)
contains the selected IDs, native metric CSV contents, runtime information,
and provenance.

| Measurement | Tutorial re-split | Saved checkpoint holdout |
|---|---:|---:|
| Evaluated scenarios | 71 | 93 |
| Scenarios in the original training partition | 48 | 0 |
| Generator Pg RMSE (MW) | 12.479683 | 16.109234 |
| Mean objective gap (%) | 1.293915 | 1.267224 |
| Mean active residual (MW) | 0.948622 | 0.971157 |
| Mean reactive residual (MVar) | 0.231132 | 0.235902 |

Generator-power RMSE rises by 3.630 MW, while mean objective gap falls by
0.027 percentage points. These are descriptive measurements on two
different selections, with 23 shared scenarios. The experiment’s scope is
one checkpoint and a 1,000-scenario slice; isolating a memorization effect
requires a separately controlled design. Reproducing the paper’s complete
test evaluation requires its full saved test split.

## How to use the fix

For the **Tiny model trained on this local slice**, retain its own local
training/validation/test split. For the **pretrained IEEE 118 Small
checkpoint**, select its saved test membership and retain its normalizer.
Describe the resulting 93 cases as a local subset of the original holdout.
The tutorial’s **2000-to-118 transfer experiment** has a different
pretraining population and needs its own provenance description.

For a lightweight membership check:

```sh
git clone https://github.com/blucca/genco-split-audit.git
cd genco-split-audit
python -m pip install numpy
python audit.py
```

For the native model comparison, fork the repository and run **Pinned CPU
holdout comparison** from its Actions tab. It downloads the pinned data and
weights, constructs the split tensors, runs both evaluations, and saves the
configurations, selection receipts, logs, and CSVs. The existing results are
also available directly in the repository. The native workflow is optional
for inspecting the ID-level finding.

## Why this helps GENCO users

A checkpoint carries a training history as well as weights. Keeping the
saved split alongside the model lets small, low-resource experiments retain
that history. A fresh local split remains useful for a freshly trained
model; the saved-split intersection supplies the appropriate evaluation
population for this pretrained checkpoint.

The contribution is a reproducible evaluation tool and a concrete tutorial
repair: visible membership, native selected-ID checks, and measured results.
We [reported the pinned tutorial issue upstream](https://github.com/gridfm/gridfm-graphkit/issues/145)
with the source anchors and the proposed selection change. The standalone
workflow is available for researchers to use today.

## Fixed sources and attribution

- **OpenGridFM code:** `gridfm/gridfm-graphkit` at
  [`9d42d8233268dc92c07f334a643b20b134880f4e`](https://github.com/gridfm/gridfm-graphkit/tree/9d42d8233268dc92c07f334a643b20b134880f4e),
  including the OPF tutorial, Small configuration, group splitter, and native evaluator.
- **Data:** `gridfm/opf_small_case118_ieee` at
  [`5377c37bd0e75980590788ffdcb9100d3bc330ff`](https://huggingface.co/datasets/gridfm/opf_small_case118_ieee/tree/5377c37bd0e75980590788ffdcb9100d3bc330ff).
- **Model:** `gridfm/genco-opf-datakit` at
  [`2d2401572f6a2d0c1be79841b2def99cf45e1639`](https://huggingface.co/gridfm/genco-opf-datakit/tree/2d2401572f6a2d0c1be79841b2def99cf45e1639),
  `case118_ieee/small/seed0/best_model_state_dict.pt`,
  run `465594abc8eb4c96b908363afc84a0e1`.
- **Audit evidence:** [source provenance](https://github.com/blucca/genco-split-audit/blob/main/provenance.json),
  [membership and cross-tabulation](https://github.com/blucca/genco-split-audit/blob/main/result.json),
  [native results](https://github.com/blucca/genco-split-audit/blob/main/docs/comparison.json).

The code, checkpoint, and dataset are by the OpenGridFM/GENCO contributors,
under Apache-2.0. The audit, experiments, and this report were produced by
Blucca’s GPT-6 Astra engineering agent. The audit is also Apache-2.0. Its
[earlier DEV build story](https://dev.to/blucca/the-laziest-ai-benchmark-still-needs-a-real-holdout-51i8)
describes the same underlying work; this report presents the scientific
method and results together.

## Entrant and contact

**Blucca** ([blucca](https://huggingface.co/blucca)) — individual entry.

Contact: **belgialucca@gmail.com**.

**Upstream follow-through:** On October 21, the maintainer [requested a patch against `lfe-tutorial`](https://github.com/gridfm/gridfm-graphkit/issues/145#issuecomment-6299447532) using the saved-test intersection described above. The minimal upstream change is being prepared; the standalone audit and native results are already available.

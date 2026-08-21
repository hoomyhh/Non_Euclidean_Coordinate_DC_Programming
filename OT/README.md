# Sparse UOT experiments

The OT notebook contains the optimization methods used by both Monte Carlo
runners:

- `run_sparse_ot_columnwise_mc.py` runs the original synthetic clustered
  instance.
- `run_sparse_ot_cifar10_mc.py` runs the same methods and work accounting on
  frozen CIFAR-10 image features.

## CIFAR-10 protocol

The CIFAR experiment uses source images from the training split and target
images from the test split. Images are represented by L2-normalized penultimate
features from torchvision's ImageNet-pretrained ResNet-18 with the fixed
`IMAGENET1K_V1` weights. The ground cost is squared Euclidean distance between
features, divided by the median cost of the sampled instance by default.

Labels are not used to build the cost or solve the transport problem. They are
used only for final label-transfer diagnostics.

The default optimization parameters and method budgets match the OT notebook,
except that the CIFAR experiment uses `top_k=4`. Source and target masses are

```text
source_mass[i] = num_target / num_source
target_mass[j] = 1
```

so their total mass and the scale of the current regularization parameters are
preserved.

## Preparing features

The first command downloads CIFAR-10 and the fixed ResNet-18 weights if they
are not already cached, extracts all train/test features, and writes one
versioned `.npz` cache:

```bash
python ai-context/code/OT/run_sparse_ot_cifar10_mc.py \
  --download \
  --prepare-features-only
```

The default cache is
`ai-context/code/OT/data/cifar10_resnet18_imagenet1k_v1_features.npz`.
Subsequent experiment runs do not need dataset or model downloads.

## Smoke test

Run every method with two short outer iterations and the notebook correctness
checks:

```bash
python ai-context/code/OT/run_sparse_ot_cifar10_mc.py \
  --num-runs 1 \
  --smoke-test \
  --output-dir ai-context/code/OT/outputs/sparse_ot_cifar10_smoke
```

## Full Monte Carlo run

```bash
python ai-context/code/OT/run_sparse_ot_cifar10_mc.py \
  --num-runs 10 \
  --output-dir ai-context/code/OT/outputs/sparse_ot_cifar10_mc10
```

Use `--resume` to retain completed per-run CSV files after an interrupted run.
The problem seed increases by one per trial and the solver seed by `1009`, as
in the synthetic runner. Sampling is class-stratified by default; use
`--sampling random` for unconstrained empirical batches.

Independent shards can use `--run-start S --num-runs T` to execute global run
indices `S, ..., T-1`. Their per-run files retain the global indices and can be
combined in one output directory before a final `--resume` aggregation.

Useful budget overrides for screening are:

```text
--uniform-sweeps
--gs-sweeps
--full-outer-iterations
--full-inner-iterations
```

## Outputs

The output directory contains:

- `raw_histories_run_*.csv`: objective, feasibility, time, and work histories;
- `final_run_*.csv`: final optimization and label diagnostics by method;
- `instance_run_*.json`: sampled indices, class counts, and cost statistics;
- `monte_carlo_final_summary_by_method.csv`: mean, standard deviation, and SEM;
- `average_*.csv` and `tikz_csv/`: averaged convergence curves;
- `paper_csv/`: files prepared for the manuscript.

The two files

```text
paper_csv/convergence_sparse_ot__objective.csv
paper_csv/convergence_sparse_ot__feasibility.csv
```

have the same columns and method identifiers as the files currently consumed
by `AISTATS2027/Figs/OT.tex`. The existing manuscript data are not overwritten
by default. After inspecting a completed run, they can be published explicitly:

```bash
python ai-context/code/OT/run_sparse_ot_cifar10_mc.py \
  --num-runs 10 \
  --resume \
  --paper-data-dir AISTATS2027/Data
```

This also copies `sparse_ot_cifar10_final_metrics.csv`, which contains the
candidate Appendix-table metrics:

- label-transfer accuracy and balanced accuracy;
- transported same-label mass fraction;
- mean label confidence;
- top-Q mass fraction;
- final objective, infeasibility, work, and optimization time.

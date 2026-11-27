<<<<<<< HEAD
# Sparse unbalanced optimal transport on CIFAR-10

Code for the sparse unbalanced OT experiment of the paper.

- `sparse_uot.py`: the problem, the five methods, the certified inner stopping rule and the work accounting.
- `run_cifar10.py`: features, problem instances, runs and summaries.

## Setup

```bash
pip install -r requirements.txt
```

`torch` and `torchvision` are needed only to compute the image features.

## Reproducing the results

```bash
python run_cifar10.py features --download   # CIFAR-10 + ImageNet ResNet-18 features
python run_cifar10.py run                   # 10 seeds x 5 methods
python run_cifar10.py summarize             # curves and targets table
```

`run` writes one file per seed and method to `results/runs/`. Seeds and methods can be run as separate processes, for example

```bash
python run_cifar10.py run --seeds 3 --methods kl_random kl_gs_breg
```

One (seed, method) takes about 5 to 20 minutes on a single CPU core.
`summarize` writes `results/curves_objective.csv` and `results/curves_infeasibility.csv` (mean and 95% confidence band over seeds) and `results/targets.csv` (matvec passes until the mean objective is within 25, 10 and 5% of the best one at 30000 passes), and prints the share of subproblems that reached the step cap without certification.

## Setting

Each seed defines the sampling of m = 256 training and n = 512 test images (class-stratified) and the column order. The cost is the median-normalized squared distance between L2-normalized ResNet-18 features, a = (n/m)1, b = 1, and
tau_a = tau_b = 100, gamma = 0.5, lambda = 10, Q = 4.

| Method | Update |
|---|---|
| `kl_random` (KL-Random) | one column, random-reshuffled order |
| `kl_gs_breg` (KL-GS(Breg)) | best of 8 columns by the predicted Bregman decrease (gap contribution) |
| `kl_gs_pg` (KL-GS(PG)) | best of 8 columns by the normalized projected gradient |
| `kl_full` (KL-Full) | full plan, entropy steps with L = tau_a + tau_b |
| `e_full` (E-Full) | full plan, projected-gradient steps with a Euclidean Lipschitz bound |

Column steps use the columnwise constant L_j on the trust region |log(z_i/p_ij)| <= 4.
Every subproblem is solved by repeated steps until its error is certified below theta * max(Delta_k, decrease so far), theta = 0.1, or 200 steps are reached.
Work is counted in matvec-pass equivalents (m n entries): 3m per column step, 3mn per full step, m (mn) per certificate evaluation, and 2m per scored Gauss-Southwell candidate.
Budgets: 300 sweeps (n column updates each) for the column methods and 100 outer iterations for the full methods.
=======
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
>>>>>>> eb1f918f0029d61c00177886b747926c2700a8ba

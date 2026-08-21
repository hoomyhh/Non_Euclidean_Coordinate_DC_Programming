# Sparse UOT experiments

This directory contains two sparse unbalanced optimal transport experiments:
a clustered synthetic problem and a CIFAR-10 feature problem.

## Code structure

- `sparse_ot_core.py` contains the problem model, update rules, solvers, and
  MatVec work accounting.
- `experiment_io.py` runs all methods and writes Monte Carlo summaries and
  plotting CSVs.
- `run_synthetic.py` defines and runs the synthetic experiment.
- `run_cifar10.py` prepares frozen ResNet-18 features and runs the CIFAR-10
  experiment. CIFAR labels are used only for final evaluation metrics.

Install the dependencies from the root of the code repository:

```bash
python -m pip install -r OT_new/requirements.txt
```

## Running the experiments

Run the synthetic 30k MatVec experiment:

```bash
python OT_new/run_synthetic.py \
  --num-runs 10 \
  --uniform-sweeps 10000 \
  --gs-sweeps 1579 \
  --full-outer-iterations 100 \
  --full-inner-iterations 100 \
  --record-every-sweeps 1 \
  --output-dir OT_new/outputs/synthetic_mc10_30k
```

Prepare the CIFAR-10 feature cache once:

```bash
python OT_new/run_cifar10.py --download --prepare-features-only
```

Run the CIFAR-10 experiment. Its default method budgets reproduce the 30k
MatVec protocol:

```bash
python OT_new/run_cifar10.py \
  --num-runs 10 \
  --output-dir OT_new/outputs/cifar10_mc10_30k
```

Use `--help` to list data, sampling, method-budget, and output options.

## Results

Each output directory contains:

- `raw_histories_run_*.csv` and `final_run_*.csv` for individual trials;
- `monte_carlo_final_summary_by_method.csv` for final metrics;
- `average_objective_vs_matvec.csv` and
  `average_feasibility_vs_matvec.csv` for convergence plots;
- `tikz_csv/` with per-method plotting data;
- `paper_csv/` with LaTeX-ready objective, feasibility, and final-metric CSVs.

The CIFAR feature cache is stored in `OT_new/data/`. Generated data,
experiment outputs, Python caches, and editor files are excluded from Git.

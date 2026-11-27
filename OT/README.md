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

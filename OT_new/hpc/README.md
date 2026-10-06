# CIFAR-10 sparse UOT on Arrhenius

Each (configuration, seed, method) is an independent single-core run.
`ot_node.slurm` runs all lines of a task file side by side on 72 cores of a
GPU node (our allocation cannot use the cpu partition; the GPU stays idle) and
then aggregates each configuration. Configurations are defined in `configs.sh`.

One-time setup (paths as in `ot_node.slurm`):

```bash
P=/nobackup/proj/disk/ulio_inverse/akarak/ot
mkdir -p $P/data $P/outputs
# from the workstation: copy the feature cache (no torch or CIFAR needed then)
rsync -av OT_new/data/cifar10_resnet18_imagenet1k_v1_features.npz arrhenius:$P/data/
```

Run, from `OT_new/`:

```bash
bash hpc/make_tasks.sh $P/tasks.txt 10 paper_gamma certified
mkdir -p hpc_logs
sbatch hpc/ot_node.slurm
```

Per-run logs go to `$P/outputs/<config>/logs/`, paper CSVs to
`$P/outputs/<config>/paper_csv/`. To re-aggregate by hand:
`python run_cifar10.py --aggregate-only --output-dir $P/outputs/<config>`.

#!/usr/bin/env python3
"""Sparse unbalanced OT between CIFAR-10 images.

    python run_cifar10.py features --download   # ResNet-18 features, once
    python run_cifar10.py run                   # 10 seeds x 5 methods
    python run_cifar10.py summarize             # curves and table

`run` accepts --seeds and --methods, so seeds and methods can be run as
separate processes; `summarize` uses whatever results are present.  Each seed
defines the sampling of images and the column order.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sparse_uot as su

HERE = Path(__file__).resolve().parent

# Settings of the paper.
NUM_SOURCE, NUM_TARGET = 256, 512
PROBLEM = dict(tau_a=100.0, tau_b=100.0, gamma=0.5, lam=10.0, q=4)
SWEEPS = {"kl_random": 300, "kl_gs_breg": 300, "kl_gs_pg": 300, "kl_full": 100, "e_full": 100}
BUDGET = 30000  # matvec-pass equivalents used for the targets table


# ----------------------------------------------------------------- features


def extract_features(cache, cifar_root, download, device):
    """L2-normalized penultimate-layer features of an ImageNet-pretrained
    ResNet-18 for the CIFAR-10 training and test sets."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms(antialias=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    arrays = {}
    for split, train in (("train", True), ("test", False)):
        data = CIFAR10(root=str(cifar_root), train=train, transform=transform, download=download)
        features, labels = [], []
        with torch.inference_mode():
            for images, targets in DataLoader(data, batch_size=128, shuffle=False):
                values = torch.nn.functional.normalize(model(images.to(device)), p=2, dim=1)
                features.append(values.cpu().numpy().astype(np.float32))
                labels.append(targets.numpy().astype(np.int64))
        arrays[f"{split}_features"] = np.concatenate(features)
        arrays[f"{split}_labels"] = np.concatenate(labels)
        print(f"{split}: {arrays[f'{split}_features'].shape}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)


def load_features(cache):
    def normalize(x):
        x = np.asarray(x, dtype=np.float32)
        return np.ascontiguousarray(x / np.linalg.norm(x, axis=1, keepdims=True), dtype=np.float32)

    with np.load(cache, allow_pickle=False) as data:
        return (
            normalize(data["train_features"]),
            np.asarray(data["train_labels"], dtype=np.int64),
            normalize(data["test_features"]),
            np.asarray(data["test_labels"], dtype=np.int64),
        )


# ---------------------------------------------------------------- instances


def stratified_sample(labels, size, rng):
    classes = np.unique(labels)
    counts = np.full(classes.size, size // classes.size, dtype=np.int64)
    counts[rng.permutation(classes.size)[: size % classes.size]] += 1
    chosen = [
        rng.choice(np.flatnonzero(labels == c), int(k), replace=False)
        for c, k in zip(classes, counts)
    ]
    return np.asarray(rng.permutation(np.concatenate(chosen)), dtype=np.int64)


def make_instance(features, seed):
    """m training images (sources) and n test images (targets), class-stratified;
    cost = median-normalized squared distance between their features."""
    train_x, train_y, test_x, test_y = features
    rng = np.random.default_rng(seed)
    source = stratified_sample(train_y, NUM_SOURCE, rng)
    target = stratified_sample(test_y, NUM_TARGET, rng)
    cost = np.maximum(2.0 - 2.0 * (train_x[source] @ test_x[target].T), 0.0).astype(np.float64)
    cost /= float(np.median(cost))
    problem = su.Problem(
        cost=np.ascontiguousarray(cost),
        a=np.full(NUM_SOURCE, NUM_TARGET / NUM_SOURCE, dtype=np.float64),
        b=np.ones(NUM_TARGET, dtype=np.float64),
        **PROBLEM,
    )
    return problem, source, target


# ---------------------------------------------------------------------- run


def run(args):
    features = load_features(args.features)
    runs = args.out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        problem, source, target = make_instance(features, seed)
        (runs / f"instance_{seed:02d}.json").write_text(
            json.dumps({"train_indices": source.tolist(), "test_indices": target.tolist()})
        )
        for method in args.methods:
            settings = su.Settings(sweeps=SWEEPS[method], order_seed=1009 * seed)
            started = time.perf_counter()
            _, history = su.solve(problem, settings, method)
            history.insert(0, "seed", seed)
            history.to_csv(runs / f"seed_{seed:02d}_{method}.csv", index=False)
            last = history.iloc[-1]
            print(
                f"seed {seed:2d} {su.LABELS[method]:12s} objective {last.objective:9.3f} "
                f"infeasibility {last.infeasibility:.2e} matvec {last.matvec:8.0f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )


# ---------------------------------------------------------------- summarize


def mean_curve(histories, column, grid_size=401):
    """Mean and 95% confidence band over seeds on a common log grid."""
    curves = []
    for _, h in histories.groupby("seed"):
        h = h.sort_values("matvec").groupby("matvec", as_index=False).last()
        curves.append((h["matvec"].to_numpy(float), h[column].to_numpy(float)))
    start = max(float(x[x > 0.0].min()) for x, _ in curves)
    end = min(float(x.max()) for x, _ in curves)
    grid = np.geomspace(start, end, grid_size)
    values = np.vstack([np.interp(grid, x, y) for x, y in curves])
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    sem = std / math.sqrt(len(curves))
    return pd.DataFrame(
        {"matvec": grid, column: mean, f"{column}_lo": mean - 1.96 * sem, f"{column}_hi": mean + 1.96 * sem}
    )


def summarize(args):
    files = sorted((args.out / "runs").glob("seed_*.csv"))
    if not files:
        raise SystemExit(f"No results in {args.out / 'runs'}.")
    histories = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    curves = {}
    for column in ("objective", "infeasibility"):
        frames = []
        for method in su.METHODS:
            h = histories[histories["method"] == su.LABELS[method]]
            if h["seed"].nunique() > 0:
                frame = mean_curve(h, column)
                frame.insert(0, "method", su.LABELS[method])
                frames.append(frame)
        curves[column] = pd.concat(frames, ignore_index=True)
        curves[column].to_csv(args.out / f"curves_{column}.csv", index=False)

    # Work until the mean objective is within 25, 10, 5% of the best mean
    # objective at the budget.
    objective = curves["objective"]
    final = {
        method: float(np.interp(args.budget, c["matvec"], c["objective"]))
        for method, c in objective.groupby("method")
    }
    best = min(final.values())
    rows = []
    for method, c in objective.groupby("method", sort=False):
        c = c[c["matvec"] <= args.budget]
        row = {"method": method}
        for margin in (25, 10, 5):
            reached = c[c["objective"] <= (1 + margin / 100) * best]
            row[f"within_{margin}%"] = round(float(reached["matvec"].iloc[0]), -2) if len(reached) else None
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "targets.csv", index=False)
    print(f"best mean objective at {args.budget} matvec: {best:.1f}")
    print(table.to_string(index=False))

    certification = histories.groupby(["method", "seed"]).last().groupby("method")[
        ["subproblems", "uncertified", "bregman_steps"]
    ].sum()
    certification["uncertified_%"] = 100 * certification["uncertified"] / certification["subproblems"]
    certification["steps_per_subproblem"] = certification["bregman_steps"] / certification["subproblems"]
    print(certification[["uncertified_%", "steps_per_subproblem"]].round(1).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("features", "run", "summarize"))
    parser.add_argument("--features", type=Path, default=HERE / "data" / "cifar10_resnet18_features.npz")
    parser.add_argument("--cifar-root", type=Path, default=HERE / "data" / "cifar10")
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 and the ResNet-18 weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--methods", nargs="+", choices=su.METHODS, default=list(su.METHODS))
    parser.add_argument("--budget", type=float, default=BUDGET)
    args = parser.parse_args()

    if args.command == "features":
        extract_features(args.features, args.cifar_root, args.download, args.device)
    elif args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()

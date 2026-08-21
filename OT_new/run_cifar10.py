#!/usr/bin/env python3
"""Monte Carlo sparse-UOT benchmark on frozen CIFAR-10 image features.

The problem model and optimization methods are imported from sparse_ot_core.
This file owns only CIFAR-10 feature preparation, instance construction,
label-transfer metrics, experiment configuration, and output publication.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

import experiment_io as io
import sparse_ot_core as core

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "data" / "cifar10"
DEFAULT_FEATURE_CACHE = (
    SCRIPT_DIR / "data" / "cifar10_resnet18_imagenet1k_v1_features.npz"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cifar10_mc10_30k"

FEATURE_CACHE_VERSION = 1
FEATURE_EXTRACTOR = "torchvision.resnet18"
FEATURE_WEIGHTS = "IMAGENET1K_V1"
CIFAR10_NUM_CLASSES = 10

EXTRA_FINAL_METRICS = (
    "label_transfer_accuracy",
    "label_transfer_balanced_accuracy",
    "transported_label_mass_fraction",
    "mean_label_confidence",
    "top1_source_label_accuracy",
    "topq_mass_fraction",
    "transported_mass_total",
)

PAPER_METRICS = (
    "objective",
    "marginal_relative_l1_residual",
    "label_transfer_accuracy",
    "label_transfer_balanced_accuracy",
    "transported_label_mass_fraction",
    "mean_label_confidence",
    "topq_mass_fraction",
    "matvec_pass_equivalent",
    "optimization_time_seconds",
)


@dataclass(frozen=True)
class CifarFeatureData:
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class CifarProblemInstance:
    problem: core.SparseOTProblem
    source_indices: np.ndarray
    target_indices: np.ndarray
    source_labels: np.ndarray
    target_labels: np.ndarray
    cost_scale: float


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Feature cache contains a zero feature vector.")
    return np.ascontiguousarray(features / norms, dtype=np.float32)


def load_feature_cache(path: Path) -> CifarFeatureData:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as cached:
        required = {
            "train_features",
            "train_labels",
            "test_features",
            "test_labels",
            "metadata_json",
        }
        missing = required.difference(cached.files)
        if missing:
            raise ValueError(f"Feature cache is missing keys: {sorted(missing)}")
        train_features = _normalize_rows(cached["train_features"])
        test_features = _normalize_rows(cached["test_features"])
        train_labels = np.asarray(cached["train_labels"], dtype=np.int64)
        test_labels = np.asarray(cached["test_labels"], dtype=np.int64)
        metadata = json.loads(str(cached["metadata_json"].item()))

    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("Cached features must be two-dimensional arrays.")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Train and test feature dimensions do not match.")
    if train_features.shape[0] != train_labels.size:
        raise ValueError("Train feature and label counts do not match.")
    if test_features.shape[0] != test_labels.size:
        raise ValueError("Test feature and label counts do not match.")
    if int(metadata.get("cache_version", -1)) != FEATURE_CACHE_VERSION:
        raise ValueError(
            f"Unsupported feature cache version: {metadata.get('cache_version')}"
        )
    return CifarFeatureData(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        metadata=metadata,
    )


def resolve_torch_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resnet_checkpoint_path(weights, torch_module) -> Path:
    filename = Path(urlparse(weights.url).path).name
    return Path(torch_module.hub.get_dir()) / "checkpoints" / filename


def prepare_feature_cache(
    cache_path: Path,
    cifar_root: Path,
    allow_downloads: bool,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> CifarFeatureData:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    checkpoint_path = _resnet_checkpoint_path(weights, torch)
    if not allow_downloads and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"ResNet-18 weights are not cached at {checkpoint_path}. "
            "Re-run with --download to allow torchvision to fetch them."
        )

    transform = weights.transforms(antialias=True)
    train_data = CIFAR10(
        root=str(cifar_root), train=True, transform=transform, download=allow_downloads
    )
    test_data = CIFAR10(
        root=str(cifar_root), train=False, transform=transform, download=allow_downloads
    )
    device = resolve_torch_device(device_name)
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)

    def extract(dataset, split_name: str) -> tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        feature_batches = []
        label_batches = []
        started = time.perf_counter()
        with torch.inference_mode():
            for batch_index, (images, labels) in enumerate(loader):
                values = model(images.to(device, non_blocking=device.type == "cuda"))
                values = torch.nn.functional.normalize(values, p=2, dim=1)
                feature_batches.append(
                    values.cpu().numpy().astype(np.float32, copy=False)
                )
                label_batches.append(labels.numpy().astype(np.int64, copy=False))
                if (batch_index + 1) % 50 == 0:
                    seen = min((batch_index + 1) * batch_size, len(dataset))
                    print(
                        f"[{split_name}] extracted {seen}/{len(dataset)} features",
                        flush=True,
                    )
        print(
            f"[{split_name}] feature extraction took "
            f"{(time.perf_counter() - started) / 60.0:.2f} min",
            flush=True,
        )
        return (
            np.ascontiguousarray(np.concatenate(feature_batches), dtype=np.float32),
            np.ascontiguousarray(np.concatenate(label_batches), dtype=np.int64),
        )

    train_features, train_labels = extract(train_data, "train")
    test_features, test_labels = extract(test_data, "test")
    metadata = {
        "cache_version": FEATURE_CACHE_VERSION,
        "dataset": "CIFAR-10",
        "feature_extractor": FEATURE_EXTRACTOR,
        "feature_weights": FEATURE_WEIGHTS,
        "feature_dimension": int(train_features.shape[1]),
        "features_l2_normalized": True,
        "torch_version": torch.__version__,
        "torchvision_weights_url": weights.url,
        "preprocessing": repr(transform),
        "train_size": int(train_features.shape[0]),
        "test_size": int(test_features.shape[0]),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(cache_path.name + ".tmp")
    with temporary_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary_path.replace(cache_path)
    print(f"Saved frozen feature cache to {cache_path}", flush=True)
    return load_feature_cache(cache_path)


def ensure_feature_cache(args: argparse.Namespace) -> CifarFeatureData:
    if args.feature_cache.exists():
        return load_feature_cache(args.feature_cache)
    return prepare_feature_cache(
        cache_path=args.feature_cache,
        cifar_root=args.cifar_root,
        allow_downloads=args.download,
        device_name=args.device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )


def sample_indices(
    labels: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    sampling: str,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if not 1 <= sample_size <= labels.size:
        raise ValueError(
            f"sample_size must be between 1 and {labels.size}; got {sample_size}."
        )
    if sampling == "random":
        return np.asarray(
            rng.choice(labels.size, sample_size, replace=False), dtype=np.int64
        )
    if sampling != "stratified":
        raise ValueError(f"Unknown sampling strategy: {sampling}")

    classes = np.unique(labels)
    counts = np.full(classes.size, sample_size // classes.size, dtype=np.int64)
    remainder_order = rng.permutation(classes.size)
    counts[remainder_order[: sample_size % classes.size]] += 1
    selected = []
    for class_value, count in zip(classes, counts):
        candidates = np.flatnonzero(labels == class_value)
        if count > candidates.size:
            raise ValueError(
                f"Class {class_value} has {candidates.size} examples, fewer than {count}."
            )
        selected.append(rng.choice(candidates, int(count), replace=False))
    return np.asarray(rng.permutation(np.concatenate(selected)), dtype=np.int64)


def build_cifar_problem(
    config: dict,
    feature_data: CifarFeatureData,
    sampling: str,
    cost_normalization: str,
) -> CifarProblemInstance:
    rng = np.random.default_rng(int(config["problem_seed"]))
    source_indices = sample_indices(
        feature_data.train_labels, int(config["num_source"]), rng, sampling
    )
    target_indices = sample_indices(
        feature_data.test_labels, int(config["num_target"]), rng, sampling
    )
    source_features = feature_data.train_features[source_indices]
    target_features = feature_data.test_features[target_indices]
    similarities = source_features @ target_features.T
    cost = np.maximum(2.0 - 2.0 * similarities, 0.0).astype(np.float64)

    if cost_normalization == "median":
        cost_scale = float(np.median(cost))
        if not np.isfinite(cost_scale) or cost_scale <= 0.0:
            raise ValueError(f"Invalid median cost scale: {cost_scale}")
        cost /= cost_scale
    elif cost_normalization == "none":
        cost_scale = 1.0
    else:
        raise ValueError(f"Unknown cost normalization: {cost_normalization}")

    num_source = int(config["num_source"])
    num_target = int(config["num_target"])
    target_mass = np.ones(num_target, dtype=np.float64)
    source_mass = np.full(num_source, num_target / num_source, dtype=np.float64)
    problem = core.SparseOTProblem(
        cost=np.ascontiguousarray(cost),
        source_mass=source_mass,
        target_mass=target_mass,
        source_kl_weight=float(config["source_kl_weight"]),
        target_kl_weight=float(config["target_kl_weight"]),
        quadratic_weight=float(config["quadratic_weight"]),
        sparsity_weight=float(config["sparsity_weight"]),
        top_k=int(config["top_k"]),
    )
    return CifarProblemInstance(
        problem=problem,
        source_indices=source_indices,
        target_indices=target_indices,
        source_labels=feature_data.train_labels[source_indices],
        target_labels=feature_data.test_labels[target_indices],
        cost_scale=cost_scale,
    )


def transport_label_metrics(
    plan: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    plan = np.asarray(plan, dtype=np.float64)
    source_labels = np.asarray(source_labels, dtype=np.int64)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    if plan.shape != (source_labels.size, target_labels.size):
        raise ValueError("Plan dimensions do not match the source and target labels.")

    class_scores = np.zeros((CIFAR10_NUM_CLASSES, plan.shape[1]), dtype=np.float64)
    for class_value in range(CIFAR10_NUM_CLASSES):
        class_scores[class_value] = plan[source_labels == class_value].sum(axis=0)
    predictions = np.argmax(class_scores, axis=0)
    target_column_mass = np.maximum(plan.sum(axis=0), np.finfo(float).tiny)
    confidence = class_scores.max(axis=0) / target_column_mass

    per_class_accuracy = []
    for class_value in np.unique(target_labels):
        mask = target_labels == class_value
        per_class_accuracy.append(
            float(np.mean(predictions[mask] == target_labels[mask]))
        )

    matching_mask = source_labels[:, None] == target_labels[None, :]
    total_mass = float(plan.sum())
    matching_mass = float(plan[matching_mask].sum())
    top1_predictions = source_labels[np.argmax(plan, axis=0)]
    q = min(max(int(top_k), 1), plan.shape[0])
    topq_values = np.partition(plan, plan.shape[0] - q, axis=0)[-q:, :]
    return {
        "label_transfer_accuracy": float(np.mean(predictions == target_labels)),
        "label_transfer_balanced_accuracy": float(np.mean(per_class_accuracy)),
        "transported_label_mass_fraction": matching_mass
        / max(total_mass, np.finfo(float).tiny),
        "mean_label_confidence": float(np.mean(confidence)),
        "top1_source_label_accuracy": float(np.mean(top1_predictions == target_labels)),
        "topq_mass_fraction": float(
            topq_values.sum() / max(total_mass, np.finfo(float).tiny)
        ),
        "transported_mass_total": total_mass,
    }


def configure_experiment(args: argparse.Namespace, feature_dimension: int) -> dict:
    config = copy.deepcopy(core.CONFIG)
    config.update(
        {
            "num_source": int(args.num_source),
            "num_target": int(args.num_target),
            "dimension": int(feature_dimension),
            "top_k": int(args.top_k),
            "problem_seed": int(args.problem_seed),
            "solver_seed": int(args.solver_seed),
            "candidate_batch_size": int(args.candidate_batch_size),
            "record_every_sweeps": int(args.record_every_sweeps),
        }
    )
    optional_scalars = {
        "source_kl_weight": args.source_kl_weight,
        "target_kl_weight": args.target_kl_weight,
        "quadratic_weight": args.quadratic_weight,
        "sparsity_weight": args.sparsity_weight,
        "block_log_radius": args.block_log_radius,
    }
    for key, value in optional_scalars.items():
        if value is not None:
            config[key] = float(value)

    outer = copy.deepcopy(config["outer_iterations_by_method"])
    if args.uniform_sweeps is not None:
        outer["uniform"] = int(args.uniform_sweeps)
    if args.gs_sweeps is not None:
        outer["lipschitz"] = int(args.gs_sweeps)
        outer["bregman_gap"] = int(args.gs_sweeps)
    if args.full_outer_iterations is not None:
        outer["full_entropy"] = int(args.full_outer_iterations)
        outer["full_euclidean"] = int(args.full_outer_iterations)
    config["outer_iterations_by_method"] = outer

    inner = copy.deepcopy(config["max_inner_iterations_by_method"])
    if args.full_inner_iterations is not None:
        inner["full_entropy"] = int(args.full_inner_iterations)
        inner["full_euclidean"] = int(args.full_inner_iterations)
    config["max_inner_iterations_by_method"] = inner

    return config


def instance_metadata(instance: CifarProblemInstance, config: dict) -> dict:
    problem = instance.problem
    return {
        "problem_seed": int(config["problem_seed"]),
        "source_indices": instance.source_indices.tolist(),
        "target_indices": instance.target_indices.tolist(),
        "source_class_counts": np.bincount(
            instance.source_labels, minlength=CIFAR10_NUM_CLASSES
        ).tolist(),
        "target_class_counts": np.bincount(
            instance.target_labels, minlength=CIFAR10_NUM_CLASSES
        ).tolist(),
        "cost_scale": float(instance.cost_scale),
        "normalized_cost_min": float(problem.cost.min()),
        "normalized_cost_median": float(np.median(problem.cost)),
        "normalized_cost_max": float(problem.cost.max()),
        "source_mass_total": float(problem.source_mass.sum()),
        "target_mass_total": float(problem.target_mass.sum()),
    }


def run_one(
    run_index: int,
    base_config: dict,
    feature_data: CifarFeatureData,
    args: argparse.Namespace,
):
    config = io.make_run_config(base_config, run_index)
    instance = build_cifar_problem(
        config,
        feature_data,
        sampling=args.sampling,
        cost_normalization=args.cost_normalization,
    )

    def metric_function(plan: np.ndarray) -> dict[str, float]:
        return transport_label_metrics(
            plan,
            instance.source_labels,
            instance.target_labels,
            instance.problem.top_k,
        )

    raw, final = io.run_methods(
        instance.problem,
        config,
        run_index=run_index,
        final_metrics=metric_function,
    )
    return raw, final, instance_metadata(instance, config)


def write_paper_outputs(output_dir: Path, aggregated: io.AggregatedOutputs) -> Path:
    paper_dir = output_dir / "paper_csv"
    paper_dir.mkdir(parents=True, exist_ok=True)
    io.paper_curve(
        aggregated.objective_vs_matvec,
        "matvec_pass_equivalent",
        "objective",
        "objective",
    ).to_csv(paper_dir / "convergence_sparse_ot_cifar10__objective.csv", index=False)
    io.paper_curve(
        aggregated.feasibility_vs_matvec,
        "matvec_pass_equivalent",
        "marginal_relative_l1_residual",
        "feasibility",
    ).to_csv(paper_dir / "convergence_sparse_ot_cifar10__feasibility.csv", index=False)
    io.paper_curve(
        aggregated.objective_vs_time,
        "optimization_time_seconds",
        "objective",
        "objective",
    ).to_csv(
        paper_dir / "convergence_sparse_ot_cifar10__objective_vs_time.csv",
        index=False,
    )

    columns = ["method_key", "method", "num_runs"]
    for metric in PAPER_METRICS:
        columns.extend([f"{metric}_mean", f"{metric}_sem"])
    aggregated.final_summary[columns].to_csv(
        paper_dir / "sparse_ot_cifar10_final_metrics.csv", index=False
    )
    return paper_dir


def aggregate_and_publish(args: argparse.Namespace):
    metrics = (*io.BASE_METRICS, *EXTRA_FINAL_METRICS)
    aggregated = io.aggregate_outputs(
        args.output_dir,
        metric_columns=metrics,
        grid_size=args.grid_size,
        include_zero=args.include_zero_grid,
    )
    if aggregated is None:
        return None
    paper_dir = write_paper_outputs(args.output_dir, aggregated)
    if args.paper_data_dir is not None:
        io.publish_paper_outputs(paper_dir, args.paper_data_dir)
    return aggregated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--cifar-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-data-dir", type=Path)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--run-start", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=401)
    parser.add_argument("--num-source", type=int, default=256)
    parser.add_argument("--num-target", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--problem-seed", type=int, default=0)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument(
        "--sampling", choices=("stratified", "random"), default="stratified"
    )
    parser.add_argument(
        "--cost-normalization", choices=("median", "none"), default="median"
    )
    parser.add_argument("--source-kl-weight", type=float)
    parser.add_argument("--target-kl-weight", type=float)
    parser.add_argument("--quadratic-weight", type=float)
    parser.add_argument("--sparsity-weight", type=float)
    parser.add_argument("--block-log-radius", type=float)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--record-every-sweeps", type=int, default=1)
    parser.add_argument("--uniform-sweeps", type=int, default=10000)
    parser.add_argument("--gs-sweeps", type=int, default=1579)
    parser.add_argument("--full-outer-iterations", type=int, default=100)
    parser.add_argument("--full-inner-iterations", type=int, default=100)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-features-only", action="store_true")
    parser.add_argument("--include-zero-grid", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "num_runs",
        "grid_size",
        "num_source",
        "num_target",
        "top_k",
        "candidate_batch_size",
        "record_every_sweeps",
        "feature_batch_size",
    )
    for field in positive_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")
    if args.top_k > args.num_source:
        raise ValueError("--top-k cannot exceed --num-source.")
    if args.candidate_batch_size > args.num_target:
        raise ValueError("--candidate-batch-size cannot exceed --num-target.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative.")
    if not 0 <= args.run_start <= args.num_runs:
        raise ValueError("--run-start must be between zero and --num-runs.")


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(SCRIPT_DIR.parent.resolve()))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    feature_data = ensure_feature_cache(args)
    print(
        f"Loaded CIFAR-10 features: train={feature_data.train_features.shape}, "
        f"test={feature_data.test_features.shape}",
        flush=True,
    )
    if args.prepare_features_only:
        return 0

    config = configure_experiment(args, feature_data.train_features.shape[1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    io.write_metadata(
        args.output_dir / "monte_carlo_metadata.json",
        {
            "experiment": "CIFAR-10 train-to-test sparse UOT on frozen features",
            "core_module": "sparse_ot_core.py",
            "feature_cache": portable_path(args.feature_cache),
            "feature_cache_metadata": feature_data.metadata,
            "num_runs_requested": args.num_runs,
            "run_start": args.run_start,
            "sampling": args.sampling,
            "cost": (
                "median-normalized squared Euclidean distance between "
                "L2-normalized features"
                if args.cost_normalization == "median"
                else "squared Euclidean distance between L2-normalized features"
            ),
            "labels_used_in_optimization": False,
            "base_config": config,
        },
    )

    core.make_method_budget_table(
        core.resolve_method_integer_budgets(
            config.get("outer_iterations_by_method"),
            default=config["num_sweeps"],
            name="outer_iterations_by_method",
        ),
        core.resolve_method_integer_budgets(
            config.get("max_inner_iterations_by_method"),
            default=config["max_inner_iterations"],
            name="max_inner_iterations_by_method",
        ),
        int(config["num_target"]),
    ).to_csv(args.output_dir / "method_budget_table.csv", index=False)

    started = time.perf_counter()
    for run_index in range(args.run_start, args.num_runs):
        raw_path = args.output_dir / f"raw_histories_run_{run_index:03d}.csv"
        final_path = args.output_dir / f"final_run_{run_index:03d}.csv"
        instance_path = args.output_dir / f"instance_run_{run_index:03d}.json"
        print(f"[run {run_index:02d}] starting", flush=True)
        raw, final, run_metadata = run_one(run_index, config, feature_data, args)
        raw.to_csv(raw_path, index=False)
        final.to_csv(final_path, index=False)
        io.write_metadata(instance_path, run_metadata)
        aggregate_and_publish(args)
        print(
            f"[run {run_index:02d}] saved. Total elapsed "
            f"{(time.perf_counter() - started) / 60.0:.2f} min.",
            flush=True,
        )

    aggregate_and_publish(args)
    print(f"Done. CSV outputs are in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

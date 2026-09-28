#!/usr/bin/env python3
"""Matched-accuracy cost comparison across T_in conditions on the synthetic
sparse-UOT instance.

Given raw_histories_run_000.csv from two run_synthetic.py output directories
(one per T_in condition, same problem/solver seed so they're comparable),
reports, for each method, the matvec-pass-equivalent cost needed to reach
within X% of the best objective achieved by any method/condition -- the same
cost-to-target-accuracy logic the paper's own convergence plots use (see
F^\\star in the PSF/OT experiment appendices), rather than cost at a fixed,
unmatched sweep count.

Usage:
    python3 analyze_matched_accuracy.py <tin1_dir> <tin50_dir> [--tol 20,10,5,1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COORD_METHODS = ["uniform", "lipschitz", "bregman_gap"]
FULL_METHODS = ["full_entropy", "full_euclidean"]


def load_matched(tin1_dir: Path, tin_other_dir: Path, tin_other_label: str) -> pd.DataFrame:
    df1 = pd.read_csv(tin1_dir / "raw_histories_run_000.csv")
    df1["condition"] = "T_in=1"
    df_other = pd.read_csv(tin_other_dir / "raw_histories_run_000.csv")
    df_other["condition"] = tin_other_label

    df1_coord = df1[df1.method_key.isin(COORD_METHODS)]
    df_other_coord = df_other[df_other.method_key.isin(COORD_METHODS)]
    # Full-dimensional methods are unaffected by --coord-inner-iterations;
    # keep one copy only (from the T_in=1 run) to avoid double-counting.
    df_full = df1[df1.method_key.isin(FULL_METHODS)].copy()
    df_full["condition"] = "n/a (full-dim, unaffected by T_in)"

    combined = pd.concat([df1_coord, df_other_coord, df_full], ignore_index=True)
    combined["label"] = combined["method"] + " [" + combined["condition"] + "]"
    return combined


def report(combined: pd.DataFrame, tolerances_pct: list[float]) -> None:
    obj_star = combined["objective"].min()
    print(f"Shared target objective (best across all methods/conditions): {obj_star:.6f}\n")

    print("=== Final objective + total matvec cost per method/condition ===")
    finals = (
        combined.sort_values("matvec_pass_equivalent")
        .groupby("label")
        .tail(1)[["label", "objective", "matvec_pass_equivalent", "sweep"]]
        .sort_values("objective")
    )
    print(finals.to_string(index=False))
    print()

    print("=== Matvec-pass-equivalent cost to reach within X% of the shared target ===")
    for tol in tolerances_pct:
        threshold = obj_star * (1.0 + tol / 100.0)
        rows = []
        for label, g in combined.groupby("label"):
            g = g.sort_values("matvec_pass_equivalent")
            hit = g[g["objective"] <= threshold]
            cost = hit["matvec_pass_equivalent"].iloc[0] if len(hit) else float("nan")
            rows.append({"label": label, "matvec_to_reach": cost, "reached": len(hit) > 0})
        sub = pd.DataFrame(rows).sort_values("matvec_to_reach")
        print(f"--- within {tol}% of best ({threshold:.2f}) ---")
        print(sub[["label", "matvec_to_reach", "reached"]].to_string(index=False))
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tin1_dir", type=Path)
    parser.add_argument("tin_other_dir", type=Path)
    parser.add_argument(
        "--other-label",
        default="T_in=50",
        help="Label for the second condition (default: T_in=50).",
    )
    parser.add_argument(
        "--tol",
        default="20,10,5,1",
        help="Comma-separated relative-gap tolerances in percent (default: 20,10,5,1).",
    )
    args = parser.parse_args()
    tolerances = [float(t) for t in args.tol.split(",")]
    combined = load_matched(args.tin1_dir, args.tin_other_dir, args.other_label)
    report(combined, tolerances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

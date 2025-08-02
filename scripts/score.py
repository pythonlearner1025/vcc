#!/usr/bin/env python3
"""
compute_vcc_score.py

Parse a Virtual-Cell-Challenge summary-stats CSV and compute:
  * Differential-Expression Score (DES)
  * Perturbation-Discrimination Score (PDS; L1 version)
  * Mean-Absolute-Error (MAE)
  * Overall leaderboard score

Usage
-----
$ python compute_vcc_score.py results.csv \
        --des-baseline 0.12 \
        --pds-baseline 0.32 \
        --mae-baseline 51.7
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, help="summary-stats CSV file", default="cell-eval-outdir/agg_results.csv")
    p.add_argument("--des-baseline", type=float, default=0.106,
                   help="DESbaseline from the raw-score table") 
    p.add_argument("--pds-baseline", type=float, default=0.516,
                   help="PDSbaseline (L1) from the raw-score table")
    p.add_argument("--mae-baseline", type=float, default=51.7,
                   help="MAEbaseline from the raw-score table")
    return p.parse_args()

def scaled_fraction(metric: float, baseline: float) -> float:
    """
    Scale DES or PDS to [0,1] range.
    If the prediction is worse than baseline, clip at 0.
    """
    if metric <= baseline:
        return 0.0
    return (metric - baseline) / (1.0 - baseline)

def scaled_mae(metric: float, baseline: float) -> float:
    """
    Scale MAE (lower-is-better).
    If the prediction is worse (higher) than baseline, clip at 0.
    """
    if metric >= baseline:
        return 0.0
    return (baseline - metric) / baseline

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load the CSV and grab the `mean` row
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        sys.exit(f"Error reading {args.csv}: {e}")

    # Make sure the first column is the index
    if df.columns[0] != "statistic":
        sys.exit("CSV must have a 'statistic' column as its first column.")

    df.set_index("statistic", inplace=True)

    if "mean" not in df.index:
        sys.exit("CSV does not contain a 'mean' row to read metric values.")

    mean_row = df.loc["mean"]

    # ------------------------------------------------------------------
    # 2. Pull out the three core metrics
    # ------------------------------------------------------------------
    try:
        des  = float(mean_row["de_sig_genes_recall"])
        pds  = float(mean_row["discrimination_score_l1"])
        mae  = float(mean_row["mae"])
    except KeyError as e:
        sys.exit(f"Missing expected column: {e}")

    # ------------------------------------------------------------------
    # 3. Scale vs. baseline
    #    DESscaled = (DESpred − DESbase) / (1 − DESbase)  (clipped ≥0)
    #    PDSscaled = (PDSpred − PDSbase) / (1 − PDSbase)  (clipped ≥0)
    #    MAEscaled = (MAEbase − MAEpred) /  MAEbase       (clipped ≥0)
    # ------------------------------------------------------------------
    des_sc  = scaled_fraction(des, args.des_baseline)
    pds_sc  = scaled_fraction(pds, args.pds_baseline)
    mae_sc  = scaled_mae(mae, args.mae_baseline)

    # ------------------------------------------------------------------
    # 4. Overall leaderboard score
    #    S = (DESscaled + PDSscaled + MAEscaled) / 3
    # ------------------------------------------------------------------
    overall = (des_sc + pds_sc + mae_sc) / 3.0

    # ------------------------------------------------------------------
    # 5. Dump results
    # ------------------------------------------------------------------
    print("=== Virtual-Cell-Challenge evaluation ===")
    print(f"DES  (raw)  : {des:.6f}")
    print(f"PDS  (raw)  : {pds:.6f}")
    print(f"MAE  (raw)  : {mae:.6f}")
    print("-" * 40)
    print(f"DES  (scaled): {des_sc:.6f}")
    print(f"PDS  (scaled): {pds_sc:.6f}")
    print(f"MAE  (scaled): {mae_sc:.6f}")
    print("-" * 40)
    print(f"OVERALL SCORE: {overall:.6f}")

if __name__ == "__main__":
    main()

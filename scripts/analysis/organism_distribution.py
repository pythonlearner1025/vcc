#!/usr/bin/env python
"""
organism_distribution.py
------------------------

A small standalone utility that replicates the key part of the tutorial notebook:
1. Counts how many *samples* (SRX accessions) exist per organism.
2. Sums the total number of *cells* (`obs_count`) per organism.

It prints the two tables to stdout.

Requirements (same as the notebook):
    pip install pandas pyarrow gcsfs

Usage
-----
python organism_distribution.py \
    --gcs_base_path gs://arc-scbasecount/2025-02-25/ \
    --feature_type GeneFull_Ex50pAS

Both arguments default to the values above, so you can typically just run:
    python organism_distribution.py
"""
from __future__ import annotations

import argparse
import os
from typing import List

import gcsfs  # type: ignore
import pandas as pd
import pyarrow.dataset as ds

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def get_file_table(gcs_base_path: str, target: str) -> pd.DataFrame:
    """Recursively list files called *target* underneath *gcs_base_path*.

    Returns a DataFrame with two columns:
    1. organism  – scientific name encoded by the directory name
    2. file_path – full GCS URI to the file
    """
    fs = gcsfs.GCSFileSystem()

    # E.g. pattern -> "gs://arc-scbasecount/2025-02-25/metadata/GeneFull_Ex50pAS/**"
    pattern = os.path.join(gcs_base_path.rstrip("/"), "**")
    files: List[str] = fs.glob(pattern)

    # Keep only the files whose basename matches *target*
    files = [f for f in files if os.path.basename(f) == target]

    table_rows = []
    for f in files:
        # Path structure: .../<feature_type>/<organism>/<target>
        organism = f.rstrip("/").split("/")[-2]
        table_rows.append((organism, f))

    return pd.DataFrame(table_rows, columns=["organism", "file_path"])


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def main(gcs_base_path: str, feature_type: str) -> None:  # pragma: no cover
    fs = gcsfs.GCSFileSystem()

    # Path that contains per-sample Parquet files
    gcs_path = f"{gcs_base_path.rstrip('/')}/metadata/{feature_type}"

    # ------------------------------------------------------------------
    # 1. Locate all per-sample metadata Parquet files
    # ------------------------------------------------------------------
    sample_pq_files = get_file_table(gcs_path, target="sample_metadata.parquet")

    if sample_pq_files.empty:
        raise SystemExit(
            "No sample_metadata.parquet files were found – check the base path and feature type."
        )

    # ------------------------------------------------------------------
    # 2. Load them into a single DataFrame
    # ------------------------------------------------------------------
    dfs: List[pd.DataFrame] = []
    for _, row in sample_pq_files.iterrows():
        ds_handle = ds.dataset(row.file_path, filesystem=fs, format="parquet")
        dfs.append(ds_handle.to_table().to_pandas())

    sample_metadata = pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # 3. Compute desired statistics
    # ------------------------------------------------------------------
    sample_counts = (
        sample_metadata["organism"].value_counts().sort_values(ascending=False)
    )

    cell_counts = (
        sample_metadata.groupby("organism")["obs_count"].sum().sort_values(ascending=False)
    )

    # ------------------------------------------------------------------
    # 4. Display
    # ------------------------------------------------------------------
    print("Samples per organism (descending):")
    print(sample_counts.to_string())

    print("\nTotal cells per organism (descending):")
    print(cell_counts.to_string())


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise number of samples and total cell counts per organism in the Arc scBaseCount dataset."
    )
    parser.add_argument(
        "--gcs_base_path",
        default="gs://arc-scbasecount/2025-02-25/",
        help="Root GCS path containing the dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--feature_type",
        default="GeneFull_Ex50pAS",
        help="STARsolo feature_type directory name (default: %(default)s)",
    )

    args = parser.parse_args()
    main(args.gcs_base_path, args.feature_type)

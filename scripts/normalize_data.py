#!/usr/bin/env python3
"""
Script to copy and normalize scRNA-seq data files.

This script copies all data from the 'data' directory to 'data_normalized',
applying CP10K normalization (normalize to 10,000 counts + log1p) to all
.h5 and .h5ad files to avoid having to normalize during training.
"""

import os
import shutil
import h5py
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import List
import json
from tqdm import tqdm


def create_directory_structure(src_dir: Path, dst_dir: Path):
    """Create the same directory structure in destination as in source."""
    for dirpath, dirnames, filenames in os.walk(src_dir):
        # Calculate relative path from source
        rel_path = Path(dirpath).relative_to(src_dir)
        # Create corresponding directory in destination
        dst_path = dst_dir / rel_path
        dst_path.mkdir(parents=True, exist_ok=True)


def copy_non_data_files(src_dir: Path, dst_dir: Path):
    """Copy all non-.h5/.h5ad files from source to destination."""
    print("Copying non-data files...")
    
    for dirpath, dirnames, filenames in os.walk(src_dir):
        rel_path = Path(dirpath).relative_to(src_dir)
        dst_path = dst_dir / rel_path
        
        for filename in filenames:
            if not filename.endswith(('.h5', '.h5ad')):
                src_file = Path(dirpath) / filename
                dst_file = dst_path / filename
                
                print(f"Copying: {rel_path / filename}")
                shutil.copy2(src_file, dst_file)


def normalize_h5ad_file(src_file: Path, dst_file: Path):
    """Normalize a .h5ad file using scanpy (CP10K + log1p)."""
    print(f"Normalizing .h5ad file: {src_file.name}")
    
    # Load the data
    adata = sc.read_h5ad(src_file)
    
    # Apply CP10K normalization
    # Normalize each cell to 10,000 total counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Log1p transformation
    sc.pp.log1p(adata)
    
    # Save the normalized data
    adata.write_h5ad(dst_file)
    
    print(f"    -> Saved normalized data to: {dst_file}")
    print(f"    -> Expression values now in range [0, {adata.X.max():.2f}]")

def normalize_h5_batch_file(src_file: Path, dst_file: Path):
    """Normalize a .h5 batch file (CP10K + log1p)."""
    print(f"Normalizing .h5 file: {src_file.name}")
    
    # Read the source file
    with h5py.File(src_file, 'r') as src:
        # Read the expression matrix
        X = src['X'][:]
        
        # Convert to float32 for normalization
        X = X.astype(np.float32)
        
        # CP10K normalization: scale each cell to 10,000 total counts
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        X = (X / row_sums) * 10000
        
        # Log1p transformation
        X = np.log1p(X)
        
        # Assert and print to verify normalization
        normalized_row_sums = X.sum(axis=1)
        print(f"    -> Row sums after normalization: min={normalized_row_sums.min():.2f}, max={normalized_row_sums.max():.2f}")
        
        # Create destination file with normalized data
        with h5py.File(dst_file, 'w') as dst:
            # Copy the normalized expression matrix
            dst.create_dataset('X', data=X, compression='gzip', compression_opts=4)
            
            # Copy any other datasets/attributes from source
            for key in src.keys():
                if key != 'X':
                    if isinstance(src[key], h5py.Dataset):
                        dst.create_dataset(key, data=src[key][:])
                    
            # Copy attributes
            for attr_name, attr_value in src.attrs.items():
                dst.attrs[attr_name] = attr_value
    
    print(f"    -> Saved normalized data to: {dst_file}")
    print(f"    -> Expression values now in range [0, {X.max():.2f}]")


def normalize_all_data_files(src_dir: Path, dst_dir: Path):
    """Find and normalize all .h5 and .h5ad files."""
    print("\nNormalizing data files...")
    
    # Find all .h5ad files
    h5ad_files = list(src_dir.rglob("*.h5ad"))
    if h5ad_files:
        print(f"\nFound {len(h5ad_files)} .h5ad file(s):")
        for src_file in h5ad_files:
            rel_path = src_file.relative_to(src_dir)
            dst_file = dst_dir / rel_path
            normalize_h5ad_file(src_file, dst_file)
    
    # Find all .h5 files
    h5_files = list(src_dir.rglob("*.h5"))
    if h5_files:
        print(f"\nFound {len(h5_files)} .h5 file(s):")
        iterator = tqdm(h5_files, desc="Normalizing .h5 files")
        for src_file in iterator:
            rel_path = src_file.relative_to(src_dir)
            dst_file = dst_dir / rel_path
            normalize_h5_batch_file(src_file, dst_file)


def main():
    """Main function to copy and normalize data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Copy and normalize scRNA-seq data files")
    parser.add_argument("--src", default="data", help="Source directory (default: data)")
    parser.add_argument("--dst", default="data_normalized", help="Destination directory (default: data_normalized)")
    args = parser.parse_args()
    
    # Define source and destination directories
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory '{src_dir}' not found!")
    

    # Check if destination already exists
    if dst_dir.exists():
        response = input(f"Directory '{dst_dir}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        # Remove existing directory
        shutil.rmtree(dst_dir)
    
    print(f"Creating normalized data in '{dst_dir}'...")
    
    create_directory_structure(src_dir, dst_dir)
    
    # Copy non-data files
    copy_non_data_files(src_dir, dst_dir)
    
    # Normalize data files
    normalize_all_data_files(src_dir, dst_dir)
    
    print("\n✓ Normalization complete!")
    print(f"Normalized data saved to: {dst_dir.absolute()}")
    
    # Create a marker file to indicate this is normalized data
    marker_file = dst_dir / ".normalized"
    with open(marker_file, 'w') as f:
        f.write("This directory contains CP10K normalized data (10K counts + log1p).\n")
        f.write("Generated by scripts/normalize_data.py\n")
    
    print("\nTo use the normalized data, update your training scripts to use 'data_normalized' instead of 'data'")
    print("and set normalize=False in your dataloader configurations.")


if __name__ == "__main__":
    main()
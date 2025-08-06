#!/usr/bin/env python3
"""
build_symbol2ensembl.py

Parse a GTF file and create a mapping from gene symbols (HGNC) to stable
Ensembl gene IDs.  The script expects that Ensembl IDs are available in the
`gene_xref` attribute of the GTF record in the form `ENSEMBL:ENSG...`.

Usage
-----
python build_symbol2ensembl.py --gtf <path/to/genomic.gtf> --out symbol2ens.pkl
"""

import argparse
import gzip
import pickle
import re
from pathlib import Path
from collections import defaultdict
import json
import sys

ENSEMBL_RE = re.compile(r"(?:Ensembl|ENSEMBL):(ENSG[0-9A-Z]+)")
TRANSCRIPT_RE = re.compile(r"(?:Ensembl|ENSEMBL):(ENST[0-9A-Z]+)")


def open_text(path: Path):
    """
    Transparently open plain or gzipped text files for reading.
    """
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")

def build_lookup(gtf_path: Path):
    """
    Iterate over a GTF file and build symbol -> ENSG dictionary.

    Only `gene` feature lines are considered. For every gene we
    extract `gene_name` and the Ensembl gene ID from `gene_id`.
    
    For GENCODE GTFs, gene_id already contains the Ensembl ID.
    For RefSeq GTFs, we look for ENSEMBL: in gene_xref or db_xref.

    If multiple Ensembl IDs map to the same symbol we keep the first
    occurrence and report duplicates at the end.
    """
    symbol2ens = {}
    duplicates = defaultdict(set)

    with open_text(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue

            attr_field = fields[8]
            attrs = {}
            for token in attr_field.split(";"):
                token = token.strip()
                if not token:
                    continue
                if " " in token:
                    key, val = token.split(" ", 1)
                    attrs[key] = val.strip('"')
            
            # Get gene symbol
            gene_name = attrs.get("gene_name") or attrs.get("gene") 
            
            # Get Ensembl ID - try gene_id first (GENCODE format)
            gene_id = attrs.get("gene_id", "")
            if gene_id.startswith("ENSG"):
                # Remove version suffix if present (e.g., ENSG00000290825.2 -> ENSG00000290825)
                ens_id = gene_id.split(".")[0]
            else:
                # Fall back to looking in xref attributes (RefSeq format)
                xref = attrs.get("gene_xref", attrs.get("db_xref", ""))
                m = ENSEMBL_RE.search(xref)
                ens_id = m.group(1) if m else None

            if gene_name and ens_id:
                if gene_name not in symbol2ens:
                    symbol2ens[gene_name] = ens_id
                else:
                    duplicates[gene_name].add(ens_id)

    if duplicates:
        sys.stderr.write(
            f"Warning: {len(duplicates)} symbols had multiple Ensembl IDs. "
            "First occurrence was kept. Examples: "
            f"{list(duplicates.items())[:5]}\n"
        )

    return symbol2ens

def main():
    parser = argparse.ArgumentParser(description="Build symbol→Ensembl lookup from a GTF file.")
    parser.add_argument("--gtf", required=True, type=Path, help="Path to genomic.gtf")
    parser.add_argument("--out", required=True, type=Path, help="Output pickle file (.pkl or .json)")
    args = parser.parse_args()

    print(f"Parsing {args.gtf} ...")
    lookup = build_lookup(args.gtf)
    if not lookup:
        print("Warning: No Ensembl gene IDs were found in the GTF. "
              "Gene symbols will be left unchanged during processing.")
    else:
        print(f"Mapped {len(lookup):,} unique symbols to Ensembl IDs.")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.out.suffix == ".json":
        args.out.write_text(json.dumps(lookup))
    else:
        with open(args.out, "wb") as fh:
            pickle.dump(lookup, fh)

    print(f"Saved lookup table to {args.out}")

if __name__ == "__main__":
    main()

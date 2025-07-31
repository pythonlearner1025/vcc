#!/usr/bin/env python3
"""
Embed an Ensembl gene with ESM-2.

Example:
    python embed_gene.py ENSG00000177565 --out dec2_esm.pt --device cuda
"""
import argparse
from pathlib import Path
import requests
import torch
import esm                       # from fair-esm
from Bio import SeqIO            # only used for FASTA parsing

ENSEMBL = "https://rest.ensembl.org"           # Ensembl REST root

# ---------- Ensembl helpers --------------------------------------------------
def lookup_gene(gene_id: str) -> dict:
    """Return the JSON object describing the gene (expand=1 fetches transcripts)."""
    url = f"{ENSEMBL}/lookup/id/{gene_id}?expand=1"
    r = requests.get(url, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()                # contains 'biotype', 'Transcript', etc.

def fetch_canonical_protein(gene_json: dict) -> tuple[str, str]:
    """
    Return (protein_id, fasta_text) for the gene’s canonical transcript.
    If no canonical flag, we take the first protein-coding transcript.
    """
    if gene_json.get("biotype") != "protein_coding":
        return None, None

    # pick canonical transcript where is_canonical == 1
    transcripts = gene_json.get("Transcript", [])
    canonical = next((t for t in transcripts if t.get("is_canonical") == 1), None)
    canonical = canonical or next((t for t in transcripts if "Translation" in t), None)
    if canonical is None:
        return None, None

    pep_id = canonical["Translation"]["id"]
    url = f"{ENSEMBL}/sequence/id/{pep_id}?content-type=text/x-fasta"
    r = requests.get(url, headers={"Accept": "text/x-fasta"})
    r.raise_for_status()
    return pep_id, r.text

# ---------- ESM-2 embedding ---------------------------------------------------
def embed(sequence: str, device: str = "cpu") -> torch.Tensor:
    """
    Produce the mean-pooled final-layer embedding (1280 d) for an AA sequence.
    """
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()   # downloads from HF
    model = model.to(device).half().eval()                   # FP16, inference
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("seq", sequence)])
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[33])                # final layer
    rep = out["representations"][33][0, 1:-1].mean(0)        # mean over residues
    return rep.cpu()                                         # detach to host

# ---------- main -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensembl_gene_id", help="e.g. ENSG00000177565")
    p.add_argument("--out", default="embedding.pt", help="output .pt file")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # 1) gene lookup ----------------------------------------------------------
    gene = lookup_gene(args.ensembl_gene_id)
    if gene.get("biotype") != "protein_coding":
        print(f"{args.ensembl_gene_id}: biotype={gene.get('biotype')}. Not protein-coding.")
        return

    # 2) download protein sequence -------------------------------------------
    pep_id, fasta = fetch_canonical_protein(gene)
    if fasta is None:
        print("No protein sequence found.")
        return
    seq = "".join(fasta.splitlines()[1:])            # strip FASTA header

    # 3) embed ---------------------------------------------------------------
    emb = embed(seq, device=args.device)

    # 4) save ----------------------------------------------------------------
    torch.save(
        {
            "gene_id": args.ensembl_gene_id,
            "protein_id": pep_id,
            "embedding": emb.half(),                 # keep as float16
        },
        args.out,
    )
    print(f"Saved {emb.shape[0]}-dim embedding for {args.ensembl_gene_id} → {args.out}")

if __name__ == "__main__":
    main()

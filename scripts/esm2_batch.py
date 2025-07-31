#!/usr/bin/env python3
"""
Batch process all HVG genes to generate ESM-2 embeddings.

Example:
    python scripts/esm2_batch.py --hvg_file hvg_seuratv3_2000.txt --out esm_all.pt --device cuda
"""
import argparse
from pathlib import Path
import requests
import torch
import esm                       # from fair-esm
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

ENSEMBL = "https://rest.ensembl.org"           # Ensembl REST root
RATE_LIMIT = None  # Will be initialized in main() based on args

# ---------- Ensembl helpers --------------------------------------------------
def lookup_gene(gene_id: str, retries: int = 3) -> Optional[dict]:
    """Return the JSON object describing the gene (expand=1 fetches transcripts)."""
    url = f"{ENSEMBL}/lookup/id/{gene_id}?expand=1"
    for attempt in range(retries):
        try:
            with RATE_LIMIT:  # Acquire semaphore for rate limiting
                r = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
                r.raise_for_status()
                return r.json()
        except requests.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                wait_time = 2 ** (attempt + 1)  # Longer backoff for rate limit
                print(f"Rate limited on {gene_id}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            else:
                print(f"Failed to lookup {gene_id}: {e}")
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            print(f"Failed to lookup {gene_id}: {e}")
            return None

def fetch_all_protein_isoforms(gene_json: dict, gene_id: str) -> List[Tuple[str, str]]:
    """
    Return list of (protein_id, sequence) for all protein isoforms of the gene.
    """
    if gene_json.get("biotype") != "protein_coding":
        return []

    isoforms = []
    transcripts = gene_json.get("Transcript", [])
    
    # Get all transcripts with translations
    protein_transcripts = [t for t in transcripts if "Translation" in t]
    
    if not protein_transcripts:
        return []
    
    # Sort by canonical status (canonical first) and then by ID
    protein_transcripts.sort(key=lambda t: (t.get("is_canonical", 0) != 1, t["id"]))
    
    for transcript in protein_transcripts:
        pep_id = transcript["Translation"]["id"]
        
        # Fetch sequence with retries
        for attempt in range(3):
            try:
                with RATE_LIMIT:  # Acquire semaphore for rate limiting
                    url = f"{ENSEMBL}/sequence/id/{pep_id}?content-type=text/x-fasta"
                    r = requests.get(url, headers={"Accept": "text/x-fasta"}, timeout=30)
                    r.raise_for_status()
                    # Extract sequence from FASTA
                    seq = "".join(r.text.splitlines()[1:])
                    isoforms.append((pep_id, seq))
                    break
            except requests.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    wait_time = 2 ** (attempt + 1)
                    print(f"Rate limited on {pep_id}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"Failed to fetch protein sequence for {gene_id} isoform {pep_id}: {e}")
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                print(f"Failed to fetch protein sequence for {gene_id} isoform {pep_id}: {e}")
                # Continue to next isoform instead of failing completely
                break
    
    return isoforms

def fetch_gene_data(gene_id: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Fetch gene info and all protein isoforms for a single gene."""
    gene_json = lookup_gene(gene_id)
    if gene_json is None:
        return gene_id, []
        
    if gene_json.get("biotype") != "protein_coding":
        return gene_id, []
    
    isoforms = fetch_all_protein_isoforms(gene_json, gene_id)
    return gene_id, isoforms

# ---------- ESM-2 embedding ---------------------------------------------------
def embed_batch(sequences: List[Tuple[str, str]], model, alphabet, device: str = "cuda") -> List[torch.Tensor]:
    """
    Produce mean-pooled final-layer embeddings for a batch of sequences.
    Returns list of embeddings (some may be None if sequence was too long).
    """
    batch_converter = alphabet.get_batch_converter()
    
    # Filter out very long sequences (ESM2 has max length constraints)
    MAX_LEN = 1022  # ESM2 max length minus special tokens
    valid_seqs = []
    valid_indices = []
    
    for i, (label, seq) in enumerate(sequences):
        if len(seq) <= MAX_LEN:
            valid_seqs.append((label, seq))
            valid_indices.append(i)
    
    if not valid_seqs:
        return [None] * len(sequences)
    
    # Convert to tokens
    _, _, tokens = batch_converter(valid_seqs)
    tokens = tokens.to(device)
    
    # Get embeddings
    with torch.no_grad():
        out = model(tokens, repr_layers=[33])
    
    # Extract representations and mean pool
    embeddings = []
    for i in range(len(valid_seqs)):
        rep = out["representations"][33][i, 1:-1].mean(0)  # mean over residues
        embeddings.append(rep.cpu().half())
    
    # Reconstruct full list with None for invalid sequences
    result = [None] * len(sequences)
    for idx, emb in zip(valid_indices, embeddings):
        result[idx] = emb
    
    return result

# ---------- main -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hvg_file", default="hvg_seuratv3_2000.txt", help="File with gene IDs")
    p.add_argument("--out", default="esm_all.pt", help="output .pt file")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for ESM2 (A100 can handle 16-32)")
    p.add_argument("--model_size", default="650M", choices=["650M", "3B"], help="ESM2 model size")
    p.add_argument("--parallel_workers", type=int, default=10, help="Number of parallel workers for Ensembl API calls (reduce if getting 429 errors)")
    p.add_argument("--rate_limit", type=int, default=10, help="Max concurrent Ensembl requests (reduce if getting 429 errors)")
    args = p.parse_args()

    # Initialize rate limiter
    global RATE_LIMIT
    RATE_LIMIT = Semaphore(args.rate_limit)
    
    # Read gene IDs
    with open(args.hvg_file, 'r') as f:
        gene_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(gene_ids)} genes...")
    
    # Load ESM2 model
    print(f"Loading ESM2-{args.model_size} model...")
    if args.model_size == "3B":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        embed_dim = 2560
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        embed_dim = 1280
    
    model = model.to(args.device).half().eval()
    
    # Collect sequences and metadata
    gene_data = [None] * len(gene_ids)  # Pre-allocate to maintain order
    gene_id_to_idx = {gene_id: i for i, gene_id in enumerate(gene_ids)}
    failed_genes = []
    
    print(f"Fetching protein sequences from Ensembl (parallel with {args.parallel_workers} workers)...")
    with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        # Submit all tasks
        future_to_gene = {executor.submit(fetch_gene_data, gene_id): gene_id 
                          for gene_id in gene_ids}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_gene), total=len(gene_ids)):
            gene_id = future_to_gene[future]
            idx = gene_id_to_idx[gene_id]
            
            try:
                result = future.result()
                gene_data[idx] = result
                if len(result[1]) == 0:  # No isoforms found
                    # Check if it was a lookup failure
                    test_json = lookup_gene(gene_id)
                    if test_json is None:
                        failed_genes.append(gene_id)
            except Exception as e:
                print(f"Error processing {gene_id}: {e}")
                gene_data[idx] = (gene_id, [])
                failed_genes.append(gene_id)
    
    # Prepare all isoforms for embedding
    print("\nPreparing isoform data...")
    all_isoforms = []  # List of (gene_id, protein_id, sequence) tuples
    gene_to_isoform_indices = {}  # Maps gene_id to list of indices in all_isoforms
    
    current_idx = 0
    for gene_id, isoforms in gene_data:
        if isoforms:
            gene_to_isoform_indices[gene_id] = []
            for protein_id, seq in isoforms:
                all_isoforms.append((gene_id, protein_id, seq))
                gene_to_isoform_indices[gene_id].append(current_idx)
                current_idx += 1
        else:
            # Gene with no isoforms gets a placeholder
            gene_to_isoform_indices[gene_id] = [current_idx]
            all_isoforms.append((gene_id, None, None))
            current_idx += 1
    
    print(f"Total isoforms to process: {len(all_isoforms)}")
    
    # Process embeddings in batches
    print(f"\nGenerating ESM2 embeddings in batches of {args.batch_size}...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(all_isoforms), args.batch_size)):
        batch = all_isoforms[i:i + args.batch_size]
        
        # Prepare sequences for this batch
        sequences = []
        for gene_id, protein_id, seq in batch:
            if seq is not None:
                sequences.append((f"{gene_id}_{protein_id}", seq))
            else:
                sequences.append((gene_id, ""))  # placeholder
        
        # Get embeddings
        if any(seq[1] for seq in sequences):  # if any valid sequences
            embeddings = embed_batch(sequences, model, alphabet, args.device)
        else:
            embeddings = [None] * len(sequences)
        
        # Store embeddings
        for j, (gene_id, protein_id, seq) in enumerate(batch):
            if embeddings[j] is not None:
                all_embeddings.append(embeddings[j])
            else:
                # Zero vector for non-protein-coding or failed genes
                all_embeddings.append(torch.zeros(embed_dim, dtype=torch.half))
    
    # Create embedding matrix
    emb_matrix = torch.stack(all_embeddings)
    
    # Create protein_ids list in same order as embeddings
    protein_ids = [isoform[1] if isoform[1] else f"{isoform[0]}_no_protein" 
                   for isoform in all_isoforms]
    
    # Save in the specified format with isoform mapping
    save_dict = {
        "emb": emb_matrix,  # Shape: [n_total_isoforms, embed_dim]
        "genes": gene_ids,  # Original gene order
        "protein_ids": protein_ids,  # Protein IDs in same order as emb
        "gene_to_isoform_indices": gene_to_isoform_indices,  # Mapping
    }
    
    torch.save(save_dict, args.out)
    
    # Print summary
    n_protein_coding = sum(1 for _, isoforms in gene_data if isoforms)
    n_isoforms = sum(len(isoforms) for _, isoforms in gene_data if isoforms)
    n_embedded = sum(1 for emb in all_embeddings if (emb != 0).any())
    
    print(f"\nSummary:")
    print(f"Total genes: {len(gene_ids)}")
    print(f"Protein-coding genes: {n_protein_coding}")
    print(f"Total protein isoforms: {n_isoforms}")
    print(f"Successfully embedded: {n_embedded}")
    print(f"Failed lookups: {len(failed_genes)}")
    print(f"Embedding matrix shape: {emb_matrix.shape}")
    print(f"Genes with multiple isoforms: {sum(1 for g, indices in gene_to_isoform_indices.items() if len(indices) > 1)}")
    print(f"Saved to: {args.out}")

if __name__ == "__main__":
    main()
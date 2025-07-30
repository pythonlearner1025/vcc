#!/usr/bin/env python3
"""
Test script for the per-experiment HVG consensus approach
"""

import numpy as np
from collections import defaultdict
import sys
import os

# Add scripts directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def find_consensus_hvgs(experiment_hvgs: dict, n_top: int) -> list:
    """
    Find consensus HVGs across experiments using a frequency-based ranking approach.
    
    Strategy:
    1. Count how many experiments each gene appears in as HVG
    2. Rank genes by frequency (how often they're selected as HVG)
    3. For genes with same frequency, use average rank across experiments
    4. Select top n_top genes
    """
    gene_frequency = defaultdict(int)  # How many experiments selected this gene
    gene_ranks = defaultdict(list)     # Rank of gene in each experiment that selected it
    
    # Count frequencies and collect ranks
    for exp_id, hvg_set in experiment_hvgs.items():
        hvg_list = list(hvg_set)  # Convert set to list to get ranking
        
        for rank, gene in enumerate(hvg_list):
            gene_frequency[gene] += 1
            gene_ranks[gene].append(rank)
    
    # Score genes: primary by frequency, secondary by average rank
    gene_scores = []
    for gene in gene_frequency:
        frequency = gene_frequency[gene]
        avg_rank = np.mean(gene_ranks[gene])  # Lower rank = better
        
        # Score: frequency (higher better) + 1/(avg_rank + 1) (lower rank better)
        # This prioritizes genes that appear in many experiments and have good ranks
        score = frequency + 1.0 / (avg_rank + 1)
        gene_scores.append((score, -frequency, avg_rank, gene))  # negative freq for sorting
    
    # Sort by score (descending), then by frequency (descending), then by avg rank (ascending)
    gene_scores.sort(reverse=True)
    
    consensus_hvgs = [gene for _, _, _, gene in gene_scores[:n_top]]
    
    # Log statistics
    print("HVG consensus statistics:")
    print(f"  Total unique HVGs across experiments: {len(gene_frequency)}")
    
    freq_counts = defaultdict(int)
    for freq in gene_frequency.values():
        freq_counts[freq] += 1
    
    for freq in sorted(freq_counts.keys(), reverse=True):
        count = freq_counts[freq]
        print(f"  Genes in {freq}/{len(experiment_hvgs)} experiments: {count} genes")
    
    # Show top consensus genes
    print("Top 10 consensus HVGs:")
    for i, (score, _, avg_rank, gene) in enumerate(gene_scores[:10]):
        freq = gene_frequency[gene]
        print(f"  {i+1}. {gene} (in {freq}/{len(experiment_hvgs)} exp, avg_rank={avg_rank:.1f}, score={score:.3f})")
    
    print(f"Selected {len(consensus_hvgs)} consensus HVGs")
    return consensus_hvgs


def test_consensus_hvgs():
    """Test the consensus HVG selection with synthetic data"""
    print("Testing HVG consensus approach with synthetic data...\n")
    
    # Create synthetic experiment HVG data
    # Simulate 5 experiments with some overlapping HVGs
    experiment_hvgs = {
        'SRX001': ['ACTB', 'GAPDH', 'RPL32', 'RPS18', 'MALAT1', 'NEAT1', 'MT-CO1', 'MT-ND1', 'XIST', 'EEF1A1'],
        'SRX002': ['ACTB', 'GAPDH', 'RPL32', 'HSPA1A', 'HSPA1B', 'FOS', 'JUN', 'EGR1', 'MT-CO1', 'XIST'],
        'SRX003': ['ACTB', 'RPS18', 'MALAT1', 'NEAT1', 'FOS', 'JUN', 'CD74', 'HLA-DRA', 'B2M', 'TMSB4X'],
        'SRX004': ['GAPDH', 'RPL32', 'HSPA1A', 'MT-CO1', 'MT-ND1', 'CD74', 'HLA-DRA', 'B2M', 'ISG15', 'IFI6'],
        'SRX005': ['ACTB', 'MALAT1', 'FOS', 'JUN', 'EGR1', 'CD74', 'B2M', 'TMSB4X', 'ISG15', 'IFI6']
    }
    
    print("Input experiment HVGs:")
    for exp_id, hvgs in experiment_hvgs.items():
        print(f"  {exp_id}: {hvgs}")
    print()
    
    # Test consensus selection
    consensus_hvgs = find_consensus_hvgs(experiment_hvgs, n_top=15)
    
    print(f"\nFinal consensus HVGs (top 15): {consensus_hvgs}")
    
    # Analyze which genes were most consistently selected
    gene_counts = defaultdict(int)
    for hvgs in experiment_hvgs.values():
        for gene in hvgs:
            gene_counts[gene] += 1
    
    print("\nGene frequency analysis:")
    for gene, count in sorted(gene_counts.items(), key=lambda x: x[1], reverse=True):
        in_consensus = "✓" if gene in consensus_hvgs else " "
        print(f"  {in_consensus} {gene}: {count}/5 experiments")


if __name__ == "__main__":
    test_consensus_hvgs()

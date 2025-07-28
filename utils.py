"""Common utilities for data processing and tokenization."""

import torch
import json
from pathlib import Path
from typing import Dict


def create_simple_tokenizer(vocab_size: int = 64):
    """Create a binning tokenizer for gene expression values."""
    class SimpleTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            # Log-scale bins from 0.01 to 10000
            self.bins = torch.logspace(-2, 4, vocab_size - 1)
            self.bins = torch.cat([torch.tensor([0.0]), self.bins])
            
        def __call__(self, x):
            if isinstance(x, torch.Tensor):
                tokens = torch.bucketize(x, self.bins)
            else:
                x = torch.from_numpy(x)
                tokens = torch.bucketize(x, self.bins)
            return tokens.clamp(0, self.vocab_size - 1)
    
    return SimpleTokenizer(vocab_size)


def load_hvg_info(data_path: str) -> Dict:
    """Load HVG information from preprocessed file."""
    hvg_info_path = Path(data_path).parent / 'hvg_info.json'
    
    if not hvg_info_path.exists():
        raise FileNotFoundError(
            f"HVG info not found at {hvg_info_path}. "
            "Please run preprocess_hvgs.py --process-vcc first."
        )
    
    with open(hvg_info_path, 'r') as f:
        return json.load(f)
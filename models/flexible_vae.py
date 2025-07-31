#!/usr/bin/env python3
"""
Flexible Variational Autoencoder (VAE) for single-cell RNA-seq perturbation prediction.

This module implements a flexible, modular VAE architecture designed for two-phase training:
1. Phase 1: Self-supervised pretraining on large scRNA-seq datasets (learn cellular representations)
2. Phase 2: Perturbation fine-tuning with target gene injection (learn perturbation effects)

Key Design Principles:
- Modular architecture: Easy to adjust latent space size and embeddings
- Flexible gene embeddings: Support for ESM2, Gene2Vec, or custom embeddings
- Robust data handling: Ensures valid control/perturbation pairs exist
- Clean separation: Encoder sees only cellular state, decoder applies perturbations
- Extensible: Easy to add new embedding types or architectures

Architecture:
- Encoder: gene_expression + experiment_id → latent_representation
- Decoder: latent_representation + experiment_id + target_gene_embedding → perturbed_expression

Authors: ag
Created: July 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class VAEConfig:
    """
    Configuration class for the Flexible VAE.
    
    This config supports various experimental setups and can be easily modified
    for different latent space sizes, embedding dimensions, and architectures.
    """
    # === Core Architecture ===
    input_dim: int = 1808  # Number of genes (will be set from data)
    latent_dim: int = 512  # Latent space dimension - ADJUSTABLE for experiments
    hidden_dims: List[int] = None  # Hidden layer dimensions [1024, 512, 256]
    
    # === Conditioning Dimensions ===
    experiment_embed_dim: int = 32  # Batch/experiment embedding size
    target_gene_embed_dim: int = 128  # Target gene embedding size (ESM2, Gene2Vec, etc.)
    
    # === Vocabulary Sizes (set from data) ===
    n_experiments: int = 500  # Number of unique experiments/batches
    n_genes: int = 1808  # Number of genes in vocabulary
    
    # === Training Parameters ===
    learning_rate: float = 1e-3
    batch_size: int = 256
    dropout_rate: float = 0.1
    
    # === Loss Weighting ===
    reconstruction_weight: float = 1.0
    kld_weight: float = 1.0  # Beta-VAE parameter
    
    # === Architecture Options ===
    use_batch_norm: bool = True
    activation: str = "relu"  # "relu", "gelu", "swish"
    
    def __post_init__(self):
        """Set default hidden dimensions if not provided."""
        if self.hidden_dims is None:
            # Default: gradually decreasing dimensions
            self.hidden_dims = [
                min(2048, self.input_dim // 2),
                min(1024, self.input_dim // 4),
                min(512, self.input_dim // 8)
            ]


class GeneEmbedding(ABC):
    """
    Abstract base class for different gene embedding strategies.
    
    This allows flexible swapping between different gene representation methods:
    - ESM2 embeddings
    - Gene2Vec embeddings  
    - Learned embeddings
    - One-hot encodings
    """
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the gene embeddings."""
        pass
    
    @abstractmethod
    def get_embeddings(self, gene_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for gene IDs.
        
        Args:
            gene_ids: Tensor of gene indices [batch_size]
            
        Returns:
            Gene embeddings [batch_size, embedding_dim]
        """
        pass


class LearnedGeneEmbedding(GeneEmbedding, nn.Module):
    """Learned gene embeddings (similar to word embeddings)."""
    
    def __init__(self, n_genes: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_genes, embed_dim)
        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, 0, 0.1)
    
    def get_embedding_dim(self) -> int:
        return self.embed_dim
    
    def get_embeddings(self, gene_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(gene_ids)


class PretrainedGeneEmbedding(GeneEmbedding):
    """
    Pretrained gene embeddings (ESM2, Gene2Vec, etc.).
    
    This class wraps pretrained embeddings and provides the interface
    for the VAE to use them.
    """
    
    def __init__(self, embedding_matrix: torch.Tensor, freeze: bool = True):
        """
        Initialize with pretrained embeddings.
        
        Args:
            embedding_matrix: Pretrained embeddings [n_genes, embed_dim]
            freeze: Whether to freeze embeddings during training
        """
        self.embedding_matrix = embedding_matrix
        self.freeze = freeze
        self.embed_dim = embedding_matrix.size(1)
        
        if not freeze:
            # Make embeddings learnable
            self.embedding_matrix = nn.Parameter(embedding_matrix)
    
    def get_embedding_dim(self) -> int:
        return self.embed_dim
    
    def get_embeddings(self, gene_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[gene_ids]


class FlexibleEncoder(nn.Module):
    """
    Flexible encoder that learns cellular representations.
    
    Design Rationale:
    - Only sees gene expression and experiment context
    - No perturbation information to avoid confounding
    - Learns pure cellular state representations
    - Modular activation functions and normalization
    """
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Experiment embedding for batch effect modeling
        self.experiment_embedding = nn.Embedding(
            config.n_experiments, 
            config.experiment_embed_dim
        )
        
        # Input dimension: gene expression + experiment embedding
        input_dim = config.input_dim + config.experiment_embed_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend(self._make_layer(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space projections
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, config.latent_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_dim: int, out_dim: int) -> List[nn.Module]:
        """Create a single encoder layer with configurable components."""
        layers = [nn.Linear(in_dim, out_dim)]
        
        if self.config.use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        
        # Configurable activation
        if self.config.activation == "relu":
            layers.append(nn.ReLU())
        elif self.config.activation == "gelu":
            layers.append(nn.GELU())
        elif self.config.activation == "swish":
            layers.append(nn.SiLU())
        
        layers.append(nn.Dropout(self.config.dropout_rate))
        
        return layers
    
    def _initialize_weights(self):
        """Initialize weights using best practices."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def forward(self, gene_expression: torch.Tensor, 
                experiment_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode gene expression to latent parameters.
        
        Args:
            gene_expression: Gene expression data [batch_size, n_genes]
            experiment_ids: Experiment/batch IDs [batch_size]
            
        Returns:
            mu: Latent means [batch_size, latent_dim]
            logvar: Latent log variances [batch_size, latent_dim]
        """
        # Get experiment embeddings for batch effect modeling
        exp_emb = self.experiment_embedding(experiment_ids)
        
        # Concatenate gene expression with experiment context
        x = torch.cat([gene_expression, exp_emb], dim=1)
        
        # Encode to hidden representation
        h = self.encoder(x)
        
        # Project to latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class FlexibleDecoder(nn.Module):
    """
    Flexible decoder that applies perturbations to cellular representations.
    
    Design Rationale:
    - Takes latent cellular state + experiment context + target gene
    - Target gene embedding can be swapped (ESM2, Gene2Vec, learned)
    - Applies perturbation effects in a biologically meaningful way
    - Modular design for easy experimentation
    """
    
    def __init__(self, config: VAEConfig, gene_embedding: GeneEmbedding):
        super().__init__()
        self.config = config
        self.gene_embedding = gene_embedding
        
        # Experiment embedding (shared with encoder)
        self.experiment_embedding = nn.Embedding(
            config.n_experiments, 
            config.experiment_embed_dim
        )
        
        # Input dimension: latent + experiment + target gene embedding
        input_dim = (config.latent_dim + 
                    config.experiment_embed_dim + 
                    gene_embedding.get_embedding_dim())
        
        # Build decoder layers (reverse of encoder)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in reversed(config.hidden_dims):
            layers.extend(self._make_layer(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, config.input_dim))
        
        self.decoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_dim: int, out_dim: int) -> List[nn.Module]:
        """Create a single decoder layer with configurable components."""
        layers = [nn.Linear(in_dim, out_dim)]
        
        if self.config.use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        
        # Configurable activation
        if self.config.activation == "relu":
            layers.append(nn.ReLU())
        elif self.config.activation == "gelu":
            layers.append(nn.GELU())
        elif self.config.activation == "swish":
            layers.append(nn.SiLU())
        
        layers.append(nn.Dropout(self.config.dropout_rate))
        
        return layers
    
    def _initialize_weights(self):
        """Initialize weights using best practices."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def forward(self, latent: torch.Tensor, 
                experiment_ids: torch.Tensor,
                target_gene_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode latent representation to gene expression with optional perturbation.
        
        Args:
            latent: Latent cellular state [batch_size, latent_dim]
            experiment_ids: Experiment/batch IDs [batch_size]
            target_gene_ids: Target gene IDs for perturbation [batch_size]
                           If None, performs control (no perturbation) decoding
        
        Returns:
            Reconstructed gene expression [batch_size, n_genes]
        """
        batch_size = latent.size(0)
        device = latent.device
        
        # Get experiment embeddings
        exp_emb = self.experiment_embedding(experiment_ids)
        
        # Get target gene embeddings
        if target_gene_ids is not None:
            gene_emb = self.gene_embedding.get_embeddings(target_gene_ids)
        else:
            # No perturbation: use zero/neutral gene embedding
            gene_emb = torch.zeros(
                batch_size, 
                self.gene_embedding.get_embedding_dim(), 
                device=device
            )
        
        # Concatenate all conditioning information
        decoder_input = torch.cat([latent, exp_emb, gene_emb], dim=1)
        
        # Decode to gene expression
        reconstructed = self.decoder(decoder_input)
        
        return reconstructed


class FlexibleVAE(nn.Module):
    """
    Flexible VAE for single-cell perturbation prediction.
    
    This is the main model class that orchestrates the encoder and decoder.
    Designed for two-phase training:
    
    Phase 1 (Pretraining):
    - Train on large scRNA-seq datasets without perturbation labels
    - Learn general cellular representations and batch effects
    - Use: forward(expression, experiment_ids, target_gene_ids=None)
    
    Phase 2 (Perturbation Fine-tuning):
    - Fine-tune on paired control/perturbation data
    - Learn specific target gene perturbation effects
    - Use: forward(expression, experiment_ids, target_gene_ids)
    
    Key Features:
    - Flexible latent space size (config.latent_dim)
    - Pluggable gene embeddings (ESM2, Gene2Vec, learned)
    - Robust handling of missing perturbation data
    - Clean separation of cellular state and perturbation effects
    """
    
    def __init__(self, config: VAEConfig, gene_embedding: GeneEmbedding):
        super().__init__()
        self.config = config
        self.gene_embedding = gene_embedding
        
        # Core architecture
        self.encoder = FlexibleEncoder(config)
        self.decoder = FlexibleDecoder(config, gene_embedding)
        
        # Share experiment embeddings between encoder and decoder
        self.decoder.experiment_embedding = self.encoder.experiment_embedding
        
        logger.info(f"Initialized FlexibleVAE with latent_dim={config.latent_dim}")
        logger.info(f"Gene embedding dim: {gene_embedding.get_embedding_dim()}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, gene_expression: torch.Tensor,
                experiment_ids: torch.Tensor,
                target_gene_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            gene_expression: Gene expression data [batch_size, n_genes]
            experiment_ids: Experiment/batch IDs [batch_size]
            target_gene_ids: Target gene IDs for perturbation [batch_size]
                           If None, performs control reconstruction
        
        Returns:
            Dictionary containing:
            - reconstructed: Reconstructed gene expression
            - mu: Latent means
            - logvar: Latent log variances  
            - z: Sampled latent representation
        """
        # Encode to latent space (no perturbation information)
        mu, logvar = self.encoder(gene_expression, experiment_ids)
        
        # Sample latent representation
        z = self.reparameterize(mu, logvar)
        
        # Decode with optional perturbation
        reconstructed = self.decoder(z, experiment_ids, target_gene_ids)
        
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def encode(self, gene_expression: torch.Tensor,
               experiment_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode gene expression to latent parameters."""
        return self.encoder(gene_expression, experiment_ids)
    
    def decode(self, latent: torch.Tensor,
               experiment_ids: torch.Tensor,
               target_gene_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode latent representation to gene expression."""
        return self.decoder(latent, experiment_ids, target_gene_ids)
    
    def predict_perturbation(self, control_expression: torch.Tensor,
                           experiment_ids: torch.Tensor,
                           target_gene_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict perturbation effects by injecting target genes into latent space.
        
        This is the key functionality for perturbation prediction:
        1. Encode control cells (no perturbation info in encoder)
        2. Decode with target gene perturbation
        
        Args:
            control_expression: Control cell expression [batch_size, n_genes]
            experiment_ids: Experiment IDs [batch_size]
            target_gene_ids: Target genes to perturb [batch_size]
        
        Returns:
            Predicted perturbed expression [batch_size, n_genes]
        """
        self.eval()
        with torch.no_grad():
            # Encode control expression
            mu, _ = self.encode(control_expression, experiment_ids)
            
            # Decode with target gene perturbation
            perturbed_expression = self.decode(mu, experiment_ids, target_gene_ids)
        
        return perturbed_expression
    
    def generate(self, experiment_ids: torch.Tensor,
                 target_gene_ids: Optional[torch.Tensor] = None,
                 n_samples: int = 1) -> torch.Tensor:
        """
        Generate synthetic cells by sampling from prior.
        
        Args:
            experiment_ids: Experiment IDs [batch_size]
            target_gene_ids: Target gene IDs [batch_size] (optional)
            n_samples: Number of samples to generate per condition
        
        Returns:
            Generated gene expression [batch_size * n_samples, n_genes]
        """
        device = next(self.parameters()).device
        batch_size = len(experiment_ids)
        
        # Repeat conditions for multiple samples
        if n_samples > 1:
            experiment_ids = experiment_ids.repeat_interleave(n_samples)
            if target_gene_ids is not None:
                target_gene_ids = target_gene_ids.repeat_interleave(n_samples)
        
        # Sample from standard normal prior
        z = torch.randn(batch_size * n_samples, self.config.latent_dim, device=device)
        
        # Decode to gene expression
        with torch.no_grad():
            generated = self.decode(z, experiment_ids, target_gene_ids)
        
        return generated


def flexible_vae_loss(outputs: Dict[str, torch.Tensor],
                     targets: torch.Tensor,
                     config: VAEConfig) -> Dict[str, torch.Tensor]:
    """
    Compute VAE loss with proper weighting.
    
    Args:
        outputs: Model outputs dictionary
        targets: Target gene expression [batch_size, n_genes]
        config: VAE configuration
    
    Returns:
        Dictionary containing loss components
    """
    reconstructed = outputs['reconstructed']
    mu = outputs['mu']
    logvar = outputs['logvar']
    
    # Reconstruction loss (MSE for continuous expression data)
    reconstruction_loss = F.mse_loss(reconstructed, targets, reduction='sum')
    
    # KL divergence loss (regularizes latent space)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Weighted total loss (beta-VAE formulation)
    total_loss = (config.reconstruction_weight * reconstruction_loss + 
                  config.kld_weight * kld_loss)
    
    return {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'kld_loss': kld_loss
    }


# Utility functions for creating gene embeddings
def create_learned_gene_embedding(n_genes: int, embed_dim: int) -> LearnedGeneEmbedding:
    """Create learned gene embeddings."""
    return LearnedGeneEmbedding(n_genes, embed_dim)


def load_pretrained_gene_embedding(embedding_path: str, freeze: bool = True) -> PretrainedGeneEmbedding:
    """
    Load pretrained gene embeddings (ESM2, Gene2Vec, etc.).
    
    Args:
        embedding_path: Path to embedding matrix (.npy, .pt, or .pth)
        freeze: Whether to freeze embeddings during training
    
    Returns:
        PretrainedGeneEmbedding instance
    """
    if embedding_path.endswith('.npy'):
        embedding_matrix = torch.from_numpy(np.load(embedding_path)).float()
    elif embedding_path.endswith(('.pt', '.pth')):
        embedding_matrix = torch.load(embedding_path)
    else:
        raise ValueError(f"Unsupported embedding format: {embedding_path}")
    
    return PretrainedGeneEmbedding(embedding_matrix, freeze=freeze)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing FlexibleVAE...")
    
    # Create config for testing
    config = VAEConfig(
        input_dim=1000,
        latent_dim=128,
        hidden_dims=[512, 256],
        n_experiments=10,
        n_genes=1000,
        target_gene_embed_dim=64
    )
    
    # Create learned gene embedding
    gene_embedding = create_learned_gene_embedding(
        n_genes=config.n_genes,
        embed_dim=config.target_gene_embed_dim
    )
    
    # Create model
    model = FlexibleVAE(config, gene_embedding)
    
    # Test data
    batch_size = 32
    gene_expression = torch.randn(batch_size, config.input_dim)
    experiment_ids = torch.randint(0, config.n_experiments, (batch_size,))
    target_gene_ids = torch.randint(0, config.n_genes, (batch_size,))
    
    # Test Phase 1: Pretraining (no perturbation)
    print("\nTesting Phase 1 (Pretraining):")
    outputs_control = model(gene_expression, experiment_ids, target_gene_ids=None)
    print(f"Control reconstruction shape: {outputs_control['reconstructed'].shape}")
    
    # Test Phase 2: Perturbation fine-tuning
    print("\nTesting Phase 2 (Perturbation):")
    outputs_pert = model(gene_expression, experiment_ids, target_gene_ids)
    print(f"Perturbation reconstruction shape: {outputs_pert['reconstructed'].shape}")
    
    # Test perturbation prediction
    print("\nTesting perturbation prediction:")
    predicted = model.predict_perturbation(gene_expression, experiment_ids, target_gene_ids)
    print(f"Predicted perturbation shape: {predicted.shape}")
    
    # Test loss computation
    loss_dict = flexible_vae_loss(outputs_control, gene_expression, config)
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    print(f"\nFlexibleVAE test completed successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# Compatibility alias for legacy code
vae_loss_function = flexible_vae_loss

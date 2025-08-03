"""
Variational Autoencoder (VAE) for single-cell RNA-seq perturbation prediction.

This module implements a flexible, modular VAE architecture designed for two-phase training:
1. Phase 1 (Pretraining): Self-supervised pretraining on large scRNA-seq datasets (learn cellular representations)
    The idea is to allow the model to learn a latent representation across experiments of what is important for the cells, something akin to having HVG selection, and then being able to reconstruct the full transcriptomic profile. The different cell type representations should be learned via this compression implicitly, and shouldn't be provided to the model. If two different experiments share the cell type, their transcriptomic profiles should match. Here, a large corpus of data can be used, preferably human. Importantly, this training is agnostic to whether a cell is perturbed, cancerous or whatever.

    - Train on large scRNA-seq datasets without perturbation labels
    - Learn general cellular representations

2. Phase 2 (Perturbation Fine-tuning): Perturbation fine-tuning with target gene injection (learn perturbation effects)
    To fine tune the model, a finalised encoder and decoder will be needed to streamline training. To allow training for what the effects are, a model that changes the latent representation will be used. An uperturbed cell's representation (ideally precaculated for all, stored somewhere) will be used as an input to an ST-like model. The goal of the ST-like model is to induce a change in the latent representation of the cell, such that when the pretrained decoder is run, the expression profile matches the perturbed cell profiles (randomly sampled).

    - Fine-tune on paired control/perturbation data
    - Learn specific target gene perturbation effects

Some ideas to test:
    - Pretraining the phase 1 VAE on only the control cells, such that the model does not see the perturbation profiles beforehand. This could potentially help with generalization?

Architectures:

BASE VAE:
    - Encoder: gene_expression profile → latent_representation
    - Decoder: latent_representation → perturbed_expression
        * Has to have non-negative outputs, so ReLU act.?

Authors: AG
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
    Configuration class for the BASE VAE.
    
    This config supports various experimental setups and can be easily modified
    for different latent space sizes.
    """
    # === Core Architecture ===
    input_dim: int = 1808  # Number of genes (will be set from data)
    latent_dim: int = 512  # Latent space dimension - ADJUSTABLE for experiments
    hidden_dims: List[int] = None  # Hidden layer dimensions [1024, 512, 256]
    
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
            # Default: gradually decreasing dimensions)
            self.hidden_dims = [
                min(2048, self.input_dim // 2),
                min(1024, self.input_dim // 4),
                min(512, self.input_dim // 8)
            ]

class BaseEncoder(nn.Module):
    """
    Flexible encoder that learns cellular representations.
        - Only sees gene expression context
    """
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Input dimension: gene expression. [batch_size, n_genes]
        input_dim = config.input_dim
        
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
        elif self.config.activation == "softplus":
            layers.append(nn.Softplus())
        elif self.config.activation == "leaky_relu":
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        elif self.config.activation == "tanh":
            layers.append(nn.Tanh())
        elif self.config.activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif self.config.activation == "elu":
            layers.append(nn.ELU())
        
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
    
    def forward(self, gene_expression: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode gene expression to latent parameters.
        
        Args:
            gene_expression: Gene expression data [batch_size, n_genes]
            
        Returns:
            mu: Latent means [batch_size, latent_dim]
            logvar: Latent log variances [batch_size, latent_dim]
        """
        
        # Concatenate gene expression with experiment context
        x = gene_expression
        
        # Encode to hidden representation
        h = self.encoder(x)
        
        # Project to latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Returns the mean, logvar - these will be used for sampling from the learned probabilistic latent rep.
        return mu, logvar

class BaseDecoder(nn.Module):
    """
    Flexible decoder that predicts cellular representations from a latent representation.
        - Takes latent cellular state
    """
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Input dimension: latent
        input_dim = config.latent_dim
        
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
        elif self.config.activation == "softplus":
            layers.append(nn.Softplus())
        elif self.config.activation == "leaky_relu":
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        elif self.config.activation == "tanh":
            layers.append(nn.Tanh())
        elif self.config.activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif self.config.activation == "elu":
            layers.append(nn.ELU())
        
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
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to gene expression.
        
        Args:
            latent: Latent cellular state [batch_size, latent_dim]
        
        Returns:
            Reconstructed gene expression [batch_size, n_genes]
        """
        #batch_size = latent.size(0)
        #device = latent.device

        decoder_input = latent
        
        # Decode to gene expression
        reconstructed = self.decoder(decoder_input)
        
        # TODO: Implement this proper
        # Decoder output should be non-negative, and we shall apply a final activation:
        if hasattr(self.config, 'output_activation'):
            if self.config.output_activation == 'relu':
                reconstructed = F.relu(reconstructed)
            elif self.config.output_activation == 'sigmoid':
                reconstructed = torch.sigmoid(reconstructed)
            elif self.config.output_activation == 'softmax':
                reconstructed = F.softmax(reconstructed, dim=-1)
            elif self.config.output_activation == 'tanh':
                reconstructed = torch.tanh(reconstructed)
            else:
                raise ValueError(f"Unsupported output activation: {self.config.output_activation}")
            

        return reconstructed

# TODO: Class for the ST model and the ST + Decoder to streamline fine-tuning

class BaseVAE(nn.Module):
    """
    Base VAE for single-cell representation learning.
    
    This is the main model class that orchestrates the encoder and decoder.
    
    Key Features:
    - Flexible latent space size (config.latent_dim)
    """
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Core architecture
        self.encoder = BaseEncoder(config)
        self.decoder = BaseDecoder(config)
        
        logger.info(f"Initialized BaseVAE with latent_dim={config.latent_dim}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, gene_expression: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            gene_expression: Gene expression data [batch_size, n_genes]
        
        Returns:
            Dictionary containing:
            - reconstructed: Reconstructed gene expression
            - mu: Latent means
            - logvar: Latent log variances  
            - z: Sampled latent representation
        """
        # Check for NaN in input
        if torch.isnan(gene_expression).any():
            logger.warning("NaN detected in input gene_expression")
            gene_expression = torch.nan_to_num(gene_expression, nan=0.0)
        
        # Encode to latent space
        mu, logvar = self.encoder(gene_expression)
        
        # Clamp to prevent NaN/inf
        mu = torch.clamp(mu, min=-10, max=10)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # Sample latent representation
        z = self.reparameterize(mu, logvar)
        
        # Check for NaN in latent
        if torch.isnan(z).any():
            logger.warning("NaN detected in latent z, replacing with zeros")
            z = torch.nan_to_num(z, nan=0.0)
        
        # Decode with optional perturbation
        reconstructed = self.decoder(z)
        
        # Check and fix NaN in output
        if torch.isnan(reconstructed).any():
            logger.warning("NaN detected in reconstructed output, replacing with zeros")
            reconstructed = torch.nan_to_num(reconstructed, nan=0.0)
        
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def encode(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """Encode gene expression to latent parameters."""
        return self.encoder(gene_expression)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to gene expression."""
        return self.decoder(latent)
    
    # DEPRECEATED: This can be later tested, whether learning should be done based on experiment id (essentl. cell type), such that we can then specify H1 cells for MAYBE more accuracy, tbd.
    def DEPR_generate(self, experiment_ids: torch.Tensor,
                 target_gene_ids: Optional[torch.Tensor] = None,
                 n_samples: int = 1) -> torch.Tensor:
        """
        Generate synthetic cells by sampling from prior.
        
        Args:
            experiment_ids: Experiment IDs [batch_size]
            n_samples: Number of samples to generate per condition
        
        Returns:
            Generated gene expression [batch_size * n_samples, n_genes]
        """
        device = next(self.parameters()).device
        batch_size = len(experiment_ids)
        
        # Repeat conditions for multiple samples
        if n_samples > 1:
            experiment_ids = experiment_ids.repeat_interleave(n_samples)
        
        # Sample from standard normal prior
        z = torch.randn(batch_size * n_samples, self.config.latent_dim, device=device)
        
        # Decode to gene expression
        with torch.no_grad():
            generated = self.decode(z, experiment_ids, target_gene_ids)
        
        return generated
    
def vae_loss_func(outputs: Dict[str, torch.Tensor],
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

    # Normalize MSE loss by batch size
    reconstruction_loss = F.mse_loss(reconstructed, targets, reduction='sum') / targets.size(0)

    # KL divergence for standard Gaussian prior
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / targets.size(0)

    # Total loss with configurable weighting
    total_loss = (config.reconstruction_weight * reconstruction_loss +
                  config.kld_weight * kld_loss)

    return {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'kld_loss': kld_loss
    }

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Base VAE setup...")
    
    # Create config for testing
    config = VAEConfig(
        input_dim=1000,
        latent_dim=128,
        hidden_dims=[512, 256]
    )
    
    # Create model
    model = BaseVAE(config)
    
    # Test data
    batch_size = 32
    gene_expression = torch.randn(batch_size, config.input_dim)
    
    # Test Phase 1: Pretraining (no perturbation)
    print("\nTesting Phase 1 (Pretraining):")
    outputs_control = model(gene_expression)
    print(f"Control reconstruction shape: {outputs_control['reconstructed'].shape}")
    
    # Test loss computation
    loss_dict = vae_loss_func(outputs_control, gene_expression, config)
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    print(f"\nFlexibleVAE test completed successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
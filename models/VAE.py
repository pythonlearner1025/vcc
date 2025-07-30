#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) for single-cell RNA-seq data.

This module implements a conditional VAE that can predict perturbation effects
from single-cell gene expression data. The model is designed to work with the batch file
format produced by download.py.

Key Features:
- Encoder: Takes gene expression + experiment ID (no perturbation info)
- Decoder: Takes latent + experiment ID + optional perturbation ID
- Can train on general scRNA-seq data without perturbation labels (pretraining)
- Can inject perturbations at latent space to predict perturbed profiles
- Can predict control profiles from perturbed profiles
- Cell types emerge naturally in latent space without explicit conditioning

Training Paradigms:
1. Pretraining: Train on general scRNA-seq data without perturbation labels
2. Fine-tuning: Train on paired control/perturbation data
3. Inference: Inject perturbations into control cells or predict controls from perturbed cells
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class VAEConfig:
    """Configuration for the VAE model."""
    # Architecture
    input_dim: int = 1808  # Number of genes in VCC
    latent_dim: int = 2000  # Latent space dimension # HVG-like?
    hidden_dims: List[int] = None  # Hidden layer dimensions for encoder/decoder
    
    # Conditioning
    # n_cell_types: int = 50  # Number of unique cell types (estimate)
    n_perturbations: int = 100  # Number of unique perturbations (estimate)
    n_experiments: int = 500  # Number of unique SRX accessions (estimate)
    
    # cell_type_embed_dim: int = 32
    perturbation_embed_dim: int = 32
    experiment_embed_dim: int = 16
    
    # Loss weighting
    kld_weight: float = 1.0
    reconstruction_weight: float = 1.0
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]


class VAEEncoder(nn.Module):
    """Encoder network for VAE."""
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Input dimension includes gene expression + experiment embedding only
        # No perturbation information goes into the encoder
        conditioning_dim = config.experiment_embed_dim
        input_dim = config.input_dim + conditioning_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space projections
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, config.latent_dim)
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent parameters.
        
        Args:
            x: Gene expression data [batch_size, input_dim]
            conditioning: Experiment embedding only [batch_size, experiment_embed_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Concatenate gene expression with conditioning
        x_cond = torch.cat([x, conditioning], dim=1)
        
        # Encode
        h = self.encoder(x_cond)
        
        # Get latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder network for VAE."""
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Input dimension includes latent + experiment + perturbation embeddings
        # Perturbation is only added at decode time
        conditioning_dim = (config.perturbation_embed_dim + 
                          config.experiment_embed_dim)
        input_dim = config.latent_dim + conditioning_dim
        
        # Build decoder layers (reverse of encoder)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in reversed(config.hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, config.input_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor, 
                conditioning: torch.Tensor = None,
                experiment_emb: torch.Tensor = None,
                perturbation_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Decode latent representation to gene expression.
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            conditioning: Concatenated perturbation + experiment embeddings [batch_size, conditioning_dim]
                         OR None to use experiment_emb and perturbation_emb separately
            experiment_emb: Experiment embeddings [batch_size, experiment_embed_dim] (if conditioning is None)
            perturbation_emb: Perturbation embeddings [batch_size, perturbation_embed_dim] (if conditioning is None)
                             If None, uses zero perturbation (control condition)
            
        Returns:
            reconstructed: Reconstructed gene expression [batch_size, input_dim]
        """
        if conditioning is not None:
            # Use pre-concatenated conditioning
            z_cond = torch.cat([z, conditioning], dim=1)
        else:
            # Build conditioning from components
            if experiment_emb is None:
                raise ValueError("Either conditioning or experiment_emb must be provided")
            
            if perturbation_emb is None:
                # No perturbation - use zero embedding (control condition)
                batch_size = z.size(0)
                device = z.device
                perturbation_emb = torch.zeros(batch_size, self.config.perturbation_embed_dim, device=device)
            
            conditioning = torch.cat([perturbation_emb, experiment_emb], dim=1)
            z_cond = torch.cat([z, conditioning], dim=1)
        
        # Decode
        reconstructed = self.decoder(z_cond)
        
        return reconstructed


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for single-cell RNA-seq data.
    
    Architecture:
    - Encoder: Takes gene expression + experiment (SRX) embedding → latent space
    - Decoder: Takes latent + experiment + perturbation embeddings → reconstructed expression
    
    Training paradigm:
    - Control cells: encode → decode with control perturbation
    - Perturbed cells: encode → decode with target perturbation
    - The model learns to predict perturbation effects in the latent space
    
    Cell types emerge naturally in the latent space without explicit conditioning.
    """
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers for conditioning
        self.perturbation_embedding = nn.Embedding(config.n_perturbations, config.perturbation_embed_dim)
        self.experiment_embedding = nn.Embedding(config.n_experiments, config.experiment_embed_dim)
        
        # Encoder and decoder
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
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
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_encoder_conditioning(self, experiment_ids: torch.Tensor) -> torch.Tensor:
        """Get experiment embedding for encoder (no perturbation info)."""
        experiment_emb = self.experiment_embedding(experiment_ids)
        return experiment_emb
    
    def get_decoder_conditioning(self, 
                               perturbation_ids: torch.Tensor, 
                               experiment_ids: torch.Tensor) -> torch.Tensor:
        """Get concatenated perturbation + experiment embeddings for decoder."""
        perturbation_emb = self.perturbation_embedding(perturbation_ids)
        experiment_emb = self.experiment_embedding(experiment_ids)
        return torch.cat([perturbation_emb, experiment_emb], dim=1)
    
    def forward(self, x: torch.Tensor, 
                experiment_ids: torch.Tensor,
                perturbation_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Gene expression data [batch_size, input_dim]
            experiment_ids: Experiment IDs (SRX accessions) [batch_size]
            perturbation_ids: Perturbation type IDs [batch_size] (for decoder only)
                             If None, uses control condition (no perturbation)
            
        Returns:
            Dictionary containing:
            - reconstructed: Reconstructed gene expression
            - mu: Latent means
            - logvar: Latent log variances
            - z: Sampled latent representation
        """
        # Get conditioning embeddings
        encoder_conditioning = self.get_encoder_conditioning(experiment_ids)
        
        # Encode (without perturbation information)
        mu, logvar = self.encoder(x, encoder_conditioning)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode (with optional perturbation information)
        experiment_emb = self.experiment_embedding(experiment_ids)
        
        if perturbation_ids is not None:
            perturbation_emb = self.perturbation_embedding(perturbation_ids)
            reconstructed = self.decoder(z, experiment_emb=experiment_emb, perturbation_emb=perturbation_emb)
        else:
            # No perturbation provided - use control condition (zero perturbation)
            reconstructed = self.decoder(z, experiment_emb=experiment_emb, perturbation_emb=None)
        
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def encode(self, x: torch.Tensor, 
               experiment_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters (experiment conditioning only)."""
        encoder_conditioning = self.get_encoder_conditioning(experiment_ids)
        return self.encoder(x, encoder_conditioning)
    
    def decode(self, z: torch.Tensor,
               experiment_ids: torch.Tensor,
               perturbation_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode latent representation to gene expression (with optional perturbation conditioning)."""
        experiment_emb = self.experiment_embedding(experiment_ids)
        
        if perturbation_ids is not None:
            perturbation_emb = self.perturbation_embedding(perturbation_ids)
            return self.decoder(z, experiment_emb=experiment_emb, perturbation_emb=perturbation_emb)
        else:
            # No perturbation - control condition
            return self.decoder(z, experiment_emb=experiment_emb, perturbation_emb=None)
    
    def inject_perturbation(self, 
                           control_expression: torch.Tensor,
                           experiment_ids: torch.Tensor,
                           target_perturbation_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate perturbed profiles from control profiles by injecting perturbation at latent space.
        
        This is the key functionality for predicting perturbation effects:
        1. Encode control cell with only experiment conditioning (no perturbation)
        2. Decode the latent representation with target perturbation conditioning
        
        Args:
            control_expression: Control cell gene expression [batch_size, input_dim]
            experiment_ids: Experiment IDs (SRX accessions) [batch_size]
            target_perturbation_ids: Target perturbation IDs [batch_size]
            
        Returns:
            Predicted perturbed gene expression [batch_size, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # Encode control expression (no perturbation information)
            mu, logvar = self.encode(control_expression, experiment_ids)
            
            # Sample from latent distribution (or use mean for deterministic prediction)
            z = mu  # Use mean for deterministic prediction
            # z = self.reparameterize(mu, logvar)  # Use for stochastic prediction
            
            # Decode with target perturbation
            perturbed_expression = self.decode(z, experiment_ids, target_perturbation_ids)
        
        return perturbed_expression
    
    def predict_control_from_perturbed(self,
                                     perturbed_expression: torch.Tensor,
                                     experiment_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict control profiles from perturbed profiles by removing perturbation conditioning.
        
        Args:
            perturbed_expression: Perturbed cell gene expression [batch_size, input_dim]
            experiment_ids: Experiment IDs (SRX accessions) [batch_size]
            
        Returns:
            Predicted control gene expression [batch_size, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # Encode perturbed expression (no perturbation information in encoder)
            mu, logvar = self.encode(perturbed_expression, experiment_ids)
            
            # Sample from latent distribution
            z = mu  # Use mean for deterministic prediction
            
            # Decode without perturbation (control condition)
            control_expression = self.decode(z, experiment_ids, perturbation_ids=None)
        
        return control_expression
    
    def generate(self, experiment_ids: torch.Tensor,
                 perturbation_ids: Optional[torch.Tensor] = None,
                 n_samples: int = 1) -> torch.Tensor:
        """
        Generate new samples by sampling from prior and decoding.
        
        Args:
            experiment_ids: Experiment IDs (SRX accessions) [batch_size]
            perturbation_ids: Perturbation type IDs [batch_size] (optional)
                             If None, generates control samples
            n_samples: Number of samples to generate per condition
            
        Returns:
            Generated gene expression data [batch_size * n_samples, input_dim]
        """
        device = next(self.parameters()).device
        batch_size = len(experiment_ids)
        
        # Repeat condition IDs for multiple samples
        if n_samples > 1:
            experiment_ids = experiment_ids.repeat_interleave(n_samples)
            if perturbation_ids is not None:
                perturbation_ids = perturbation_ids.repeat_interleave(n_samples)
        
        # Sample from standard normal distribution
        z = torch.randn(batch_size * n_samples, self.config.latent_dim, device=device)
        
        # Decode
        with torch.no_grad():
            generated = self.decode(z, experiment_ids, perturbation_ids)
        
        return generated


def vae_loss_function(outputs: Dict[str, torch.Tensor], 
                     targets: torch.Tensor,
                     config: VAEConfig) -> Dict[str, torch.Tensor]:
    """
    Compute VAE loss (reconstruction + KL divergence).
    
    Args:
        outputs: Model outputs dictionary
        targets: Target gene expression [batch_size, input_dim]
        config: VAE configuration
        
    Returns:
        Dictionary containing loss components
    """
    reconstructed = outputs['reconstructed']
    mu = outputs['mu']
    logvar = outputs['logvar']
    
    # Reconstruction loss (MSE for continuous data)
    reconstruction_loss = F.mse_loss(reconstructed, targets, reduction='sum')
    
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = (config.reconstruction_weight * reconstruction_loss + 
                  config.kld_weight * kld_loss)
    
    return {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'kld_loss': kld_loss
    }


class VAETrainer:
    """Trainer class for the VAE model."""
    
    def __init__(self, model: ConditionalVAE, config: VAEConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        x = batch['x'].to(self.device)
        experiment_ids = batch['experiment_ids'].to(self.device)
        
        # Handle optional perturbation IDs
        perturbation_ids = None
        if 'perturbation_ids' in batch and batch['perturbation_ids'] is not None:
            perturbation_ids = batch['perturbation_ids'].to(self.device)
        
        # Forward pass
        outputs = self.model(x, experiment_ids, perturbation_ids)
        
        # Compute loss
        losses = vae_loss_function(outputs, x, self.config)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Return scalar losses
        return {k: v.item() for k, v in losses.items()}
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step."""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            x = batch['x'].to(self.device)
            experiment_ids = batch['experiment_ids'].to(self.device)
            
            # Handle optional perturbation IDs
            perturbation_ids = None
            if 'perturbation_ids' in batch and batch['perturbation_ids'] is not None:
                perturbation_ids = batch['perturbation_ids'].to(self.device)
            
            # Forward pass
            outputs = self.model(x, experiment_ids, perturbation_ids)
            
            # Compute loss
            losses = vae_loss_function(outputs, x, self.config)
        
        # Return scalar losses
        return {k: v.item() for k, v in losses.items()}
    
    # def predict_cell_type(self, x: torch.Tensor, 
    #                      perturbation_ids: torch.Tensor,
    #                      experiment_ids: torch.Tensor) -> torch.Tensor:
    #     """
    #     Predict cell type by trying all possible cell types and choosing the one
    #     that gives the best reconstruction.
    #     """
    #     self.model.eval()
    #     batch_size = x.size(0)
    #     device = x.device
        
    #     best_cell_types = []
        
    #     with torch.no_grad():
    #         for i in range(batch_size):
    #             x_single = x[i:i+1].expand(self.config.n_cell_types, -1)
    #             pert_single = perturbation_ids[i:i+1].expand(self.config.n_cell_types)
    #             exp_single = experiment_ids[i:i+1].expand(self.config.n_cell_types)
                
    #             # Try all possible cell types
    #             all_cell_types = torch.arange(self.config.n_cell_types, device=device)
                
    #             # Get reconstructions for all cell types
    #             outputs = self.model(x_single, all_cell_types, pert_single, exp_single)
                
    #             # Compute reconstruction errors
    #             recon_errors = F.mse_loss(outputs['reconstructed'], x_single, reduction='none').sum(dim=1)
                
    #             # Choose cell type with minimum reconstruction error
    #             best_cell_type = torch.argmin(recon_errors)
    #             best_cell_types.append(best_cell_type)
        
    #     return torch.stack(best_cell_types)
    
    def predict_perturbation_effect(self, x: torch.Tensor,
                                  perturbation_ids: torch.Tensor,
                                  experiment_ids: torch.Tensor,
                                  control_perturbation_id: int = 0) -> torch.Tensor:
        """
        Predict perturbation effect by comparing with control condition.
        
        Args:
            x: Perturbed gene expression
            perturbation_ids: Current perturbation IDs
            experiment_ids: Experiment IDs (SRX accessions)
            control_perturbation_id: ID for control/untreated condition
            
        Returns:
            Predicted gene expression under control condition
        """
        self.model.eval()
        batch_size = x.size(0)
        device = x.device
        
        # Create control condition
        control_perturbation_ids = torch.full((batch_size,), control_perturbation_id, device=device)
        
        with torch.no_grad():
            # Encode the perturbed state
            mu, logvar = self.model.encode(x, perturbation_ids, experiment_ids)
            
            # Sample from latent distribution
            z = self.model.reparameterize(mu, logvar)
            
            # Decode with control condition
            control_expression = self.model.decode(z, control_perturbation_ids, experiment_ids)
        
        return control_expression


# Utility functions for creating vocabulary mappings
def create_metadata_vocabularies(metadata_df) -> Dict[str, Dict]:
    """
    Create vocabulary mappings for categorical metadata.
    
    Args:
        metadata_df: DataFrame with categorical columns
        
    Returns:
        Dictionary with vocabulary mappings for each categorical column
    """
    vocabularies = {}
    
    categorical_columns = ['perturbation', 'srx_accession']
    
    for col in categorical_columns:
        if col in metadata_df.columns:
            unique_values = metadata_df[col].unique()
            vocab = {val: idx for idx, val in enumerate(unique_values)}
            vocab['<UNK>'] = len(vocab)  # Unknown token
            vocabularies[col] = vocab
    
    return vocabularies


def encode_metadata(metadata_df, vocabularies: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
    """
    Encode categorical metadata using vocabularies.
    
    Args:
        metadata_df: DataFrame with metadata
        vocabularies: Vocabulary mappings
        
    Returns:
        Dictionary with encoded metadata tensors
    """
    encoded = {}
    
    for col, vocab in vocabularies.items():
        if col in metadata_df.columns:
            # Map values to IDs, using <UNK> for unknown values
            ids = [vocab.get(val, vocab['<UNK>']) for val in metadata_df[col]]
            encoded[f'{col}_ids'] = torch.tensor(ids, dtype=torch.long)
    
    return encoded


if __name__ == "__main__":
    # Example usage and testing
    config = VAEConfig(
        input_dim=2000,
        latent_dim=128,
        hidden_dims=[512, 256],
        n_perturbations=100,
        n_experiments=500
    )
    
    # Create model
    model = ConditionalVAE(config)
    
    # Test forward pass with perturbation
    batch_size = 32
    x = torch.randn(batch_size, config.input_dim)
    perturbation_ids = torch.randint(0, config.n_perturbations, (batch_size,))
    experiment_ids = torch.randint(0, config.n_experiments, (batch_size,))
    
    outputs = model(x, experiment_ids, perturbation_ids)
    
    print(f"Model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {outputs['reconstructed'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass without perturbation (control condition)
    print("\nTesting control condition (no perturbation):")
    outputs_control = model(x, experiment_ids, perturbation_ids=None)
    print(f"Control reconstructed shape: {outputs_control['reconstructed'].shape}")
    
    # Test perturbation injection
    print("\nTesting perturbation injection:")
    control_x = torch.randn(16, config.input_dim)
    control_exp_ids = torch.randint(0, config.n_experiments, (16,))
    target_pert_ids = torch.randint(1, config.n_perturbations, (16,))
    
    perturbed_pred = model.inject_perturbation(control_x, control_exp_ids, target_pert_ids)
    print(f"Predicted perturbed shape: {perturbed_pred.shape}")
    
    # Test control prediction from perturbed
    print("\nTesting control prediction from perturbed:")
    control_pred = model.predict_control_from_perturbed(perturbed_pred, control_exp_ids)
    print(f"Predicted control shape: {control_pred.shape}")
    
    print("\nKey functionality verified:")
    print("✓ Standard VAE forward pass with perturbation")
    print("✓ Control condition (no perturbation)")
    print("✓ Perturbation injection into control cells")
    print("✓ Control prediction from perturbed cells")

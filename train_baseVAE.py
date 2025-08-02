"""
PyTorch Lightning Training Script for BaseVAE on Single-Cell RNA-seq Data.

This script implements a comprehensive, production-ready training pipeline for the BaseVAE
model using PyTorch Lightning. It includes:

- Modular configuration management
- Comprehensive logging with TensorBoard and optional WandB
- Model checkpointing and early stopping
- Learning rate scheduling
- Extensive validation and visualization routines
- UMI count preservation verification
- Gene-wise reconstruction analysis

Usage:
    python train_baseVAE.py --config config/training_config.json

Authors: AG
Created: July 2025
"""

import os
import sys
import json
import argparse
import logging
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, 
    TQDMProgressBar, ModelSummary
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Memory management
torch.backends.cudnn.benchmark = False  # Disable for deterministic behavior
torch.backends.cudnn.deterministic = True

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.VAE import BaseVAE, VAEConfig, vae_loss_func
from src.dataloaders.vae_loader import create_vae_dataloaders, DatasetConfig, MemoryMonitor


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Comprehensive training configuration."""
    
    # === Experiment Settings ===
    experiment_name: str = "baseVAE_pretraining"
    seed: int = 42
    
    # === Training Parameters ===
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: int = 32  # or 16 for mixed precision
    
    # === Learning Rate & Scheduling ===
    learning_rate: float = 1e-3
    lr_scheduler: str = "reduce_on_plateau"  # "cosine", "reduce_on_plateau", "step"
    lr_patience: int = 10
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    
    # === Early Stopping ===
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    early_stopping_mode: str = "min"
    
    # === Checkpointing ===
    save_top_k: int = 3
    checkpoint_monitor: str = "val_total_loss"
    checkpoint_mode: str = "min"
    
    # === Logging ===
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0  # Check validation every epoch
    use_wandb: bool = True  # Enable WandB logging
    wandb_project: str = "single-cell-vae"
    wandb_entity: str = 'ag'
    
    # === Validation & Visualization ===
    num_reconstruction_examples: int = 8
    num_latent_samples: int = 1000
    validate_umi_counts: bool = True
    generate_plots: bool = True
    plot_frequency: int = 5  # Generate plots every N epochs
    
    # === Hardware ===
    accelerator: str = "auto"  # "auto", "cpu", "gpu", "mps"
    devices: str = "auto"
    num_nodes: int = 1


class BaseVAELightningModule(pl.LightningModule):
    """PyTorch Lightning module for BaseVAE training with memory optimization."""
    
    def __init__(self, vae_config: VAEConfig, training_config: TrainingConfig):
        super().__init__()
        
        self.vae_config = vae_config
        self.training_config = training_config
        
        # Save hyperparameters
        self.save_hyperparameters({
            "vae_config": asdict(vae_config),
            "training_config": asdict(training_config)
        })
        
        # Initialize model
        self.model = BaseVAE(vae_config)
        
        # Metrics tracking
        self.validation_step_outputs = []
        
        # Memory monitoring
        self.log_memory_usage("Model initialized")
        
        logger.info(f"Initialized BaseVAE Lightning Module")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def log_memory_usage(self, stage: str):
        """Log current memory usage."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[{stage}] GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_memory_cached:.2f}GB cached")
        
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024**3
        logger.info(f"[{stage}] RAM Usage: {ram_usage:.2f}GB")
    
    def forward(self, batch):
        """Forward pass."""
        gene_expressions, metadata_list = batch
        return self.model(gene_expressions)
    
    # @DEPRECEATED
    def DEPR_on_train_start(self):
        """Log additional configuration when training starts."""
        try:
            # Log configuration details to WandB if available
            for logger_instance in self.loggers:
                if hasattr(logger_instance, 'experiment') and hasattr(logger_instance.experiment, 'log'):
                    # Log as text summary only (avoid metric logging issues)
                    if hasattr(logger_instance.experiment, 'log'):
                        logger_instance.experiment.log({
                            "config_summary": f"""
                            VAE Configuration:
                            - Input Dim: {self.vae_config.input_dim}
                            - Latent Dim: {self.vae_config.latent_dim}  
                            - Hidden Dims: {self.vae_config.hidden_dims}
                            - Learning Rate: {self.training_config.learning_rate}
                            - Max Epochs: {self.training_config.max_epochs}
                            - Model Params: {sum(p.numel() for p in self.model.parameters()):,}
                            """
                        })
        except Exception as e:
            logger.warning(f"Failed to log configuration details: {e}")
            # Don't attempt fallback logging since it also has issues

    def training_step(self, batch, batch_idx):
        """Training step."""
        gene_expressions, metadata_list = batch
        
        # Forward pass
        outputs = self.model(gene_expressions)
        
        # Compute losses
        loss_dict = vae_loss_func(outputs, gene_expressions, self.vae_config)
        
        # Log training metrics
        self.log("train_total_loss", loss_dict["total_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_reconstruction_loss", loss_dict["reconstruction_loss"], on_step=True, on_epoch=True)
        self.log("train_kld_loss", loss_dict["kld_loss"], on_step=True, on_epoch=True)
        
        # Log learning rate
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        return loss_dict["total_loss"]
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        gene_expressions, metadata_list = batch
        
        # Forward pass
        outputs = self.model(gene_expressions)
        
        # Compute losses
        loss_dict = vae_loss_func(outputs, gene_expressions, self.vae_config)
        
        # Compute additional metrics
        metrics = self._compute_validation_metrics(outputs, gene_expressions, metadata_list)
        
        # Combine losses and metrics
        step_output = {**loss_dict, **metrics, "gene_expressions": gene_expressions, "outputs": outputs}
        self.validation_step_outputs.append(step_output)
        
        # Log validation metrics
        self.log("val_total_loss", loss_dict["total_loss"], on_epoch=True, prog_bar=True)
        self.log("val_reconstruction_loss", loss_dict["reconstruction_loss"], on_epoch=True)
        self.log("val_kld_loss", loss_dict["kld_loss"], on_epoch=True)
        self.log("val_mse", metrics["mse"], on_epoch=True)
        self.log("val_mae", metrics["mae"], on_epoch=True)
        self.log("val_pearson_r", metrics["pearson_r"], on_epoch=True)
        
        return step_output
    
    def on_validation_epoch_end(self):
        """Process validation epoch results."""
        if not self.validation_step_outputs:
            return
        
        # Aggregate metrics
        avg_metrics = self._aggregate_validation_metrics(self.validation_step_outputs)
        
        # Log aggregated metrics
        for key, value in avg_metrics.items():
            if key.startswith("val_"):
                self.log(key, value, on_epoch=True)
        
        # Generate visualizations periodically
        if (self.training_config.generate_plots and 
            self.current_epoch % self.training_config.plot_frequency == 0):
            self._generate_validation_plots(self.validation_step_outputs)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        gene_expressions, metadata_list = batch
        
        # Forward pass
        outputs = self.model(gene_expressions)
        
        # Compute losses and metrics
        loss_dict = vae_loss_func(outputs, gene_expressions, self.vae_config)
        metrics = self._compute_validation_metrics(outputs, gene_expressions, metadata_list)
        
        # Log test metrics
        self.log("test_total_loss", loss_dict["total_loss"], on_epoch=True)
        self.log("test_reconstruction_loss", loss_dict["reconstruction_loss"], on_epoch=True)
        self.log("test_kld_loss", loss_dict["kld_loss"], on_epoch=True)
        self.log("test_mse", metrics["mse"], on_epoch=True)
        self.log("test_mae", metrics["mae"], on_epoch=True)
        self.log("test_pearson_r", metrics["pearson_r"], on_epoch=True)
        
        return {**loss_dict, **metrics}
    
    def _compute_validation_metrics(self, outputs, gene_expressions, metadata_list):
        """Compute comprehensive validation metrics."""
        reconstructed = outputs["reconstructed"].detach().cpu().numpy()
        original = gene_expressions.detach().cpu().numpy()
        
        # Basic reconstruction metrics
        mse = mean_squared_error(original.flatten(), reconstructed.flatten())
        mae = mean_absolute_error(original.flatten(), reconstructed.flatten())
        
        # Correlation metrics
        pearson_r, _ = pearsonr(original.flatten(), reconstructed.flatten())
        spearman_r, _ = spearmanr(original.flatten(), reconstructed.flatten())
        
        # UMI count preservation (if metadata available)
        umi_preservation = None
        if self.training_config.validate_umi_counts and metadata_list:
            umi_preservation = self._validate_umi_preservation(
                original, reconstructed, metadata_list
            )
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
        }
        
        if umi_preservation is not None:
            metrics["umi_preservation_error"] = umi_preservation
        
        return metrics
    
    def _validate_umi_preservation(self, original, reconstructed, metadata_list):
        """Validate UMI count preservation."""
        try:
            original_umi = np.array([meta.get("umi_count", 0) for meta in metadata_list])
            reconstructed_umi = np.sum(reconstructed, axis=1)
            
            # Compute relative error
            relative_error = np.mean(np.abs(original_umi - reconstructed_umi) / (original_umi + 1e-8))
            return relative_error
        except Exception as e:
            logger.warning(f"UMI validation failed: {e}")
            return None
    
    def _aggregate_validation_metrics(self, outputs):
        """Aggregate validation metrics across batches."""
        aggregated = {}
        
        # Simple averaging for most metrics
        for key in ["mse", "mae", "pearson_r", "spearman_r"]:
            values = [out[key] for out in outputs if key in out]
            if values:
                aggregated[f"val_avg_{key}"] = np.mean(values)
        
        # UMI preservation if available. In a weird way, this is difference of intial count vs reconstructed count? Ignore for now
        umi_errors = [out.get("umi_preservation_error") for out in outputs 
                      if out.get("umi_preservation_error") is not None]
        if umi_errors:
            aggregated["val_avg_umi_preservation_error"] = np.mean(umi_errors)
        
        return aggregated
    
    def _generate_validation_plots(self, outputs):
        """Generate comprehensive validation plots."""
        if not self.training_config.generate_plots:
            return
        
        try:
            # Sample data for plotting
            sample_output = outputs[0]
            original = sample_output["gene_expressions"][:self.training_config.num_reconstruction_examples]
            reconstructed = sample_output["outputs"]["reconstructed"][:self.training_config.num_reconstruction_examples]
            latent = sample_output["outputs"]["z"][:self.training_config.num_latent_samples]
            
            # Convert to numpy
            original_np = original.detach().cpu().numpy()
            reconstructed_np = reconstructed.detach().cpu().numpy()
            latent_np = latent.detach().cpu().numpy()
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"Validation Results - Epoch {self.current_epoch}", fontsize=16)
            
            # 1. Reconstruction scatter plot
            axes[0, 0].scatter(original_np.flatten(), reconstructed_np.flatten(), alpha=0.5, s=1)
            axes[0, 0].plot([original_np.min(), original_np.max()], 
                           [original_np.min(), original_np.max()], 'r--', alpha=0.8)
            axes[0, 0].set_xlabel("Original Expression")
            axes[0, 0].set_ylabel("Reconstructed Expression")
            axes[0, 0].set_title("Gene Expression Reconstruction")
            
            # 2. Per-sample correlation
            correlations = [pearsonr(orig, recon)[0] 
                           for orig, recon in zip(original_np, reconstructed_np)]
            axes[0, 1].hist(correlations, bins=20, alpha=0.7)
            axes[0, 1].axvline(np.mean(correlations), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(correlations):.3f}')
            axes[0, 1].set_xlabel("Pearson Correlation")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Per-Sample Reconstruction Correlation")
            axes[0, 1].legend()
            
            # 3. Latent space distribution (first 2 dimensions)
            if latent_np.shape[1] >= 2:
                axes[0, 2].scatter(latent_np[:, 0], latent_np[:, 1], alpha=0.6, s=2)
                axes[0, 2].set_xlabel("Latent Dim 0")
                axes[0, 2].set_ylabel("Latent Dim 1")
                axes[0, 2].set_title("Latent Space (2D Projection)")
            
            # 4. Gene-wise reconstruction error
            gene_errors = np.mean((original_np - reconstructed_np) ** 2, axis=0)
            axes[1, 0].hist(gene_errors, bins=50, alpha=0.7)
            axes[1, 0].set_xlabel("Mean Squared Error")
            axes[1, 0].set_ylabel("Number of Genes")
            axes[1, 0].set_title("Gene-wise Reconstruction Error")
            axes[1, 0].set_yscale('log')
            
            # 5. Expression magnitude vs error
            mean_expression = np.mean(original_np, axis=0)
            axes[1, 1].scatter(mean_expression, gene_errors, alpha=0.6, s=2)
            axes[1, 1].set_xlabel("Mean Gene Expression")
            axes[1, 1].set_ylabel("Reconstruction Error")
            axes[1, 1].set_title("Expression Level vs Error")
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_yscale('log')
            
            # 6. Latent space variance
            latent_vars = np.var(latent_np, axis=0)
            axes[1, 2].bar(range(len(latent_vars)), sorted(latent_vars, reverse=True))
            axes[1, 2].set_xlabel("Latent Dimension (sorted)")
            axes[1, 2].set_ylabel("Variance")
            axes[1, 2].set_title("Latent Dimension Utilization")
            axes[1, 2].set_yscale('log')
            
            plt.tight_layout()
            
            # Log to tensorboard
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure("validation_plots", fig, self.current_epoch)
            
            # Save figure
            plots_dir = Path(self.trainer.log_dir) / "plots"
            plots_dir.mkdir(exist_ok=True)
            fig.savefig(plots_dir / f"validation_epoch_{self.current_epoch:03d}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Failed to generate validation plots: {e}")
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate
        )
        
        scheduler_config = {}
        
        if self.training_config.lr_scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.training_config.early_stopping_mode,
                factor=self.training_config.lr_factor,
                patience=self.training_config.lr_patience,
                min_lr=self.training_config.lr_min
            )
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": self.training_config.checkpoint_monitor,
                "frequency": 1
            }
        
        elif self.training_config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_config.max_epochs,
                eta_min=self.training_config.lr_min
            )
            scheduler_config = {"scheduler": scheduler}
        
        elif self.training_config.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.training_config.lr_patience,
                gamma=self.training_config.lr_factor
            )
            scheduler_config = {"scheduler": scheduler}
        
        if scheduler_config:
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        else:
            return optimizer


class VAEDataModule(pl.LightningDataModule):
    """Optimized PyTorch Lightning DataModule for VAE training."""
    
    def __init__(self, dataset_config: DatasetConfig):
        super().__init__()
        self.dataset_config = dataset_config
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        MemoryMonitor.log_memory_usage("Before datamodule setup")
        
        # Create optimized dataloaders using our new function
        self.train_loader, self.val_loader, self.test_loader = create_vae_dataloaders(
            self.dataset_config
        )
        
        MemoryMonitor.log_memory_usage("After datamodule setup")
    
    def train_dataloader(self):
        if self.train_loader is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self.train_loader
    
    def val_dataloader(self):
        if self.val_loader is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self.val_loader
    
    def test_dataloader(self):
        if self.test_loader is None:
            logger.info("No test dataloader available - this is normal if test_split=0")
            return None
        return self.test_loader
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up resources."""
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        gc.collect()
        MemoryMonitor.log_memory_usage("After datamodule teardown")


def setup_callbacks(training_config: TrainingConfig, log_dir: str) -> List[pl.Callback]:
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename=f"{training_config.experiment_name}-{{epoch:02d}}-{{val_total_loss:.4f}}",
        monitor=training_config.checkpoint_monitor,
        mode=training_config.checkpoint_mode,
        save_top_k=training_config.save_top_k,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=training_config.checkpoint_monitor,
        patience=training_config.early_stopping_patience,
        mode=training_config.early_stopping_mode,
        min_delta=training_config.early_stopping_min_delta,
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # TQDM progress bar (more stable than Rich)
    progress_bar = TQDMProgressBar(refresh_rate=20)
    callbacks.append(progress_bar)
    
    # Model summary
    model_summary = ModelSummary(max_depth=2)
    callbacks.append(model_summary)
    
    return callbacks

def setup_logger(training_config: TrainingConfig, log_dir: str, 
                vae_config: VAEConfig = None, dataset_config: DatasetConfig = None) -> List[pl.loggers.Logger]:
    """Setup experiment logger with robust configuration handling."""
    # TensorBoard logger (always available)
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=training_config.experiment_name,
        version=None  # Auto-increment
    )
    
    loggers = [tb_logger]
    
    # WandB logger (optional)
    if training_config.use_wandb:
        try:
            # Initialize WandB with error handling
            wandb_logger = WandbLogger(
                project=training_config.wandb_project,
                entity=training_config.wandb_entity,
                name=training_config.experiment_name,
                save_dir=log_dir
            )
            
            # Verify WandB logger is working
            if hasattr(wandb_logger, 'experiment') and wandb_logger.experiment is not None:
                loggers.append(wandb_logger)
                logger.info("WandB logging enabled successfully")
                
                # Log additional configuration details that might not fit in config
                try:
                    # Log complex configurations as text
                    if hasattr(wandb_logger.experiment, 'log'):
                        wandb_logger.experiment.log({
                            "vae_hidden_dims_detail": str(getattr(vae_config, 'hidden_dims', [])),
                            "full_vae_config": str(asdict(vae_config)),
                            "full_dataset_config": str(asdict(dataset_config)),
                            "full_training_config": str(asdict(training_config))
                        })
                except Exception as log_error:
                    logger.warning(f"Failed to log additional config details: {log_error}")
            else:
                logger.warning("WandB logger initialization appeared to succeed but experiment is None")
                
        except Exception as e:
            logger.warning(f"Failed to initialize WandB logger: {e}")
            logger.warning("Continuing with TensorBoard logging only")
            
            # Try to provide more detailed error information
            if "permission" in str(e).lower():
                logger.warning("WandB permission error - check project access and entity name")
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                logger.warning("WandB network error - check internet connection")
            else:
                logger.warning(f"WandB error details: {type(e).__name__}: {e}")
    
    return loggers


def load_config(config_path: str) -> Tuple[VAEConfig, DatasetConfig, TrainingConfig]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Extract configurations
    vae_config = VAEConfig(**config_dict.get("vae_config", {}))
    dataset_config = DatasetConfig(**config_dict.get("dataset_config", {}))
    training_config = TrainingConfig(**config_dict.get("training_config", {}))
    
    return vae_config, dataset_config, training_config


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration dictionary."""
    return {
        "vae_config": asdict(VAEConfig()),
        "dataset_config": asdict(DatasetConfig()),
        "training_config": asdict(TrainingConfig())
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train BaseVAE on single-cell data")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration JSON file")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory for logging and checkpoints")
    parser.add_argument("--create_default_config", action="store_true",
                       help="Create a default configuration file and exit")
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_default_config:
        default_config = create_default_config()
        config_path = "config/default_training_config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default configuration at: {config_path}")
        return
    
    # Load configuration
    if args.config:
        vae_config, dataset_config, training_config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        # Use default configurations
        vae_config = VAEConfig()
        dataset_config = DatasetConfig()
        training_config = TrainingConfig()
        logger.info("Using default configurations")
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seeds for reproducibility
    pl.seed_everything(training_config.seed, workers=True)
    
    # Initialize data module
    data_module = VAEDataModule(dataset_config)
    
    # Initialize model
    model = BaseVAELightningModule(vae_config, training_config)
    
    # Setup callbacks and loggers
    callbacks = setup_callbacks(training_config, str(log_dir))
    loggers = setup_logger(training_config, str(log_dir), vae_config, dataset_config)
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=training_config.max_epochs,
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        num_nodes=training_config.num_nodes,
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        val_check_interval=training_config.val_check_interval,
        log_every_n_steps=training_config.log_every_n_steps,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True
    )
    
    # Log configuration
    logger.info(f"VAE Configuration: {vae_config}")
    logger.info(f"Dataset Configuration: {dataset_config}")
    logger.info(f"Training Configuration: {training_config}")
    logger.info(f"Log directory: {log_dir}")
    
    # Train the model
    try:
        trainer.fit(model, data_module)
        
        # Test the model
        # trainer.test(model, data_module)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
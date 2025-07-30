"""
Dataset module for single-cell RNA-seq data.
"""

from .dataset import ScRNADataset

# Optional imports that might have dependencies
try:
    from .download import ScRNADownloader
except ImportError:
    ScRNADownloader = None

try:
    from .vcc_paired_dataloader import (
        VCCPairedDataset,
        create_vcc_paired_dataloader,
        create_train_val_dataloaders,
    )
except ImportError:
    VCCPairedDataset = None
    create_vcc_paired_dataloader = None
    create_train_val_dataloaders = None

try:
    from .scrna_hvg_dataset import (
        ScRNADatasetWithHVGs,
        create_scrna_hvg_dataloader
    )
except ImportError:
    ScRNADatasetWithHVGs = None
    create_scrna_hvg_dataloader = None

# VAE-specific imports
try:
    from .vae_paired_dataloader import VAEPairedDataset, create_vae_dataloaders, load_vae_data
except ImportError:
    VAEPairedDataset = None
    create_vae_dataloaders = None
    load_vae_data = None

__all__ = [
    'ScRNADataset',
    'ScRNADownloader',
    'VCCPairedDataset',
    'create_vcc_paired_dataloader',
    'create_train_val_dataloaders',
    'ScRNADatasetWithHVGs',
    'create_scrna_hvg_dataloader',
    'VAEPairedDataset',
    'create_vae_dataloaders',
    'load_vae_data'
]
    'VCCPairedDataset',
    'create_vcc_paired_dataloader',
    'create_train_val_dataloaders',
    'ScRNADatasetWithHVGs',
    'create_scrna_hvg_dataloader'
]

#!/usr/bin/env python3
"""
Cleanup script for VCC repository refactoring.

This script helps transition from legacy VAE implementation to the new
Flexible VAE architecture by:
1. Moving legacy files to a legacy/ directory
2. Creating compatibility aliases
3. Updating import paths
4. Generating migration guide

Usage:
    python cleanup_legacy.py --backup    # Move files to legacy/ (safe)
    python cleanup_legacy.py --remove    # Remove legacy files (destructive)
    python cleanup_legacy.py --analyze   # Just analyze what would be changed
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

# Legacy files to be handled
LEGACY_FILES = {
    'training_scripts': [
        'train_vae.py',
        'train_vae_clean.py'
    ],
    'example_scripts': [
        'example_vae.py', 
        'example_vae_clean.py',
        'example_perturbation_injection.py'
    ],
    'model_files': [
        'models/VAE.py'
    ],
    'inference_scripts': [
        'inference_vae.py'  # If exists
    ]
}

# New files that replace legacy ones
REPLACEMENTS = {
    'train_vae.py': 'train_flexible_vae.py',
    'train_vae_clean.py': 'train_flexible_vae.py',
    'example_vae.py': 'example_flexible_vae.py',
    'example_vae_clean.py': 'example_flexible_vae.py', 
    'example_perturbation_injection.py': 'example_flexible_vae.py',
    'models/VAE.py': 'models/flexible_vae.py',
    'inference_vae.py': 'inference_flexible_vae.py',
    'dataset/vae_paired_dataloader.py': 'dataset/flexible_dataloader.py'
}


def analyze_legacy_files(project_root: Path) -> Dict[str, List[Path]]:
    """Analyze which legacy files exist in the project."""
    found_files = {}
    
    for category, files in LEGACY_FILES.items():
        found_files[category] = []
        for file_path in files:
            full_path = project_root / file_path
            if full_path.exists():
                found_files[category].append(full_path)
    
    return found_files


def create_migration_guide(project_root: Path, found_files: Dict[str, List[Path]]):
    """Create a migration guide for users."""
    guide_path = project_root / 'MIGRATION_GUIDE.md'
    
    content = """# Migration Guide: Legacy VAE → Flexible VAE

This guide helps migrate from the legacy VAE implementation to the new Flexible VAE architecture.

## What Changed

The VCC repository has been refactored to use a more modular, extensible VAE architecture:

- **New Architecture**: `models/flexible_vae.py` replaces `models/VAE.py`
- **Unified Training**: `train_flexible_vae.py` replaces multiple training scripts
- **Better Examples**: `example_flexible_vae.py` provides comprehensive examples
- **Enhanced Data Loading**: `dataset/flexible_dataloader.py` with robust phase-aware loading
- **Inference Tools**: `inference_flexible_vae.py` for model deployment

## Quick Migration

### Training Scripts

**Old**:
```bash
python train_vae.py --data_path data.npz --latent_dim 128
python train_vae_clean.py --data_path data.npz --latent_dim 128
```

**New**:
```bash
# Phase 1: Pretraining
python train_flexible_vae.py --phase 1 --data_path data.npz --latent_dim 128

# Phase 2: Fine-tuning  
python train_flexible_vae.py --phase 2 --data_path paired_data.npz --latent_dim 128 --pretrained_model phase1_model.pt
```

### Example Scripts

**Old**:
```bash
python example_vae.py
python example_perturbation_injection.py
```

**New**:
```bash
python example_flexible_vae.py --mode synthetic
```

### Model Import

**Old**:
```python
from models.VAE import ConditionalVAE, VAEConfig
```

**New**:
```python
from models.flexible_vae import FlexibleVAE, VAEConfig
```

### Data Loading

**Old**:
```python
from dataset.vae_paired_dataloader import VAEPairedDataset
```

**New**:
```python
from dataset.flexible_dataloader import PairedPerturbationDataset
# or
from dataset.flexible_dataloader import create_flexible_dataloaders
```

## Key Improvements

### 1. Modular Gene Embeddings

**New**: Easy to swap between different gene embedding types:

```python
# Learned embeddings
gene_emb = LearnedGeneEmbedding(n_genes=1000, embed_dim=128)

# Pretrained embeddings (ESM2, Gene2Vec, etc.)
embeddings = torch.load('esm2_embeddings.pt')
gene_emb = PretrainedGeneEmbedding(embeddings, freeze=True)

model = FlexibleVAE(config, gene_emb)
```

### 2. Phase-Aware Training

**New**: Clear separation of pretraining and fine-tuning phases:

```python
# Phase 1: No perturbation labels needed
outputs = model(expression, experiment_ids)

# Phase 2: With target gene information
outputs = model(expression, experiment_ids, target_gene_ids=targets)
```

### 3. Configuration Management

**New**: YAML-based configuration files:

```yaml
# configs/phase1_config.yaml
latent_dim: 512
learning_rate: 0.001
batch_size: 256
kld_weight: 0.5
```

### 4. Comprehensive Inference

**New**: Dedicated inference script with multiple modes:

```bash
# Perturbation injection
python inference_flexible_vae.py inject --model model.pt --data cells.npz --perturbation_id 5

# Latent space analysis  
python inference_flexible_vae.py analyze --model model.pt --data cells.npz

# Batch prediction
python inference_flexible_vae.py predict --model model.pt --data cells.npz
```

## File Mapping

"""
    
    # Add file mappings
    for category, files in found_files.items():
        if files:
            content += f"\n### {category.replace('_', ' ').title()}\n\n"
            for file_path in files:
                rel_path = file_path.relative_to(project_root)
                replacement = REPLACEMENTS.get(str(rel_path), "No direct replacement")
                content += f"- `{rel_path}` → `{replacement}`\n"
    
    content += """
## Breaking Changes

1. **Model Class Name**: `ConditionalVAE` → `FlexibleVAE`
2. **Configuration Structure**: Some config fields renamed for clarity
3. **Data Loader Interface**: New phase-aware data loading
4. **Gene Embedding System**: Abstract base class for different embedding types

## Compatibility

The legacy files have been moved to `legacy/` directory and can still be used:

```python
# If you need legacy functionality temporarily
sys.path.append('legacy')
from VAE import ConditionalVAE  # Legacy import
```

However, we recommend migrating to the new architecture for:
- Better modularity and extensibility
- Improved documentation and examples  
- Enhanced performance and robustness
- Future feature compatibility

## Support

If you encounter issues during migration:

1. Check this guide for common patterns
2. Look at `example_flexible_vae.py` for comprehensive examples
3. Review configuration files in `configs/`
4. Open an issue on GitHub for specific problems

## Timeline

- **Current**: Both legacy and new systems available
- **Future**: Legacy system will be deprecated in next major version
- **Recommendation**: Migrate to Flexible VAE for all new work
"""
    
    with open(guide_path, 'w') as f:
        f.write(content)
    
    print(f"Migration guide created: {guide_path}")


def backup_legacy_files(project_root: Path, found_files: Dict[str, List[Path]]):
    """Move legacy files to legacy/ directory."""
    legacy_dir = project_root / 'legacy'
    legacy_dir.mkdir(exist_ok=True)
    
    moved_files = []
    
    for category, files in found_files.items():
        for file_path in files:
            # Create category subdirectory in legacy/
            category_dir = legacy_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Move file
            destination = category_dir / file_path.name
            print(f"Moving {file_path} → {destination}")
            
            try:
                shutil.move(str(file_path), str(destination))
                moved_files.append((file_path, destination))
            except Exception as e:
                print(f"Error moving {file_path}: {e}")
    
    # Create legacy __init__.py files for imports
    for category in found_files.keys():
        category_dir = legacy_dir / category
        if category_dir.exists():
            init_file = category_dir / '__init__.py'
            init_file.write_text(f"# Legacy {category} - deprecated\n")
    
    return moved_files


def remove_legacy_files(project_root: Path, found_files: Dict[str, List[Path]]):
    """Remove legacy files (destructive operation)."""
    removed_files = []
    
    print("WARNING: This will permanently delete legacy files!")
    confirmation = input("Type 'DELETE' to confirm: ")
    
    if confirmation != 'DELETE':
        print("Operation cancelled.")
        return removed_files
    
    for category, files in found_files.items():
        for file_path in files:
            print(f"Removing {file_path}")
            try:
                file_path.unlink()
                removed_files.append(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    return removed_files


def create_compatibility_aliases(project_root: Path):
    """Create compatibility aliases for common imports."""
    
    # Create compatibility file for model imports
    compat_file = project_root / 'legacy_compat.py'
    
    content = '''"""
Compatibility aliases for legacy VAE imports.

This file provides backward compatibility for existing code that imports
from the legacy VAE modules. Use these imports to ease migration:

    from legacy_compat import ConditionalVAE, VAEConfig  # Legacy names
    
However, we recommend updating to the new imports:

    from models.flexible_vae import FlexibleVAE, VAEConfig  # New names
"""

import warnings
from models.flexible_vae import FlexibleVAE, VAEConfig as FlexibleVAEConfig

# Legacy aliases
ConditionalVAE = FlexibleVAE
VAEConfig = FlexibleVAEConfig

def _deprecated_warning(old_name, new_name):
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Monkey patch to add deprecation warnings
original_init = FlexibleVAE.__init__

def deprecated_init(self, *args, **kwargs):
    _deprecated_warning("ConditionalVAE", "FlexibleVAE")
    return original_init(self, *args, **kwargs)

ConditionalVAE.__init__ = deprecated_init

# Export legacy interface
__all__ = ['ConditionalVAE', 'VAEConfig']
'''
    
    with open(compat_file, 'w') as f:
        f.write(content)
    
    print(f"Compatibility aliases created: {compat_file}")


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description='Cleanup legacy VCC files')
    parser.add_argument('--analyze', action='store_true', 
                       help='Analyze what files would be affected')
    parser.add_argument('--backup', action='store_true',
                       help='Move legacy files to legacy/ directory (safe)')
    parser.add_argument('--remove', action='store_true',
                       help='Remove legacy files permanently (destructive)')
    parser.add_argument('--project_root', type=str, default='.',
                       help='Path to project root directory')
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.backup, args.remove]):
        parser.print_help()
        return
    
    project_root = Path(args.project_root).resolve()
    print(f"Working in project root: {project_root}")
    
    # Analyze existing files
    found_files = analyze_legacy_files(project_root)
    
    print("\nLegacy files found:")
    total_files = 0
    for category, files in found_files.items():
        print(f"  {category}: {len(files)} files")
        for file_path in files:
            print(f"    - {file_path.relative_to(project_root)}")
        total_files += len(files)
    
    if total_files == 0:
        print("No legacy files found. Repository already clean!")
        return
    
    # Create migration guide
    create_migration_guide(project_root, found_files)
    
    if args.analyze:
        print(f"\nAnalysis complete. {total_files} legacy files found.")
        print("Use --backup to move files to legacy/ directory")
        print("Use --remove to delete files permanently")
        return
    
    if args.backup:
        print(f"\nBacking up {total_files} legacy files...")
        moved_files = backup_legacy_files(project_root, found_files)
        create_compatibility_aliases(project_root)
        print(f"✓ Moved {len(moved_files)} files to legacy/ directory")
        print("✓ Created compatibility aliases")
        print("✓ Created migration guide")
        
    elif args.remove:
        print(f"\nRemoving {total_files} legacy files...")
        removed_files = remove_legacy_files(project_root, found_files)
        print(f"✓ Removed {len(removed_files)} files")
        print("✓ Created migration guide")
    
    print("\nCleanup completed!")
    print("Next steps:")
    print("1. Review MIGRATION_GUIDE.md")
    print("2. Update any custom scripts to use new APIs")
    print("3. Test with example_flexible_vae.py")


if __name__ == '__main__':
    main()

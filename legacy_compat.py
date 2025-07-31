"""
Compatibility aliases for legacy VAE imports.

This file provides backward compatibility for existing code that imports
from the legacy VAE modules. Use these imports to ease migration:

    from legacy_compat import ConditionalVAE, VAEConfig  # Legacy names
    
However, we recommend updating to the new imports:

    from models.flexible_vae import FlexibleVAE, VAEConfig  # New names
"""

import warnings
from models.flexible_vae import FlexibleVAE, VAEConfig as FlexibleVAEConfig
import torch.nn.functional as F

# Legacy aliases
ConditionalVAE = FlexibleVAE
VAE = FlexibleVAE  # Another common alias
VAEConfig = FlexibleVAEConfig

# Legacy loss functions
def reconstruction_loss(predicted, target, reduction='mean'):
    """Legacy reconstruction loss function."""
    return F.mse_loss(predicted, target, reduction=reduction)

def _deprecated_warning(old_name, new_name):
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Monkey patch to add deprecation warnings
original_init = FlexibleVAE.__init__

def deprecated_init(self, *args, **kwargs):
    # Check which class was used
    class_name = self.__class__.__name__
    if class_name == "FlexibleVAE":
        # Check the calling context to determine which alias was used
        import inspect
        frame = inspect.currentframe()
        try:
            caller_locals = frame.f_back.f_locals
            caller_globals = frame.f_back.f_globals
            
            # Try to detect which alias was used
            if 'ConditionalVAE' in str(frame.f_back.f_code):
                _deprecated_warning("ConditionalVAE", "FlexibleVAE")
            elif 'VAE' in str(frame.f_back.f_code) and 'FlexibleVAE' not in str(frame.f_back.f_code):
                _deprecated_warning("VAE", "FlexibleVAE")
        finally:
            del frame
    
    return original_init(self, *args, **kwargs)

ConditionalVAE.__init__ = deprecated_init
VAE.__init__ = deprecated_init

# Export legacy interface
__all__ = ['ConditionalVAE', 'VAE', 'VAEConfig', 'reconstruction_loss']

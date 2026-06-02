from .model import prithvi_wrapper
from .unet_with_transformer import UNetWithTransformer


# Dictionary to map architecture names to classes
models = {
    "prithvi": prithvi_wrapper,
    "unet_with_transformer": UNetWithTransformer,
}

# Expose the dictionary for external access
__all__ = ["models"]
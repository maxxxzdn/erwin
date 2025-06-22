"""
This module provides the following components:
- Erwin: Base model implementation
- ErwinFlash: Memory-efficient model with FlashAttention and optimized operations
"""

try:
    import flash_attn
    from .erwin_flash import ErwinTransformer
except ImportError:
    print("FlashAttention is not installed. Using the standard ErwinTransformer.")
    from .erwin import ErwinTransformer

__all__ = [
    "ErwinTransformer",
]
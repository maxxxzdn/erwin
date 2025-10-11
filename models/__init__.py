"""
This module provides the following components:
- Erwin: Base model implementation
- ErwinFlash: Memory-efficient model with FlashAttention and optimized operations
"""

try:
    import flash_attn
    flash_attn_available = True
except ImportError:
    print("FlashAttention is not installed. Using the standard ErwinTransformer.")
    flash_attn_available = False

if not flash_attn_available:
    from models.erwin import ErwinTransformer
else:
    from models.erwin_flash import ErwinTransformer

__all__ = [
    "ErwinTransformer",
]
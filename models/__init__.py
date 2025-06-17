"""
This module provides the following components:
- Erwin: Base model implementation
- ErwinFlash: Memory-efficient model with FlashAttention and optimzed operations
"""

from .erwin import ErwinTransformer
from .erwin_flash import ErwinTransformer as ErwinFlashTransformer

__all__ = [
    "ErwinTransformer",
    "ErwinFlashTransformer",
]
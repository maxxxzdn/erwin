from __future__ import annotations

# Standard library
import math
from dataclasses import dataclass
from typing import List, Literal

# Third-party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
import torch_scatter
from einops import rearrange, reduce
from flash_attn import flash_attn_qkvpacked_func

# Local modules
from balltree import build_balltree_with_rotations

# --- Utility Functions ---


def scatter_mean(src: torch.Tensor, idx: torch.Tensor, num_receivers: int):
    """
    Averages all values from src into the receivers at the indices specified by idx.
    Uses torch_scatter's optimized implementation.

    Args:
        src (torch.Tensor): Source tensor of shape (N, D).
        idx (torch.Tensor): Indices tensor of shape (N,).
        num_receivers (int): Number of receivers (usually the maximum index in idx + 1).

    Returns:
        torch.Tensor: Result tensor of shape (num_receivers, D).
    """
    return torch_scatter.scatter_mean(src, idx, dim=0, dim_size=num_receivers)


# --- Model Components ---


class BallMSA(nn.Module):
    """
    Ball Multi-Head Self-Attention (BMSA) module (eq. 8) using Flash Attention with packed QKV.

    This module performs multi-head self-attention within local regions (balls) of points.
    It utilizes Flash Attention for efficiency and supports relative positional encoding.
    """

    def __init__(
        self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 3
    ):
        """
        Initializes the BallMSA module.

        Args:
            dim (int): Input and output dimension of the features.
            num_heads (int): Number of attention heads.
            ball_size (int): Number of points within each ball.
            dimensionality (int): Dimensionality of the input coordinates (e.g., 3 for 3D points).
        """
        super().__init__()
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Linear layers for QKV projection, output projection, and positional encoding projection
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(
            dimensionality, dim
        )  # Projects relative positions to feature dimension

        # Learnable parameter for scaling distance-based attention bias
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """
        Creates a distance-based attention bias (eq. 10).

        This mask biases the attention mechanism to favor closer points within a ball.

        Args:
            pos (torch.Tensor): Positions of points within balls, shape (num_balls * ball_size, dimensionality).

        Returns:
            torch.Tensor: Attention bias tensor.
        """
        pos = rearrange(
            pos, "(n m) d -> n m d", m=self.ball_size
        )  # Reshape to (num_balls, ball_size, dimensionality)
        # Calculate pairwise distances and apply learnable scaling
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """
        Computes relative positions of leaf nodes with respect to the center of their ball (eq. 9).

        Args:
            pos (torch.Tensor): Positions of points within balls, shape (num_balls * ball_size, dimensionality).

        Returns:
            torch.Tensor: Relative positions, shape (num_balls * ball_size, dimensionality).
        """
        num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        pos = pos.view(
            num_balls, self.ball_size, dim
        )  # Reshape to (num_balls, ball_size, dimensionality)
        # Subtract ball center from each point's position
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, dtype: torch.dtype = torch.bfloat16
    ):
        """
        Forward pass of the BallMSA module.

        Args:
            x (torch.Tensor): Input features, shape (num_balls * ball_size, dim).
            pos (torch.Tensor): Positions of points, shape (num_balls * ball_size, dimensionality).

        Returns:
            torch.Tensor: Output features after attention, shape (num_balls * ball_size, dim).
        """
        # --- Weight and Bias Preparation (Casting to float16 for potential speedup) ---
        qkv_weight = self.qkv.weight.to(dtype)
        qkv_bias = self.qkv.bias.to(dtype) if self.qkv.bias is not None else None
        proj_weight = self.proj.weight.to(dtype)
        proj_bias = self.proj.bias.to(dtype) if self.proj.bias is not None else None
        pe_weight = self.pe_proj.weight.to(dtype)
        pe_bias = self.pe_proj.bias.to(dtype) if self.pe_proj.bias is not None else None

        # --- Positional Encoding ---
        # Compute relative positions and project them to feature dimension
        rel_pos = self.compute_rel_pos(pos)
        pe = F.linear(rel_pos, pe_weight, pe_bias)
        x = x + pe  # Add positional encoding to input features

        # --- QKV Projection ---
        qkv = F.linear(x, qkv_weight, qkv_bias)  # Project input features to Q, K, V

        # --- Reshape for Flash Attention ---
        B = x.shape[0] // self.ball_size  # Number of balls
        qkv = rearrange(
            qkv,
            "(b n) (three h d) -> b n three h d",  # (batch*ball_size, 3*heads*head_dim) -> (batch, ball_size, 3, heads, head_dim)
            b=B,
            n=self.ball_size,
            three=3,  # For Q, K, V
            h=self.num_heads,
            d=self.head_dim,
        )

        # --- Attention Calculation ---
        softmax_scale = 1.0 / math.sqrt(
            self.head_dim
        )  # Scaling factor for dot products

        # Apply Flash Attention (efficient implementation of self-attention)
        x = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,  # causal=False for non-autoregressive attention
        )  # Output shape: [B, ball_size, H, head_dim]

        # --- Output Projection ---
        x = rearrange(
            x, "b n h d -> (b n) (h d)"
        )  # Reshape back to (batch*ball_size, dim)
        x = F.linear(x, proj_weight, proj_bias)  # Project attended features

        return x


class ErwinEmbedding(nn.Module):
    """
    Embedding module for Erwin Transformer.

    It consists of a linear projection followed by an optional Message Passing Neural Network (MPNN).
    """

    def __init__(self, in_dim: int, dim: int, mp_steps: int, dimensionality: int = 3):
        """
        Initializes the ErwinEmbedding module.

        Args:
            in_dim (int): Dimension of the input features.
            dim (int): Dimension of the output features (embedding dimension).
            mp_steps (int): Number of message passing steps in the MPNN. If 0, MPNN is skipped.
            dimensionality (int): Dimensionality of the input coordinates.
        """
        super().__init__()
        self.mp_steps = mp_steps
        self.embed_fn = nn.Linear(in_dim, dim)  # Linear projection layer
        self.mpnn = MPNN(dim, mp_steps, dimensionality)  # MPNN layer

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass of the ErwinEmbedding module.

        Args:
            x (torch.Tensor): Input features.
            pos (torch.Tensor): Positions of points.
            edge_index (torch.Tensor): Edge indices for the graph (used by MPNN).

        Returns:
            torch.Tensor: Embedded features.
        """
        x = self.embed_fn(x)  # Apply linear projection
        # Apply MPNN if mp_steps > 0
        return self.mpnn(x, pos, edge_index) if self.mp_steps > 0 else x


class SwiGLU(nn.Module):
    """
    SwiGLU activation function module: W_3 SiLU(W_1 x) ⊗ W_2 x.

    This is a type of gated linear unit that often performs well in transformers.
    """

    def __init__(self, in_dim: int, dim: int):
        """
        Initializes the SwiGLU module.

        Args:
            in_dim (int): Input dimension.
            dim (int): Hidden dimension of the internal linear layers.
        """
        super().__init__()
        self.w1 = nn.Linear(in_dim, dim)  # First linear transformation
        self.w2 = nn.Linear(in_dim, dim)  # Second linear transformation (gate)
        self.w3 = nn.Linear(dim, in_dim)  # Output linear transformation

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the SwiGLU module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after SwiGLU activation.
        """
        # Apply SwiGLU: output_projection(gate * SiLU(input_projection_1))
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class MPNN(nn.Module):
    """
    Message Passing Neural Network (MPNN) layer (see Gilmer et al., 2017).

    The MPNN updates node features based on messages aggregated from their neighbors.
    This implementation uses fused operations for potential performance improvements.

    Message Calculation: m_ij = MLP([h_i, h_j, pos_i - pos_j])
    Aggregation: m_i = mean(m_ij)
    Update: h_i' = MLP([h_i, m_i])
    """

    def __init__(self, dim: int, mp_steps: int, dimensionality: int = 3):
        """
        Initializes the MPNN module.

        Args:
            dim (int): Dimension of node features.
            mp_steps (int): Number of message passing steps.
            dimensionality (int): Dimensionality of the coordinates.
        """
        super().__init__()
        # Fused message and update functions for each step
        self.fused_message_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        2 * dim + dimensionality, dim
                    ),  # Takes [h_i, h_j, pos_i - pos_j]
                    nn.GELU(),
                    nn.LayerNorm(
                        dim, elementwise_affine=False
                    ),  # Non-affine LayerNorm for speed
                )
                for _ in range(mp_steps)
            ]
        )

        self.update_fns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * dim, dim),  # Takes [h_i, aggregated_message_i]
                    nn.LayerNorm(
                        dim, elementwise_affine=False
                    ),  # Non-affine LayerNorm for speed
                )
                for _ in range(mp_steps)
            ]
        )

    def layer(
        self,
        message_fn: nn.Module,
        update_fn: nn.Module,
        h: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        """
        Performs a single layer of message passing.

        Args:
            message_fn (nn.Module): The neural network for computing messages.
            update_fn (nn.Module): The neural network for updating node features.
            h (torch.Tensor): Current node features.
            edge_attr (torch.Tensor): Edge attributes (e.g., relative positions).
            edge_index (torch.Tensor): Edge indices defining the graph structure.

        Returns:
            torch.Tensor: Updated node features.
        """
        row, col = edge_index  # Source and target nodes for each edge

        # --- Message Calculation ---
        # Concatenate features of source, target nodes, and edge attributes
        messages_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
        messages = message_fn(messages_input)  # Compute messages using the message_fn

        # --- Aggregation ---
        # Aggregate messages for each node (scatter_mean averages messages for each target node)
        aggregated_message = scatter_mean(
            messages, col, h.size(0)
        )  # h.size(0) is the number of nodes

        # --- Update ---
        # Concatenate current node features with aggregated messages
        update_input = torch.cat([h, aggregated_message], dim=-1)
        update = update_fn(update_input)  # Compute update using the update_fn
        h.add_(update)  # In-place addition for memory efficiency
        return h

    @torch.no_grad()
    def compute_edge_attr(self, pos: torch.Tensor, edge_index: torch.Tensor):
        """
        Computes edge attributes (relative positions) directly.

        Args:
            pos (torch.Tensor): Node positions.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Edge attributes (pos_i - pos_j).
        """
        # Efficiently compute difference between positions of connected nodes
        return pos.index_select(0, edge_index[0]) - pos.index_select(0, edge_index[1])

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass of the MPNN module.

        Args:
            x (torch.Tensor): Initial node features.
            pos (torch.Tensor): Node positions.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Node features after message passing.
        """
        edge_attr = self.compute_edge_attr(pos, edge_index)  # Calculate edge attributes
        # Perform message passing for the specified number of steps
        for message_fn, update_fn in zip(self.fused_message_update, self.update_fns):
            x = self.layer(message_fn, update_fn, x, edge_attr, edge_index)
        return x


@dataclass
class Node:
    """
    Dataclass to store hierarchical node information for the Erwin Transformer.

    This structure helps manage features, positions, and batch information
    as data flows through the encoder-decoder architecture.
    """

    x: torch.Tensor  # Node features
    pos: torch.Tensor  # Node positions
    batch_idx: torch.Tensor  # Batch index for each node
    tree_idx_rot: torch.Tensor | None = (
        None  # Indices for rotated tree (used for rotated attention)
    )
    children: Node | None = None  # Reference to child nodes (used in pooling/unpooling)


class BallPooling(nn.Module):
    """
    Pooling operation for leaf nodes within a ball (eq. 12).

    This module coarsens the representation by:
    1. Selecting balls of a specified 'stride'.
    2. Concatenating leaf node features within each ball along with their relative positions to the ball center.
    3. Applying a linear projection and batch normalization.
    The output represents the center of each ball endowed with the pooled features.
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3):
        """
        Initializes the BallPooling module.

        Args:
            in_dim (int): Dimension of input features per node.
            out_dim (int): Dimension of output features per pooled ball.
            stride (int): Pooling stride (number of nodes pooled into one).
            dimensionality (int): Dimensionality of the coordinates.
        """
        super().__init__()
        self.stride = stride
        # Linear layer projects concatenated features and relative positions
        self.proj = nn.Linear(stride * in_dim + stride * dimensionality, out_dim)
        self.norm = nn.BatchNorm1d(
            out_dim
        )  # Batch normalization for the pooled features

    def forward(self, node: Node) -> Node:
        """
        Forward pass of the BallPooling module.

        Args:
            node (Node): Input Node object containing features, positions, etc.

        Returns:
            Node: Output Node object with pooled features and updated positions/batch_idx.
        """
        if self.stride == 1:  # No pooling if stride is 1
            return Node(x=node.x, pos=node.pos, batch_idx=node.batch_idx, children=node)

        with torch.no_grad():  # Operations here don't require gradients
            # --- Prepare Pooled Node Information ---
            # Select batch indices for the new pooled nodes (centers of balls)
            batch_idx = node.batch_idx[:: self.stride]
            # Calculate new centers by averaging positions within each ball
            centers = reduce(
                node.pos, "(n s) d -> n d", "mean", s=self.stride
            )  # (num_balls, dimensionality)
            # Reshape positions for relative position calculation
            pos = rearrange(
                node.pos, "(n s) d -> n s d", s=self.stride
            )  # (num_balls, stride, dimensionality)
            # Calculate relative positions of nodes within each ball to their center
            rel_pos = rearrange(
                pos - centers[:, None], "n s d -> n (s d)"
            )  # (num_balls, stride * dimensionality)

        # --- Feature Pooling ---
        # Concatenate features of nodes within each ball and their relative positions
        x_pooled_input = torch.cat(
            [rearrange(node.x, "(n s) c -> n (s c)", s=self.stride), rel_pos], dim=1
        )  # (num_balls, stride * in_dim + stride * dimensionality)
        # Project and normalize the pooled features
        x_pooled = self.norm(self.proj(x_pooled_input))

        # Return a new Node object representing the pooled level
        return Node(
            x=x_pooled, pos=centers, batch_idx=batch_idx, children=node
        )  # Store original node as children


class BallUnpooling(nn.Module):
    """
    Ball unpooling operation for refining features (eq. 13).

    This module refines features from a coarser level to a finer level by:
    1. Computing relative positions of children nodes (from before pooling) to the center of the parent ball.
    2. Concatenating the pooled (parent) features with these relative positions.
    3. Applying a linear projection and adding this back to the original children features (self-connection), followed by batch normalization.
    The output is a refined tree with the same number of nodes as before pooling.
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3):
        """
        Initializes the BallUnpooling module.

        Args:
            in_dim (int): Dimension of input features (from the coarser/parent level).
            out_dim (int): Dimension of output features (for the finer/child level).
            stride (int): Unpooling stride (number of child nodes corresponding to one parent).
            dimensionality (int): Dimensionality of the coordinates.
        """
        super().__init__()
        self.stride = stride
        # Linear layer projects concatenated parent features and relative positions of children
        self.proj = nn.Linear(in_dim + stride * dimensionality, stride * out_dim)
        self.norm = nn.BatchNorm1d(
            out_dim
        )  # Batch normalization for the refined child features

    def forward(self, node: Node) -> Node:
        """
        Forward pass of the BallUnpooling module.

        Args:
            node (Node): Input Node object from the coarser level, containing 'children' from the finer level.

        Returns:
            Node: The 'children' Node object with refined features.
        """
        with torch.no_grad():  # Operations here don't require gradients
            # --- Calculate Relative Positions of Children ---
            # Reshape children positions and parent positions for broadcasting
            children_pos_reshaped = rearrange(
                node.children.pos, "(n m) d -> n m d", m=self.stride
            )  # (num_parents, stride, dim)
            parent_pos_expanded = node.pos[:, None]  # (num_parents, 1, dim)
            # Compute relative positions of children wrt their parent's center
            rel_pos = children_pos_reshaped - parent_pos_expanded
            rel_pos = rearrange(
                rel_pos, "n m d -> n (m d)"
            )  # (num_parents, stride * dimensionality)

        # --- Feature Refinement ---
        # Concatenate parent features with the relative positions of children
        x_unpooled_input = torch.cat(
            [node.x, rel_pos], dim=-1
        )  # (num_parents, in_dim + stride * dimensionality)
        # Project the concatenated features and reshape to match children's feature dimensions
        projected_features = rearrange(
            self.proj(x_unpooled_input), "n (m d) -> (n m) d", m=self.stride
        )  # (num_children, out_dim)

        # Add projected features to original children features (skip connection) and normalize
        node.children.x = self.norm(node.children.x + projected_features)

        return (
            node.children
        )  # Return the Node object corresponding to the refined finer level


class ErwinTransformerBlock(nn.Module):
    """
    A single block of the Erwin Transformer.

    Each block consists of:
    1. Layer Normalization (RMSNorm)
    2. Ball Multi-Head Self-Attention (BallMSA)
    3. Residual Connection
    4. Layer Normalization (RMSNorm)
    5. SwiGLU Feed-Forward Network
    6. Residual Connection
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        dimensionality: int = 3,
    ):
        """
        Initializes the ErwinTransformerBlock.

        Args:
            dim (int): Dimension of input and output features.
            num_heads (int): Number of attention heads for BallMSA.
            ball_size (int): Ball size for BallMSA.
            mlp_ratio (int): Ratio to determine the hidden dimension of the SwiGLU MLP.
                             Hidden dimension = dim * mlp_ratio.
            dimensionality (int): Dimensionality of the coordinates.
        """
        super().__init__()
        self.ball_size = ball_size  # Store ball_size, might be used if BMSA needs it directly (though typically passed via pos)
        self.norm1 = nn.RMSNorm(dim)  # First layer normalization
        self.norm2 = nn.RMSNorm(dim)  # Second layer normalization
        self.BMSA = BallMSA(
            dim, num_heads, ball_size, dimensionality
        )  # Ball-based attention
        self.swiglu = SwiGLU(dim, dim * mlp_ratio)  # SwiGLU MLP

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        """
        Forward pass of the ErwinTransformerBlock.

        Args:
            x (torch.Tensor): Input features.
            pos (torch.Tensor): Node positions (for BallMSA).

        Returns:
            torch.Tensor: Output features after the transformer block.
        """
        # First sub-layer: Attention
        x = x + self.BMSA(
            self.norm1(x), pos
        )  # Residual connection around normalized input + attention
        # Second sub-layer: Feed-Forward Network
        x = x + self.swiglu(
            self.norm2(x)
        )  # Residual connection around normalized input + FFN
        return x


class BasicLayer(nn.Module):
    """
    A basic layer in the Erwin Transformer, which can be part of the encoder, decoder, or bottleneck.

    It consists of multiple ErwinTransformerBlocks and optional pooling or unpooling operations.
    """

    def __init__(
        self,
        direction: Literal[
            "down", "up", None
        ],  # "down": encoder, "up": decoder, None: bottleneck
        depth: int,  # Number of ErwinTransformerBlocks in this layer
        stride: int | None,  # Stride for pooling (if "down") or unpooling (if "up")
        in_dim: int,  # Input feature dimension
        out_dim: int,  # Output feature dimension (after pooling/unpooling or if bottleneck)
        num_heads: int,  # Number of attention heads for blocks
        ball_size: int,  # Ball size for blocks
        mlp_ratio: int,  # MLP ratio for blocks
        rotate: bool,  # Whether to enable rotation for attention in blocks
        dimensionality: int = 3,
    ):
        super().__init__()
        # Determine hidden dimension based on direction (encoder vs. decoder/bottleneck)
        hidden_dim = in_dim if direction == "down" else out_dim

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList(
            [
                ErwinTransformerBlock(
                    hidden_dim, num_heads, ball_size, mlp_ratio, dimensionality
                )
                for _ in range(depth)
            ]
        )
        # Determine rotation schedule for blocks (alternating if enabled)
        self.rotate_schedule = (
            [i % 2 == 1 for i in range(depth)] if rotate else [False] * depth
        )  # Start rotation from the second block if enabled

        # --- Pooling/Unpooling ---
        self.pool = lambda node: node  # Default: no pooling
        self.unpool = lambda node: node  # Default: no unpooling

        if direction == "down" and stride is not None:
            self.pool = BallPooling(hidden_dim, out_dim, stride, dimensionality)
        elif direction == "up" and stride is not None:
            self.unpool = BallUnpooling(
                in_dim, hidden_dim, stride, dimensionality
            )  # Note: in_dim for unpooling is parent's dim

    def forward(self, node: Node) -> Node:
        """
        Forward pass of the BasicLayer.

        Args:
            node (Node): Input Node object.

        Returns:
            Node: Output Node object after processing by blocks and pooling/unpooling.
        """
        # --- Unpooling (for decoder layers) ---
        node = self.unpool(node)

        # --- Prepare for Rotated Attention (if applicable) ---
        # Check if any block in this layer uses rotation
        uses_rotation = any(self.rotate_schedule)
        if uses_rotation:
            assert (
                node.tree_idx_rot is not None
            ), "tree_idx_rot must be provided for rotation in BasicLayer"
            # Precompute inverse rotation mapping if rotation is used
            tree_idx_rot_inv = torch.argsort(
                node.tree_idx_rot
            )  # Map from rotated to original indices

        # --- Apply Transformer Blocks ---
        for i, blk in enumerate(self.blocks):
            should_rotate_this_block = self.rotate_schedule[i]
            if should_rotate_this_block:
                # Apply block to rotated features and positions, then un-rotate
                rotated_x = node.x[node.tree_idx_rot]
                rotated_pos = node.pos[node.tree_idx_rot]
                processed_rotated_x = blk(rotated_x, rotated_pos)
                node.x = processed_rotated_x[tree_idx_rot_inv]
            else:
                # Apply block to original features and positions
                node.x = blk(node.x, node.pos)

        # --- Pooling (for encoder layers) ---
        return self.pool(node)


class ErwinTransformer(nn.Module):
    """
    Erwin Transformer main model.

    This model implements a U-Net like architecture using Ball Multi-Head Self-Attention
    for processing point cloud data. It includes an encoder, a bottleneck, and an optional decoder.
    The architecture leverages hierarchical ball-based processing.

    Args:
        c_in (int): Number of input channels (initial feature dimension).
        c_hidden (List[int]): List of hidden channel dimensions for each encoder layer + bottleneck.
                              The decoder will use these in reverse.
        ball_sizes (List[int]): List of ball sizes for each encoder layer (and bottleneck).
                                The decoder uses these in reverse.
        enc_num_heads (List[int]): List of number of attention heads for each encoder layer (and bottleneck).
        enc_depths (List[int]): List of number of ErwinTransformerBlock layers for each encoder layer (and bottleneck).
        dec_num_heads (List[int]): List of number of attention heads for each decoder layer.
        dec_depths (List[int]): List of number of ErwinTransformerBlock layers for each decoder layer.
        strides (List[int]): List of strides for pooling in each encoder layer.
                             The decoder uses these in reverse for unpooling.
        rotate (bool): Whether to enable rotated attention mechanism.
        decode (bool): If True, the model includes a decoder to reconstruct to the original resolution.
                       If False, returns the latent representation from the bottleneck.
        mlp_ratio (int): Ratio for the hidden dimension in SwiGLU MLPs within blocks.
        dimensionality (int): Dimensionality of the input point coordinates (e.g., 3 for 3D).
        mp_steps (int): Number of message passing steps in the initial ErwinEmbedding.

    Notes:
        - Lengths of `ball_sizes`, `enc_num_heads`, `enc_depths` must be the same (N),
          representing N levels in the encoder including the bottleneck.
        - Lengths of `strides`, `dec_num_heads`, `dec_depths` must be N - 1,
          as they correspond to transitions between levels.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: List[int],
        ball_sizes: List[int],
        enc_num_heads: List[int],
        enc_depths: List[int],
        dec_num_heads: List[int],
        dec_depths: List[int],
        strides: List[int],
        rotate: bool,  # Changed from int to bool for clarity
        decode: bool = True,
        mlp_ratio: int = 4,
        dimensionality: int = 3,
        mp_steps: int = 3,
    ):
        super().__init__()
        # --- Parameter Validation ---
        num_encoder_levels = len(ball_sizes)
        assert len(enc_num_heads) == num_encoder_levels, "enc_num_heads length mismatch"
        assert len(enc_depths) == num_encoder_levels, "enc_depths length mismatch"
        assert (
            len(c_hidden) == num_encoder_levels
        ), "c_hidden length mismatch for encoder/bottleneck"

        num_transitions = num_encoder_levels - 1
        assert len(strides) == num_transitions, "strides length mismatch"
        if decode:
            assert (
                len(dec_num_heads) == num_transitions
            ), "dec_num_heads length mismatch"
            assert len(dec_depths) == num_transitions, "dec_depths length mismatch"

        self.rotate = rotate
        self.decode = decode
        self.ball_sizes = ball_sizes  # Store for ball tree construction
        self.strides = strides  # Store for ball tree construction

        # --- Initial Embedding ---
        self.embed = ErwinEmbedding(c_in, c_hidden[0], mp_steps, dimensionality)

        # --- Encoder Layers ---
        self.encoder = nn.ModuleList()
        for i in range(num_transitions):  # num_transitions = num_encoder_levels - 1
            self.encoder.append(
                BasicLayer(
                    direction="down",
                    depth=enc_depths[i],
                    stride=strides[i],
                    in_dim=c_hidden[i],
                    out_dim=c_hidden[i + 1],
                    num_heads=enc_num_heads[i],
                    ball_size=ball_sizes[i],
                    rotate=self.rotate,
                    mlp_ratio=mlp_ratio,
                    dimensionality=dimensionality,
                )
            )

        # --- Bottleneck Layer ---
        self.bottleneck = BasicLayer(
            direction=None,  # No pooling/unpooling
            depth=enc_depths[-1],  # Last depth in enc_depths
            stride=None,
            in_dim=c_hidden[-1],  # Last hidden dim
            out_dim=c_hidden[-1],
            num_heads=enc_num_heads[-1],  # Last num_heads in enc_num_heads
            ball_size=ball_sizes[-1],  # Last ball_size
            rotate=self.rotate,
            mlp_ratio=mlp_ratio,
            dimensionality=dimensionality,
        )

        # --- Decoder Layers (Optional) ---
        if self.decode:
            self.decoder = nn.ModuleList()
            # Iterate in reverse order of encoder transitions
            for i in range(num_transitions - 1, -1, -1):
                self.decoder.append(
                    BasicLayer(
                        direction="up",
                        depth=dec_depths[i],
                        stride=strides[i],  # Use corresponding stride for unpooling
                        in_dim=c_hidden[
                            i + 1
                        ],  # Input from previous (coarser) decoder layer or bottleneck
                        out_dim=c_hidden[
                            i
                        ],  # Output to match corresponding encoder layer's feature dim
                        num_heads=dec_num_heads[i],
                        ball_size=ball_sizes[i],  # Use corresponding ball_size
                        rotate=self.rotate,
                        mlp_ratio=mlp_ratio,
                        dimensionality=dimensionality,
                    )
                )
        else:
            self.decoder = None  # Explicitly set to None if not decoding

        self.in_dim = c_in
        self.out_dim = c_hidden[
            0
        ]  # Output dimension if decoding, otherwise bottleneck dim
        self.apply(self._init_weights)  # Initialize weights

    def _init_weights(self, m: nn.Module):
        """
        Initializes weights of Linear and LayerNorm layers.
        """
        if isinstance(m, nn.Linear):
            # Truncated normal initialization for Linear layer weights
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Zero initialization for biases
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Zero initialization for LayerNorm bias
            if m.weight is not None:
                nn.init.constant_(
                    m.weight, 1.0
                )  # Ones initialization for LayerNorm weight

    def forward(
        self,
        node_features: torch.Tensor,
        node_positions: torch.Tensor,
        batch_idx: torch.Tensor,
        edge_index: torch.Tensor = None,
        tree_idx: torch.Tensor = None,
        tree_mask: torch.Tensor = None,
        radius: float = None,
        **kwargs,
    ):
        # --- Preprocessing and Initialization ---
        with torch.no_grad():
            # Build ball tree and radius graph if not provided
            if tree_idx is None and tree_mask is None:
                tree_idx, tree_mask, tree_idx_rot = build_balltree_with_rotations(
                    node_positions,
                    batch_idx,
                    self.strides,
                    self.ball_sizes,
                    self.rotate,
                )
            if edge_index is None and self.embed.mp_steps:
                assert (
                    radius is not None
                ), "radius (float) must be provided if edge_index is not given to build radius graph"
                edge_index = torch_cluster.radius_graph(
                    node_positions, radius, batch=batch_idx, loop=True
                )

        # --- Embedding ---
        x = self.embed(node_features, node_positions, edge_index)

        # --- Node Initialization ---
        # Create the initial Node object with features, positions, and batch indices
        # from the ball tree.
        node = Node(
            x=x[tree_idx],
            pos=node_positions[tree_idx],
            batch_idx=batch_idx[tree_idx],
            tree_idx_rot=None,  # Rotated tree indices will be populated during encoding
        )

        # --- Encoding ---
        # Pass the node through the encoder layers
        for layer in self.encoder:
            node.tree_idx_rot = tree_idx_rot.pop(
                0
            )  # Get rotated indices for this layer
            node = layer(node)

        # --- Bottleneck ---
        # Pass the node through the bottleneck layer
        node.tree_idx_rot = tree_idx_rot.pop(
            0
        )  # Get rotated indices for the bottleneck
        node = self.bottleneck(node)

        # --- Decoding (Optional) ---
        if self.decode:
            # Pass the node through the decoder layers
            for layer in self.decoder:
                node = layer(node)
            # Return the decoded features, reordered to match the original input order
            return node.x[tree_mask][torch.argsort(tree_idx[tree_mask])]

        # --- Output (if not decoding) ---
        # Return the latent representation and batch indices from the bottleneck
        return node.x, node.batch_idx

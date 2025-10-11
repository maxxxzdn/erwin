from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
import torch_scatter
from einops import rearrange, reduce
from flash_attn import flash_attn_qkvpacked_func

from balltree import build_balltree_with_rotations


class SwiGLU(nn.Module):
    """
    SwiGLU activation function module: W_3 SiLU(W_1 x) ⊗ W_2 x.
    """

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, dim)
        self.w2 = nn.Linear(in_dim, dim)
        self.w3 = nn.Linear(dim, in_dim)

    def forward(self, x: torch.Tensor):
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class MPNN(nn.Module):
    """
    Message Passing Neural Network (MPNN) layer (see Gilmer et al., 2017).

        Message Calculation: m_ij = MLP([h_i, h_j, pos_i - pos_j])
        Aggregation: m_i = mean(m_ij)
        Update: h_i' = MLP([h_i, m_i])
    """

    def __init__(self, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        self.fused_message_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        2 * dim + dimensionality, dim
                    ),  # Takes [h_i, h_j, pos_i - pos_j]
                    nn.GELU(),
                    nn.LayerNorm(
                        dim, elementwise_affine=False
                    ),
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
                    ),
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
        row, col = edge_index

        messages = message_fn(
            torch.cat([h[row], h[col], edge_attr], dim=-1)
        )

        aggregated_message = torch_scatter.scatter_mean(
            messages, col, dim=0, dim_size=h.size(0)
        )

        update_input = torch.cat([h, aggregated_message], dim=-1)
        h.add_(update_fn(update_input))

        return h

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        edge_attr = pos.index_select(0, edge_index[0]) - pos.index_select(0, edge_index[1])
        for message_fn, update_fn in zip(self.fused_message_update, self.update_fns):
            x = self.layer(message_fn, update_fn, x, edge_attr, edge_index)
        return x


class BallMSA(nn.Module):
    """
    Ball Multi-Head Self-Attention (BMSA) module (eq. 8) using Flash Attention with packed QKV.
    """

    def __init__(
        self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 3
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(
            dimensionality, dim
        )

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """
        Computes relative positions of leaf nodes with respect to the center of their ball (eq. 9).
        """
        num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        pos = pos.view(
            num_balls, self.ball_size, dim
        )
        # Subtract ball center from each point's position
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    @torch.amp.autocast('cuda', dtype=torch.bfloat16)
    def forward(self, x: torch.Tensor, pos: torch.Tensor):

        x = x + self.pe_proj(self.compute_rel_pos(pos))

        qkv = rearrange(
            self.qkv(x),
            "(b n) (three h d) -> b n three h d",
            b=x.shape[0] // self.ball_size,
            n=self.ball_size,
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        x = rearrange(
            flash_attn_qkvpacked_func(qkv), 
            "b n h d -> (b n) (h d)"
        )

        return self.proj(x)


class ErwinEmbedding(nn.Module):
    """
    Embedding module for Erwin Transformer: Linear + MPNN.
    """

    def __init__(self, in_dim: int, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        self.mp_steps = mp_steps
        self.embed_fn = nn.Linear(in_dim, dim)
        self.mpnn = MPNN(dim, mp_steps, dimensionality) if mp_steps > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        x = self.embed_fn(x)
        return self.mpnn(x, pos, edge_index) if self.mp_steps > 0 else x


@dataclass
class Node:
    """ Dataclass to store the hierarchical node information."""
    x: torch.Tensor
    pos: torch.Tensor 
    batch_idx: torch.Tensor
    tree_idx_rot: torch.Tensor | None = None
    children: Node | None = None


class BallPooling(nn.Module):
    """
    Pooling operation for leaf nodes within a ball (eq. 12).

    This module coarsens the representation by:
    1. Selecting balls of a specified 'stride'.
    2. Concatenating leaf node features within each ball along with their relative positions to the ball center.
    3. Applying a linear projection and batch normalization.
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(stride * in_dim + stride * dimensionality, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, node: Node) -> Node:
        if self.stride == 1:
            return Node(x=node.x, pos=node.pos, batch_idx=node.batch_idx, children=node)

        with torch.no_grad():
            batch_idx = node.batch_idx[:: self.stride]
            centers = reduce(node.pos, "(n s) d -> n d", "mean", s=self.stride)
            pos = rearrange(node.pos, "(n s) d -> n s d", s=self.stride)
            rel_pos = rearrange(pos - centers[:, None], "n s d -> n (s d)")

        x_pooled_input = torch.cat([rearrange(node.x, "(n s) c -> n (s c)", s=self.stride), rel_pos], dim=1)
        x_pooled = self.norm(self.proj(x_pooled_input))

        return Node(x=x_pooled, pos=centers, batch_idx=batch_idx, children=node)


class BallUnpooling(nn.Module):
    """
    Ball unpooling operation for refining features (eq. 13).

    This module refines features from a coarser level to a finer level by:
    1. Computing relative positions of children nodes (from before pooling) to the center of the parent ball.
    2. Concatenating the pooled (parent) features with these relative positions.
    3. Applying a linear projection and adding this back to the original children features (self-connection), followed by batch normalization.
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_dim + stride * dimensionality, stride * out_dim)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, node: Node) -> Node:
        with torch.no_grad():
            children_pos_reshaped = rearrange(node.children.pos, "(n m) d -> n m d", m=self.stride)
            rel_pos = rearrange(children_pos_reshaped - node.pos[:, None], "n m d -> n (m d)")

        x_unpooled_input = torch.cat([node.x, rel_pos], dim=-1)
        projected_features = rearrange(self.proj(x_unpooled_input), "n (m d) -> (n m) d", m=self.stride)
        node.children.x = self.norm(node.children.x + projected_features)

        return node.children


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
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.BMSA = BallMSA(dim, num_heads, ball_size, dimensionality)
        self.swiglu = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.BMSA(self.norm1(x), pos)
        x = x + self.swiglu(self.norm2(x))
        return x


class BasicLayer(nn.Module):

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
        self.rotate = rotate
        hidden_dim = in_dim if direction == "down" else out_dim

        self.blocks = nn.ModuleList(
            [
                ErwinTransformerBlock(
                    hidden_dim, num_heads, ball_size, mlp_ratio, dimensionality
                )
                for _ in range(depth)
            ]
        )
        self.rotate_schedule = ([i % 2 == 1 for i in range(depth)] if rotate else [False] * depth)

        self.pool = lambda node: node  # Default: no pooling
        self.unpool = lambda node: node  # Default: no unpooling

        if direction == "down" and stride is not None:
            self.pool = BallPooling(hidden_dim, out_dim, stride, dimensionality)
        elif direction == "up" and stride is not None:
            self.unpool = BallUnpooling(in_dim, hidden_dim, stride, dimensionality)

    def forward(self, node: Node) -> Node:
        node = self.unpool(node)

        if self.rotate:
            assert node.tree_idx_rot is not None, "tree_idx_rot must be provided for rotation in BasicLayer"

        for i, blk in enumerate(self.blocks):
            if self.rotate_schedule[i]:
                node.x[node.tree_idx_rot] = blk(node.x[node.tree_idx_rot], node.pos[node.tree_idx_rot])
            else:
                node.x = blk(node.x, node.pos)

        return self.pool(node)

   
@torch.no_grad()
def prepare_tree(node_positions, batch_idx, strides, ball_sizes, rotate):
    tree_idx, tree_mask, tree_idx_rot = build_balltree_with_rotations(
        node_positions,
        batch_idx,
        strides,
        ball_sizes,
        rotate,
    )
    tree_bwd_idx = torch.argsort(tree_idx[tree_mask])
    return tree_idx, tree_mask, tree_idx_rot, tree_bwd_idx


@torch.no_grad()
def prepare_graph(node_positions, batch_idx, radius):
    return torch_cluster.radius_graph(node_positions, radius, batch=batch_idx, loop=True)


class ErwinTransformer(nn.Module):
    """
    Erwin Transformer.

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
        mp_steps: int,
        rotate: bool,
        decode: bool = True,
        mlp_ratio: int = 4,
        dimensionality: int = 3,
    ):
        super().__init__()
        num_encoder_levels = len(ball_sizes)
        assert len(enc_num_heads) == num_encoder_levels, "enc_num_heads length mismatch"
        assert len(enc_depths) == num_encoder_levels, "enc_depths length mismatch"
        assert len(c_hidden) == num_encoder_levels, "c_hidden length mismatch for encoder/bottleneck"

        num_transitions = num_encoder_levels - 1
        assert len(strides) == num_transitions, "strides length mismatch"
        if decode:
            assert len(dec_num_heads) == num_transitions, "dec_num_heads length mismatch"
            assert len(dec_depths) == num_transitions, "dec_depths length mismatch"

        self.rotate = rotate
        self.decode = decode
        self.ball_sizes = ball_sizes
        self.strides = strides

        # --- Initial Embedding ---
        self.embed = ErwinEmbedding(c_in, c_hidden[0], mp_steps, dimensionality)

        # --- Encoder Layers ---
        self.encoder = nn.ModuleList()
        for i in range(num_transitions):
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
            direction=None,
            depth=enc_depths[-1],
            stride=None,
            in_dim=c_hidden[-1],
            out_dim=c_hidden[-1],
            num_heads=enc_num_heads[-1],
            ball_size=ball_sizes[-1],
            rotate=self.rotate,
            mlp_ratio=mlp_ratio,
            dimensionality=dimensionality,
        )

        # --- Decoder Layers (Optional) ---
        if self.decode:
            self.decoder = nn.ModuleList()
            for i in range(num_transitions - 1, -1, -1):
                self.decoder.append(
                    BasicLayer(
                        direction="up",
                        depth=dec_depths[i],
                        stride=strides[i],
                        in_dim=c_hidden[i + 1],
                        out_dim=c_hidden[i],
                        num_heads=dec_num_heads[i],
                        ball_size=ball_sizes[i],
                        rotate=self.rotate,
                        mlp_ratio=mlp_ratio,
                        dimensionality=dimensionality,
                    )
                )
        else:
            self.decoder = None

        self.in_dim = c_in
        self.out_dim = c_hidden[0]
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initializes weights of Linear and LayerNorm layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(
                    m.weight, 1.0
                )

    def forward(
        self,
        node_features: torch.Tensor,
        node_positions: torch.Tensor,
        batch_idx: torch.Tensor,
        tree_idx: torch.Tensor = None,
        tree_mask: torch.Tensor = None,
        tree_idx_rot: List[torch.Tensor] = None,
        tree_bwd_idx: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        radius: float = None,
        **kwargs,
    ):
        
        if tree_idx is None or tree_mask is None or tree_idx_rot is None or tree_bwd_idx is None:
            # NOTE: optimally, this should be precomputed and passed to the model to avoid overhead
            tree_idx, tree_mask, tree_idx_rot, tree_bwd_idx = prepare_tree(
                node_positions, batch_idx, self.strides, self.ball_sizes, self.rotate
            )

        if edge_index is None and self.embed.mp_steps > 0:
            # NOTE: radius_graph usually is not an optimal choice, but it is fast
            assert radius is not None, "Radius must be provided if using MPNN in embedding"
            edge_index = prepare_graph(node_positions, batch_idx, radius=radius)

        x = self.embed(node_features, node_positions, edge_index)

        node = Node(
            x=x[tree_idx],
            pos=node_positions[tree_idx],
            batch_idx=batch_idx[tree_idx],
            tree_idx_rot=None,
        )

        for layer in self.encoder:
            node.tree_idx_rot = tree_idx_rot.pop(0)
            node = layer(node)

        node.tree_idx_rot = tree_idx_rot.pop(0)
        node = self.bottleneck(node)

        if self.decode:
            for layer in self.decoder:
                node = layer(node)
            return node.x[tree_mask][tree_bwd_idx]

        return node.x, node.batch_idx

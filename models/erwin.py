from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from einops import rearrange, reduce

from typing import Literal, List
from dataclasses import dataclass

from balltree import build_balltree_torch


def scatter_mean(src: torch.Tensor, idx: torch.Tensor, num_receivers: int):
    """ 
    Averages all values from src into the receivers at the indices specified by idx.

    Args:
        src (torch.Tensor): Source tensor of shape (N, D).
        idx (torch.Tensor): Indices tensor of shape (N,).
        num_receivers (int): Number of receivers (usually the maximum index in idx + 1).
    
    Returns:
        torch.Tensor: Result tensor of shape (num_receivers, D).
    """
    result = torch.zeros(num_receivers, src.size(1), dtype=src.dtype, device=src.device)
    count = torch.zeros(num_receivers, dtype=torch.long, device=src.device)
    result.index_add_(0, idx, src)
    count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    result = result / count.unsqueeze(1).clamp(min=1)
    return result


class SwiGLU(nn.Module):
    """ W_3 SiLU(W_1 x) ⊗ W_2 x """
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, dim)
        self.w2 = nn.Linear(in_dim, dim)
        self.w3 = nn.Linear(dim, in_dim)
    
    def forward(self, x: torch.Tensor):
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class MPNN(nn.Module):
    """ 
    Message Passing Neural Network (see Gilmer et al., 2017).
        m_ij = MLP([h_i, h_j, pos_i - pos_j])       message
        m_i = mean(m_ij)                            aggregate
        h_i' = MLP([h_i, m_i])                      update

    """
    def __init__(self, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        self.message_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * dim + dimensionality, dim), 
                nn.GELU(), 
                nn.LayerNorm(dim)
            ) for _ in range(mp_steps)
        ])

        self.update_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * dim, dim), 
                nn.LayerNorm(dim)
            ) for _ in range(mp_steps)
        ])       

    def layer(self, message_fn: nn.Module, update_fn: nn.Module, h: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor):
        row, col = edge_index
        messages = message_fn(torch.cat([h[row], h[col], edge_attr], dim=-1))
        message = scatter_mean(messages, col, h.size(0))
        update = update_fn(torch.cat([h, message], dim=-1))
        return h + update

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        edge_attr = pos[edge_index[0]] - pos[edge_index[1]]
        for message_fn, update_fn in zip(self.message_fns, self.update_fns):
            x = self.layer(message_fn, update_fn, x, edge_attr, edge_index)
        return x


class ErwinEmbedding(nn.Module):
    """ Linear projection -> MPNN."""
    def __init__(self, in_dim: int, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        self.embed_fn = nn.Linear(in_dim, dim)
        self.mpnn = MPNN(dim, mp_steps, dimensionality)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        return self.mpnn(self.embed_fn(x), pos, edge_index)


@dataclass
class Node:
    """ Dataclass to store the hierarchical node information."""
    x: torch.Tensor
    pos: torch.Tensor 
    batch_idx: torch.Tensor
    children: Node | None = None


class BallPooling(nn.Module):
    """ 
    Pooling of leaf nodes in a ball (eq. 12):
        1. select balls of size 'stride'.
        2. concatenate leaf nodes inside each ball along with their relative positions to the ball center.
        3. apply linear projection and batch normalization.
        4. the output is the center of each ball endowed with the pooled features.
    """
    def __init__(self, dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        input_dim = stride * dim + stride * dimensionality
        self.proj = nn.Linear(input_dim, 2 * dim)
        self.norm = nn.BatchNorm1d(2 * dim)

    def forward(self, node: Node) -> Node:
        if self.stride == 1: # no pooling
            return Node(x=node.x, pos=node.pos, batch_idx=node.batch_idx, children=node)

        centers = reduce(node.pos, "(n s) d -> n d", 'mean', s=self.stride)
        pos = rearrange(node.pos, "(n s) d -> n s d", s=self.stride)

        x = torch.cat([
            rearrange(node.x, "(n s) c -> n (s c)", s=self.stride),
            rearrange(pos - centers[:, None], "n s d -> n (s d)"),
        ], dim=1)
        
        x = self.norm(self.proj(x))

        batch_idx = node.batch_idx[::self.stride].contiguous()
        return Node(x=x, pos=centers, batch_idx=batch_idx, children=node)


class BallUnpooling(nn.Module):
    """
    Ball unpooling (refinement; eq. 13):
        1. compute relative positions of children (from before pooling) to the center of the ball.
        2. concatenate the pooled features with the relative positions.
        3. apply linear projection and self-connection followed by batch normalization.
        4. the output is a refined tree with the same number of nodes as before pooling.
    """
    def __init__(self, dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        input_dim = 2 * dim + stride * dimensionality
        self.proj = nn.Linear(input_dim, stride * dim)         
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, node: Node) -> Node:
        rel_pos = rearrange(node.children.pos, "(n m) d -> n m d", m=self.stride) - node.pos[:, None]

        x = torch.cat([
            node.x,
            rearrange(rel_pos, "n m d -> n (m d)")
        ], dim=-1)

        node.children.x = self.norm(node.children.x + rearrange(self.proj(x), "n (m d) -> (n m) d", m=self.stride))
        return node.children


class BallMSA(nn.Module):
    """ Ball Multi-Head Self-Attention (BMSA) module (eq. 8). """
    def __init__(self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ball_size = ball_size

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(dimensionality, dim)
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    def create_attention_mask(self, pos: torch.Tensor):
        """ Distance-based attention bias (eq. 10). """
        pos = rearrange(pos, '(n m) d -> n m d', m=self.ball_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)

    def rpe(self, pos: torch.Tensor):
        """ Relative position of leafs wrt the center of the ball (eq. 9). """
        centers = reduce(pos, '(n m) d -> n d', 'mean', m=self.ball_size)
        pos = rearrange(pos, '(n m) d -> n m d', m=self.ball_size)
        rel_pos = rearrange(pos - centers[:, None], 'n m d -> (n m) d')
        return self.pe_proj(rel_pos)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.rpe(pos)
        x = rearrange(self.qkv(x), "(n m) (H E K) -> n H m E K", H=self.num_heads, m=self.ball_size, K=3)
        x = F.scaled_dot_product_attention(
            *[t.squeeze(-1) for t in x.chunk(3, dim=-1)],
            attn_mask=self.create_attention_mask(pos))
        x = rearrange(x, "n H m E -> (n m) (H E)", H=self.num_heads, m=self.ball_size)
        return self.proj(x)


class ErwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 3):
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.BMSA = BallMSA(dim, num_heads, ball_size, dimensionality)
        self.swiglu = SwiGLU(dim, dim * 4)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.BMSA(self.norm1(x), pos)
        return x + self.swiglu(self.norm2(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        direction: Literal['down', 'up', None], # down: encoder, up: decoder, None: bottleneck
        depth: int,
        stride: int,
        dim: int,
        num_heads: int,
        ball_size: int,
        dimensionality: int = 3,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([ErwinTransformerBlock(dim, num_heads, ball_size, dimensionality) for _ in range(depth)])

        self.pool = lambda node: node
        self.unpool = lambda node: node

        if direction == 'down' and stride is not None:
            self.pool = BallPooling(dim, stride, dimensionality)
        elif direction == 'up' and stride is not None:
            self.unpool = BallUnpooling(dim, stride, dimensionality)

    def forward(self, node: Node) -> Node:
        node = self.unpool(node)
        for blk in self.blocks:
            node.x = blk(node.x, node.pos)
        return self.pool(node)


class ErwinTransformer(nn.Module):
    """ 
    Erwin Transformer.

    Args:
        c_in (int): number of input channels.
        c_hidden (int): number of hidden channels. With every layer, the number of channels is multiplied by stride.
        ball_size (List): list of ball sizes for each encoder layer (reverse for decoder).
        enc_num_heads (List): list of number of heads for each encoder layer.
        enc_depths (List): list of number of ErwinTransformerBlock layers for each encoder layer.
        dec_num_heads (List): list of number of heads for each decoder layer.
        dec_depths (List): list of number of ErwinTransformerBlock layers for each decoder layer.
        strides (List): list of strides for each encoder layer (reverse for decoder).
        decode (bool): whether to decode or not. If not, returns latent representation at the coarsest level.
        dimensionality (int): dimensionality of the input data.
        mp_steps (int): number of message passing steps in the MPNN Embedding.

    Notes:
        - lengths of ball_size, enc_num_heads, enc_depths must be the same N (as it includes encoder and bottleneck).
        - lengths of strides, dec_num_heads, dec_depths must be N - 1.
    """
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        ball_size: List,
        enc_num_heads: List,
        enc_depths: List,
        dec_num_heads: List,
        dec_depths: List,
        strides: List,
        decode: bool = True,
        dimensionality: int = 3,
        mp_steps: int = 3,
    ):
        super().__init__()
        assert len(enc_num_heads) == len(enc_depths) == len(ball_size)
        assert len(dec_num_heads) == len(dec_depths) == len(strides)
        assert len(strides) == len(ball_size) - 1
        
        self.decode = decode
        self.embed = ErwinEmbedding(c_in, c_hidden, mp_steps, dimensionality)

        strides = strides + [1] # 1 for bottleneck
        num_layers = len(enc_depths) - 1 # last one is a bottleneck
        num_hidden = [c_hidden] + [c_hidden * 2**i for i in range(1, num_layers+1)]
        
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(
                BasicLayer(
                    direction='down',
                    depth=enc_depths[i],
                    stride=strides[i],
                    dim=num_hidden[i],
                    num_heads=enc_num_heads[i],
                    ball_size=ball_size[i],
                    dimensionality=dimensionality,
                )
            )

        self.bottleneck = BasicLayer(
            direction=None,
            depth=enc_depths[-1],
            stride=None,
            dim=num_hidden[-1],
            num_heads=enc_num_heads[-1],
            ball_size=ball_size[-1],
            dimensionality=dimensionality,
        )

        if decode:
            self.decoder = nn.ModuleList()
            for i in range(num_layers - 1, -1, -1):
                self.decoder.append(
                    BasicLayer(
                        direction='up',
                        depth=dec_depths[i],
                        stride=strides[i],
                        dim=num_hidden[i],
                        num_heads=dec_num_heads[i],
                        ball_size=ball_size[i],
                        dimensionality=dimensionality,
                    )
                )

        self.in_dim = c_in
        self.out_dim = c_hidden if decode else num_hidden[-1]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0., std=0.02, a=-2., b=2.)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, node_features: torch.Tensor, node_positions: torch.Tensor, batch_idx: torch.Tensor, edge_index=None, tree_idx=None, tree_mask=None, radius=None, **kwargs):
        with torch.no_grad():
            # if not given, build the ball tree and radius graph
            if tree_idx is None and tree_mask is None:
                tree_idx, tree_mask = build_balltree_torch(node_positions, batch_idx)
            if edge_index is None:
                assert radius is not None, "radius (float) must be provided if edge_index is not given to build radius graph"
                edge_index = torch_cluster.radius_graph(node_positions, radius, batch=batch_idx, loop=True)

        x = self.embed(node_features, node_positions, edge_index)

        node = Node(
            x=x[tree_idx],
            pos=node_positions[tree_idx],
            batch_idx=batch_idx[tree_idx],
        )

        for layer in self.encoder:
            node = layer(node)

        node = self.bottleneck(node)

        if self.decode:
            for layer in self.decoder:
                node = layer(node)
            return node.x[tree_mask][torch.argsort(tree_idx[tree_mask])]

        return node.x, node.batch_idx

"""Cross-attention with the context K/V computed once per observation.

get_action runs N sampling steps over the same context_tokens. nn.MultiheadAttention
re-projects that context into K and V on every step; here those two projections are
done once and only the query side runs per step.
"""
import torch
import torch.nn.functional as F
from torch import nn


def split_proj(mha: nn.MultiheadAttention, context: torch.Tensor):
    E = mha.embed_dim
    W, b = mha.in_proj_weight, mha.in_proj_bias
    k = F.linear(context, W[E:2 * E], b[E:2 * E])
    v = F.linear(context, W[2 * E:], b[2 * E:])
    B, L, _ = k.shape
    H, D = mha.num_heads, E // mha.num_heads
    k = k.view(B, L, H, D).transpose(1, 2).contiguous()
    v = v.view(B, L, H, D).transpose(1, 2).contiguous()
    return k, v


def attend(mha: nn.MultiheadAttention, x: torch.Tensor, kv):
    E = mha.embed_dim
    W, b = mha.in_proj_weight, mha.in_proj_bias
    q = F.linear(x, W[:E], b[:E])
    B, L, _ = q.shape
    H, D = mha.num_heads, E // mha.num_heads
    q = q.view(B, L, H, D).transpose(1, 2)
    k, v = kv
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(B, L, E)
    return mha.out_proj(out)

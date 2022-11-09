import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from torch.nn.functional import linear
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        assert flash_attn_unpadded_func is not None, "FlashAttention is not installed."
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)

    def attention(
        self,
        q, k, v,
        batch_size=1,
        seqlen=77,
        key_padding_mask=None,
        softmax_scale=None,
        attention_dropout=0.0,
        causal=False,
        cu_seqlens=None,
        max_s=None,
        need_weights=False
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q,k,v: The tensor containing the query, key, and value. each of (B*S, H, D)
            key_padding_mask: a bool tensor of shape (B, S)

        """
        assert not need_weights
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda

        if cu_seqlens is None:
            max_s = seqlen
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                    device=q.device)
            output = flash_attn_unpadded_func(
                q, k, v, cu_seqlens, cu_seqlens, max_s, max_s, attention_dropout,
                softmax_scale=softmax_scale, causal=causal
            )

        return output

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # set up shape vars
        seqlen, batch_size, embed_dim = query.shape
        
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {self.num_heads}"

        # in-projection
        q, k, v = linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view((batch_size * seqlen, self.num_heads, head_dim))
        k = k.contiguous().view((batch_size * seqlen, self.num_heads, head_dim))
        v = v.contiguous().view((batch_size * seqlen, self.num_heads, head_dim))
        
        # flash attention
        attn_output = self.attention(q, k, v, batch_size, seqlen)

        # out-projection
        attn_output = attn_output.contiguous().view(seqlen * batch_size, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(seqlen, batch_size, embed_dim)

        return attn_output, None
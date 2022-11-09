import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from einops import rearrange
from torch.nn.functional import linear
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:
    flash_attn_unpadded_qkvpacked_func = None


def standard_attention_forward(qkv):
    """Implements the standard attention
    Arguments
    ---------
        qkv: (B, S, 3, H, D)
    """
    import torch.nn.functional as F
    qkv = qkv.permute(2, 0, 1, 3, 4)
    # q,k,v: `(batch_size, seqlen, num_heads, head_dim)`
    q, k, v = qkv[0], qkv[1], qkv[2]
    batch_size, seqlen, num_heads, head_dim = q.shape
    q = q.transpose(1, 2).contiguous().view(batch_size*num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).contiguous().view(batch_size*num_heads, seqlen, head_dim)
    v = v.transpose(1, 2).contiguous().view(batch_size*num_heads, seqlen, head_dim)

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)

    return attn_output.transpose(0, 1), None


def flash_attention_forward(
    qkv,
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
        qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
            if unpadded: (nnz, 3, h, d)
        key_padding_mask: a bool tensor of shape (B, S)

    """
    assert not need_weights
    assert qkv.dtype in [torch.float16, torch.bfloat16]
    assert qkv.is_cuda

    if cu_seqlens is None:
        batch_size = qkv.shape[0]
        seqlen = qkv.shape[1]
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = seqlen
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                device=qkv.device)
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_seqlens, max_s, attention_dropout,
            softmax_scale=softmax_scale, causal=causal
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)

    return output, None


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        assert flash_attn_unpadded_qkvpacked_func is not None, "FlashAttention is not installed."
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)

    """ Flash code
    def _in_projection_packed(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
    ):
        E = q.size(-1)
        if k is v:
            if q is k:
                # self-attention
                return linear(q, w, b).chunk(3, dim=-1)
            else:
                # encoder-decoder attention
                w_q, w_kv = w.split([E, E * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([E, E * 2])
                return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        # q, k, v has the same shape
        seqlen, batch_size, embed_dim = query.shape
        scale = 1 / math.sqrt(embed_dim)
        
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")

        # compute in-projection: flash_attention needs qkv packed
        # qkv: `(seqlen, batch_size, embed_dim * 3)`
        qkv = linear(query, self.in_proj_weight, self.in_proj_bias) # CHECKED

        # TODO: prep attention mask
        # TODO: merge key padding and attention masks
        
        # reshape qkv for multihead attention
        # qkv: `(batch_size, seqlen, 3, num_heads, head_dim)`
        qkv = qkv.contiguous().view(seqlen, batch_size, embed_dim, 3)
        qkv = qkv.permute((1, 0, 3, 2))
        qkv = qkv.contiguous().view(batch_size, seqlen, 3, self.num_heads, self.head_dim) # CHECKED

        # applying flash-attention
        # output: `(batch_size, seqlen, num_heads, head_dim)`
        attn_output, _ = flash_attention_forward(
            qkv=qkv,
            key_padding_mask=key_padding_mask,
            softmax_scale=scale,
            need_weights=need_weights,
        )
        # attn_output_, _ = standard_attention_forward(
        #     qkv=qkv,
        # )

        # reshape attn_output and compute out_proj
        attn_output = attn_output.contiguous().view(batch_size*seqlen, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(seqlen, batch_size, embed_dim)

        # no return attn_weights
        return attn_output, None
    """

    # """ Pytorch code
    def _in_projection_packed(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
    ):
        E = q.size(-1)
        if k is v:
            if q is k:
                # self-attention
                return linear(q, w, b).chunk(3, dim=-1)
            else:
                # encoder-decoder attention
                w_q, w_kv = w.split([E, E * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([E, E * 2])
                return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
        
    def standard_to_flash_attention_forward(self, q, k, v, bsz, n_head):    
        """Implements the standard attention
        Arguments
        ---------
            q, k, v: (batch_size, seqlen, hidden)
        """
        seqlen, head_dim = q.shape[1], q.shape[2]
        q = q.transpose(0, 1).contiguous().view(seqlen, bsz, n_head, head_dim).transpose(0, 1)
        k = k.transpose(0, 1).contiguous().view(seqlen, bsz, n_head, head_dim).transpose(0, 1)
        v = v.transpose(0, 1).contiguous().view(seqlen, bsz, n_head, head_dim).transpose(0, 1)
        qkv = torch.stack((q, k, v), dim=-1)
        qkv = qkv.permute((0, 1, 4, 2, 3))
        attn_output = flash_attention_forward(qkv=qkv)
        return attn_output[0].transpose(0, 1)
        

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None, need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        tens_ops = (query, key, value, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.out_proj.weight, self.out_proj.bias)
        import torch.nn.functional as F
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {self.num_heads}"

        #
        # compute in-projection
        #
        q, k, v = self._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.size(1)

        # adjust dropout probability
        dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        # B, Nt, E = q.shape
        # q_scaled = q / math.sqrt(E)
        # if attn_mask is not None:
        #     attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        # else:
        #     attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        # attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        # if dropout_p > 0.0:
        #     attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

        # attn_output = torch.bmm(attn_output_weights, v)

        attn_output = self.standard_to_flash_attention_forward(q, k, v, bsz, self.num_heads)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        return attn_output, None
    # """
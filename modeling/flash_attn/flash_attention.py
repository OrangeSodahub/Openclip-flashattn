import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from einops import rearrange
from torch.nn.functional import linear
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
except ImportError:
    flash_attn_unpadded_qkvpacked_func, unpad_input, pad_input = None, None, None


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
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                        device=qkv.device)
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, cu_seqlens, max_s, attention_dropout,
                    softmax_scale=softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, attention_dropout,
                    softmax_scale=softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                            indices, batch_size, seqlen),
                                'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_seqlens, max_s, attention_dropout,
                softmax_scale=softmax_scale, causal=causal
            )

        return output, None


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        assert flash_attn_unpadded_qkvpacked_func is not None, "FlashAttention is not installed."
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)

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

        # reshape attn_output and compute out_proj
        attn_output = attn_output.contiguous().view(batch_size*seqlen, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(seqlen, batch_size, embed_dim)

        # no return attn_weights
        return attn_output, None

    """ Pytorch code
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

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None, need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        tens_ops = (query, key, value, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.out_proj.weight, self.out_proj.bias)
        import torch.nn.functional as F
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
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

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # adjust dropout probability
        dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        return attn_output
    """
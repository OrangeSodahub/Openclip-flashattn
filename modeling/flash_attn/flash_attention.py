import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from einops import rearrange

from torch.nn.functional import linear
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input


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

        :param qkv: `(3 * embed_dim, embed_dim)`
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
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)

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
        
        # q, k, v has the same shape
        batch_size, seqlen, embed_dim = query.shape
        scale = 1 / math.sqrt(embed_dim)
        
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")

        # compute in-projection: flash_attention needs qkv packed
        # qkv: `(batch_size, seqlen, embed_dim * 3)`
        qkv = linear(query, self.in_proj_weight, self.in_proj_bias)

        # TODO: prep attention mask
        
        # reshape qkv for multihead attention and make em batch first
        # qkv: `(batch_size, seqlen, 3, num_heads, head_dim)`
        qkv = qkv.contiguous().view(batch_size, seqlen, embed_dim, 3)
        qkv = qkv.permute((0, 1, 3, 2))
        qkv = qkv.contiguous().view(batch_size, seqlen, 3, self.num_heads, self.head_dim)

        # applying flash-attention
        attn_output, _ = flash_attention_forward(
            qkv=qkv,
            key_padding_mask=key_padding_mask,
            softmax_scale=scale,
        )

        # reshape attn_output
        attn_output = attn_output.contiguous().view(seqlen, batch_size, embed_dim)
        attn_output.permute((1, 0, 2))

        return attn_output

        # # TODO: merge key padding and attention masks
        # if key_padding_mask is not None:
        #     assert key_padding_mask.shape == (batch_size, seqlen), \
        #         f"expecting key_padding_mask shape of {(batch_size, seqlen)}, but got {key_padding_mask.shape}"
        #     key_padding_mask = key_padding_mask.view(batch_size, 1, 1, seqlen).   \
        #         expand(-1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, 1, seqlen)
        #     if attn_mask is None:
        #         attn_mask = key_padding_mask
        #     elif attn_mask.dtype == torch.bool:
        #         attn_mask = attn_mask.logical_or(key_padding_mask)
        #     else:
        #         attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # # convert mask to float
        # if attn_mask is not None and attn_mask.dtype == torch.bool:
        #     new_attn_mask = torch.zeros_like(attn_mask, dtype=qkv.dtype)
        #     new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        #     attn_mask = new_attn_mask

        # # not training
        # dropout_p = 0.0
        
        # # applying flash-attention
        # B, Nt, E = q.shape
        # q_scaled = q / math.sqrt(E)
        # if attn_mask is not None:
        #     attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        # else:
        #     attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        # attn_output_weights = softmax(attn_output_weights, dim=-1)
        # if dropout_p > 0.0:
        #     attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        # attn_output = torch.bmm(attn_output_weights, v)

        # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        # attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        # attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # if need_weights:
        #     # optionally average attention weights over heads
        #     attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        #     if average_attn_weights:
        #         attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        #     if not is_batched:
        #         # squeeze the output if input was unbatched
        #         attn_output = attn_output.squeeze(1)
        #         attn_output_weights = attn_output_weights.squeeze(0)
        #     return attn_output, attn_output_weights
        # else:
        #     if not is_batched:
        #         # squeeze the output if input was unbatched
        #         attn_output = attn_output.squeeze(1)
        #     return attn_output, None
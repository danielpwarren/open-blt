import logging
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from xformers.ops import AttentionBias

from blt.model.base_transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    cross_entropy,
)
from blt.model.common import create_causal_mask

logger = logging.getLogger()

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm

    RMSNorm = FusedRMSNorm
except (ImportError, ModuleNotFoundError):
    logging.debug("Apex not found. Using nn.RMSNorm")
    RMSNorm = nn.RMSNorm


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class LMTransformerArgs(BaseTransformerArgs):
    seed: int = 42
    vocab_size: int = -1
    sliding_window: int | None = None


class LMTransformer(BaseTransformer):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.sliding_window = args.sliding_window
        assert args.vocab_size > 0
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str | None = None,
    ):
        if attn_impl is None:
            attn_impl = self.attn_impl
        bsz, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)

        mask = (
            mask
            if mask is not None
            else create_causal_mask(
                seqlen,
                attn_impl,
                self.attn_bias_type,
                sliding_window=self.sliding_window,
                tokens=token_values,
                eos_id=self.eos_id,
            )
        )
        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits

    def reset_parameters(self, init_std=None):
        self.norm.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        init_std = self.dim ** (-0.5)
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        super().init_weights()
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

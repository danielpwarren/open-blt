import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias

from blt.model.base_transformer import BaseTransformer, BaseTransformerArgs
from blt.model.common import create_causal_mask


class GlobalTransformer(BaseTransformer):
    """
    GlobalTransformer for the BLT model.
    
    Extends BaseTransformer with token embedding projection capabilities.
    """
    
    def __init__(self, args: BaseTransformerArgs):
        super().__init__(args)
        self.dropout = getattr(args, "dropout", 0.0)
        self.eos_id = args.eos_id
        
        # Token embedding projection
        self.token_embedding_projection = None
        if hasattr(args, "dim_token_emb") and args.dim_token_emb is not None and args.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(
                args.dim_token_emb,
                args.dim,
                bias=False,
            )
    
    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """
        Forward pass through the global transformer.
        
        Args:
            tokens: Input tokens [batch_size, seq_len]
            embeds: Input embeddings [batch_size, seq_len, dim]
            tok_idx: Token indices for position
            mask: Attention mask
            cache: KV cache for inference
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        bs, seqlen = tokens.shape
        
        h = embeds
        
        # Create mask if not provided
        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                self.attn_bias_type,
                tokens=tokens,
                eos_id=self.eos_id,
            )
        
        # Apply token embedding projection if needed
        if self.token_embedding_projection is not None and h.shape[-1] != self.dim:
            h = self.token_embedding_projection(h)
        
        # Apply dropout
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Pass through transformer layers
        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=self.attn_impl)
        
        return h, cache
    
    def init_weights(self):
        """Initialize weights for the global transformer."""
        super().init_weights()
        
        if self.token_embedding_projection is not None:
            std = 0.02
            nn.init.normal_(self.token_embedding_projection.weight, mean=0.0, std=std) 
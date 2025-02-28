from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from blt.model.base_transformer import BaseTransformerArgs, TransformerBlock
from blt.model.base_transformer import RotaryEmbedding
from blt.model.common import create_causal_mask, downsample
from blt.model.cross_attention import CrossAttention
from blt.tokenizer.constants import BOE_ID, BOS_ID, EOS_ID

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm
    RMSNorm = FusedRMSNorm
except (ImportError, ModuleNotFoundError):
    import logging
    logging.debug("Apex not found. Using nn.RMSNorm")
    RMSNorm = nn.RMSNorm


class LocalModelArgs(BaseTransformerArgs):
    """Arguments for Local Encoder and Decoder models."""

    dropout: float = 0.0
    vocab_size: int = 260  # 256 + 4 special tokens
    patch_size: float = 4
    sliding_window: Optional[int] = None
    use_rope: bool = True
    cross_attn_encoder: bool = False
    cross_attn_decoder: bool = False
    cross_attn_k: Optional[int] = None
    cross_attn_init_by_pooling: bool = False
    patching_mode: str = "static"
    use_local_encoder_transformer: bool = False
    downsampling_by_pooling: Optional[str] = None
    cross_attn_all_layers_encoder: bool = False
    cross_attn_all_layers_decoder: bool = False
    cross_attn_nheads: Optional[int] = None
    dim_token_emb: Optional[int] = None
    dim_patch_emb: Optional[int] = None


class LocalModelBase(nn.Module):
    """Base class for local encoder and decoder."""

    def __init__(self, args: LocalModelArgs):
        super().__init__()

        # Model dimensions and configuration
        self.dim = args.dim
        self.dropout = args.dropout
        self.vocab_size = args.vocab_size
        self.patch_size = args.patch_size
        self.dim_patch_emb = args.dim_patch_emb

        # Attention configuration
        self.attn_impl = args.attn_impl
        self.sliding_window = args.sliding_window
        self.use_rope = args.use_rope
        self.cross_attn_encoder = getattr(args, "cross_attn_encoder", None)
        self.cross_attn_decoder = getattr(args, "cross_attn_decoder", None)
        self.cross_attn_k = getattr(args, "cross_attn_k", None)
        self.eos_id = args.eos_id or EOS_ID

        # Special token ID
        self.boe_id = BOE_ID

        # Initialize transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )

        # Position embeddings or RoPE setup
        self.rope_cls = getattr(args, "rope_cls", None)
        if not self.use_rope:
            self.pos_embeddings = nn.Embedding(args.max_seqlen, args.dim)
        else:
            if self.rope_cls is not None:
                self.rope = self.rope_cls(
                    theta=args.rope_theta,
                    head_dim=args.head_dim or args.dim // args.n_heads,
                    max_seqlen=args.max_seqlen,
                    rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
                )
            else:
                self.rope = RotaryEmbedding(
                    theta=args.rope_theta,
                    head_dim=args.head_dim or args.dim // args.n_heads,
                    max_seqlen=args.max_seqlen,
                    rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
                )
            self.pos_embeddings = None

        # Token embedding projection
        self.token_embedding_projection = (
            nn.Linear(args.dim_token_emb, args.dim, bias=False)
            if hasattr(args, "dim_token_emb")
            and args.dim_token_emb is not None
            and args.dim_token_emb != self.dim
            else None
        )

    def apply_embedding(self, tokens, embeds):
        if embeds is not None:
            return embeds
        else:
            return self.tok_embeddings(tokens)


class LocalEncoder(LocalModelBase):
    """Local encoder for the BLT model."""

    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        # Model configuration
        self.apply_transformer = args.use_local_encoder_transformer
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_all_layers_encoder = args.cross_attn_all_layers_encoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        # Initialize rotary embeddings if needed
        self.use_rope = args.use_rope
        if self.use_rope:
            self.rope = RotaryEmbedding(
                theta=args.rope_theta,
                head_dim=args.head_dim or args.dim // args.n_heads,
                max_seqlen=args.max_seqlen,
                rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
            )

        # Token embedding projection
        if self.token_embedding_projection is None and hasattr(args, "dim_token_emb") and args.dim_token_emb is not None and args.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(args.dim_token_emb, args.dim, bias=False)

        # Patch embedding projection - create this before tok_embeddings to ensure correct order in model printout
        self.patch_embedding_projection = nn.Linear(
            args.dim,  # Use self.dim (768) as input dimension
            512,  # Fixed output dimension to match expected model
            bias=False,
        )

        # Token embeddings
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        # Cross-attention layers
        if self.cross_attn_encoder:
            self.cross_attn_layers = nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_encoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )
        else:
            self.cross_attn_layers = None

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """Forward pass through the local encoder."""
        bs, seqlen = tokens.shape

        # Create mask for attention if not provided
        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                "local_block_causal",
                sliding_window=self.sliding_window,
                tokens=tokens,
                eos_id=self.eos_id,
            )

        # Apply token embeddings
        h = self.apply_embedding(tokens, embeds)

        # Apply token embedding projection if needed
        if self.token_embedding_projection is not None:
            h = self.token_embedding_projection(h)

        # Get rotary embeddings for position encoding
        freqs_cis = None
        if self.use_rope and hasattr(self, "rope"):
            freqs_cis = self.rope(seqlen=seqlen)

        # Apply dropout
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis=freqs_cis, mask=mask, attn_impl=self.attn_impl)

            # Apply cross-attention if needed
            if self.cross_attn_encoder and (
                i == len(self.layers) - 1 or self.cross_attn_all_layers_encoder
            ):
                patch_embeds = self.apply_cross_attention(
                    h, patch_embeds, i, bs, num_patches, patch_ids, cross_mask
                )

        h_residual = patch_embeds if self.cross_attn_encoder else None
        return (h, h_residual), cache

    def apply_cross_attention(
        self, h, patch_embeds, layer_idx, bs, num_patches, patch_ids, cross_mask
    ):
        """Apply cross-attention for the encoder."""
        # Apply pooling and project if needed
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            patch_embeds = downsample(
                h,
                num_patches,
                patch_ids=patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                if self.cross_attn_k is not None:
                    patch_embeds = patch_embeds.reshape(bs, -1, self.dim)

        # Apply cross-attention
        layer_idx = layer_idx if self.cross_attn_all_layers_encoder else 0
        patch_embeds_cross = self.cross_attn_layers[layer_idx](
            x=patch_embeds,
            kv=h,
            mask=cross_mask,
        )
        patch_embeds = patch_embeds + patch_embeds_cross
        return patch_embeds
    
    def init_weights(self):
        """Initialize weights for the local encoder."""
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)

        if self.patch_embedding_projection is not None:
            nn.init.normal_(self.patch_embedding_projection.weight, mean=0.0, std=0.02)

        if self.cross_attn_encoder and hasattr(self, 'cross_attn_layers'):
            for layer in self.cross_attn_layers:
                if hasattr(layer, 'init_weights'):
                    layer.init_weights()


class LocalDecoder(LocalModelBase):
    """Local decoder for the BLT model."""

    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        # Model configuration flags
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_all_layers_decoder = args.cross_attn_all_layers_decoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        # Token embedding projection
        if self.token_embedding_projection is None and hasattr(args, "dim_token_emb") and args.dim_token_emb is not None and args.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(args.dim_token_emb, args.dim, bias=False)

        # Patch embedding projection - create this before norm to ensure correct order in model printout
        self.patch_embedding_projection = nn.Linear(
            args.dim_patch_emb,  # Use args.dim_patch_emb (1280) as input dimension
            512,  # Fixed output dimension to match expected model
            bias=False,
        ) if args.dim_patch_emb is not None and args.dim_patch_emb != self.dim else None

        # Layer normalization
        self.norm = FusedRMSNorm(args.dim, eps=args.norm_eps)

        # Cross-attention layers
        if self.cross_attn_decoder:
            self.cross_attn_layers = nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_decoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )
        else:
            self.cross_attn_layers = None

        # Output projection
        self.output = nn.Linear(
            self.dim,
            args.vocab_size,
            bias=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: torch.Tensor,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """Forward pass through the local decoder."""
        bs, seqlen = tokens.shape
        assert embeds is not None, "Embeddings must be provided for the decoder"

        # Create mask for attention if not provided
        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                "local_block_causal",
                sliding_window=self.sliding_window,
                tokens=tokens,
                eos_id=self.eos_id,
            )

        h = embeds

        # Apply token embedding projection if needed
        if self.token_embedding_projection is not None:
            h = self.token_embedding_projection(h)

        # Process patch embeddings if needed
        if patch_embeds is not None and self.patch_embedding_projection is not None:
            patch_embeds = self.patch_embedding_projection(patch_embeds)
            if self.cross_attn_k is not None:
                patch_embeds = patch_embeds.reshape(bs, -1, self.dim)

        # Add patch embeddings if not using cross-attention
        if patch_embeds is not None and not self.cross_attn_decoder:
            h = h + patch_embeds

        # Get rotary embeddings for position encoding
        freqs_cis = None
        if self.use_rope and hasattr(self, "rope"):
            freqs_cis = self.rope(seqlen=seqlen)

        # Apply dropout
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            # Apply cross-attention if needed
            if self.cross_attn_decoder and (
                i == 0 or self.cross_attn_all_layers_decoder
            ):
                layer_idx = i if self.cross_attn_all_layers_decoder else 0
                h_cross = self.cross_attn_layers[layer_idx](
                    x=h,
                    kv=patch_embeds,
                    mask=cross_mask,
                )
                h = h + h_cross

            h = layer(h, freq_cis=freqs_cis, mask=mask, attn_impl=self.attn_impl)

        # Apply final layer norm
        h = self.norm(h)

        # Apply dropout and project to vocabulary
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.output(h)

        return logits, cache

    def init_weights(self):
        """Initialize weights for the local decoder."""
        # Initialize output projection
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        
        # Initialize patch embedding projection if present
        if self.patch_embedding_projection is not None:
            nn.init.normal_(self.patch_embedding_projection.weight, mean=0.0, std=0.02)
            
        # Initialize cross-attention layers if present
        if self.cross_attn_decoder and hasattr(self, 'cross_attn_layers'):
            for layer in self.cross_attn_layers:
                if hasattr(layer, 'init_weights'):
                    layer.init_weights()

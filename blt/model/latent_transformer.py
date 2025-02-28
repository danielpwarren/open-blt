from typing import Optional

import torch
import torch.nn as nn

from blt.model.base_transformer import BaseTransformer, BaseTransformerArgs
from blt.model.common import cross_attn_mask, downsample, patch_ids_from_lengths
from blt.model.local_models import LocalDecoder, LocalEncoder, LocalModelArgs
from blt.tokenizer.constants import BOE_ID, BOS_ID, EOS_ID, PAD_ID


class ByteLatentTransformerArgs(BaseTransformerArgs):
    """Arguments for the BLT model."""

    # Basic model configuration
    vocab_size: int = 260  # 256 + 4 special tokens
    dim: int = 512
    n_heads: int = 8

    # Architecture and dimensions
    dim_token: Optional[int] = None
    dim_global: int = 512
    dim_local_decoder: int = 512
    dim_local_encoder: int = 512
    n_layers_global: int = 8
    n_layers_local_decoder: int = 4
    n_layers_local_encoder: int = 4

    # Tokenization and patching
    patch_size: float = 4
    patching_mode: str = "static"
    patching_threshold: Optional[float] = None
    max_patch_length: Optional[int] = None

    # Encoder/Decoder configuration
    tie_local_encoder_decoder_logits: bool = False
    use_local_encoder_transformer: bool = False
    max_encoder_seq_length: Optional[int] = None

    # Cross attention configurations
    cross_attn_encoder: bool = False
    cross_attn_decoder: bool = False
    cross_attn_window_encoder: Optional[int] = None
    cross_attn_window_decoder: Optional[int] = None
    cross_attn_k: Optional[int] = None
    cross_attn_nheads: Optional[int] = 4
    cross_attn_all_layers_decoder: bool = False
    cross_attn_all_layers_encoder: bool = False
    cross_attn_init_by_pooling: bool = False

    # Model behavior and optimization
    downsampling_by_pooling: Optional[str] = None
    use_rope: bool = True
    dropout: float = 0.0

    # Additional configurations
    share_encoder_decoder_emb: bool = True


def get_blt_input(
    tokens: torch.Tensor,
    nb_boe: int,
    boe_id: int,
):
    """
    Prepare input tokens for BLT model.

    Args:
        tokens: Input tokens [batch_size, seq_len]
        nb_boe: Number of BOE tokens to add
        boe_id: BOE token ID

    Returns:
        Tuple of (local_encoder_tokens, local_decoder_tokens)
    """
    batch_size, seq_len = tokens.shape
    local_encoder_tokens = local_decoder_tokens = tokens

    # Add BOE tokens to the beginning of the sequence for encoder
    if nb_boe > 0:
        padded_patch = tokens.new_full((batch_size, nb_boe), boe_id)
        local_encoder_tokens = torch.cat((padded_patch, local_encoder_tokens), dim=1)

    return local_encoder_tokens, local_decoder_tokens


def decoder_patch_ids_from_lengths(patch_lengths, nb_boe, seq_len):
    """
    Generate decoder patch IDs from patch lengths.

    Args:
        patch_lengths: Patch lengths [batch_size, num_patches]
        nb_boe: Number of BOE tokens
        seq_len: Sequence length

    Returns:
        Decoder patch IDs [batch_size, seq_len]
    """
    # Skip the first patch (which contains BOE tokens)
    if nb_boe > 0:
        decoder_patch_lengths = patch_lengths[:, 1:]
    else:
        decoder_patch_lengths = patch_lengths

    # Generate patch IDs for decoder
    decoder_patch_ids = patch_ids_from_lengths(
        patch_lengths=decoder_patch_lengths, seq_len=seq_len
    )

    return decoder_patch_ids


class ByteLatentTransformer(nn.Module):
    """
    ByteLatentTransformer (BLT) model.

    The BLT model consists of three main components:
    1. Local encoder: Processes input bytes at the token level
    2. Global transformer: Processes patch-level representations
    3. Local decoder: Generates output bytes
    """

    def __init__(self, args: ByteLatentTransformerArgs):
        super().__init__()

        # General configuration
        self.patch_size = args.patch_size
        self.patching_mode = args.patching_mode
        self.boe_id, self.bos_id, self.eos_id, self.pad_id = (
            BOE_ID,
            BOS_ID,
            EOS_ID,
            PAD_ID,
        )
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.patching_threshold = args.patching_threshold

        # Cross attention configuration
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_k = args.cross_attn_k
        self.cross_attn_window_encoder = args.cross_attn_window_encoder
        self.cross_attn_window_decoder = args.cross_attn_window_decoder

        # Create LocalEncoder args
        encoder_args = LocalModelArgs(
            dim=args.dim_local_encoder,
            n_layers=args.n_layers_local_encoder,
            n_heads=args.n_heads,
            head_dim=getattr(args, "head_dim", None),
            dim_token_emb=args.dim_token,
            dim_patch_emb=args.dim_global if args.cross_attn_encoder else None,
            dropout=args.dropout,
            vocab_size=args.vocab_size,
            patch_size=args.patch_size,
            sliding_window=getattr(args, "local_attention_window_len", None),
            use_rope=args.use_rope,
            cross_attn_encoder=args.cross_attn_encoder,
            cross_attn_decoder=False,
            cross_attn_k=args.cross_attn_k,
            cross_attn_nheads=args.cross_attn_nheads,
            cross_attn_all_layers_encoder=args.cross_attn_all_layers_encoder,
            cross_attn_init_by_pooling=args.cross_attn_init_by_pooling,
            patching_mode=args.patching_mode,
            use_local_encoder_transformer=args.use_local_encoder_transformer,
            downsampling_by_pooling=args.downsampling_by_pooling,
            max_seqlen=args.max_encoder_seq_length or 4096,
            attn_impl=getattr(args, "attn_impl", "sdpa"),
            rope_theta=getattr(args, "rope_theta", 10000.0),
            rope_use_fp32_in_outer_product=getattr(
                args, "rope_use_fp32_in_outer_product", False
            ),
            norm_eps=getattr(args, "norm_eps", 1e-5),
            eos_id=EOS_ID,
        )

        # Create GlobalTransformer args
        global_args = BaseTransformerArgs(
            dim=args.dim_global,
            n_layers=args.n_layers_global,
            n_heads=args.n_heads,
            head_dim=getattr(args, "head_dim", None),
            max_seqlen=args.max_encoder_seq_length or 4096,
            attn_impl=getattr(args, "attn_impl", "sdpa"),
            rope_theta=getattr(args, "rope_theta", 10000.0),
            rope_use_fp32_in_outer_product=getattr(
                args, "rope_use_fp32_in_outer_product", False
            ),
            norm_eps=getattr(args, "norm_eps", 1e-5),
            eos_id=EOS_ID,
        )

        # Create LocalDecoder args
        decoder_args = LocalModelArgs(
            dim=args.dim_local_decoder,
            n_layers=args.n_layers_local_decoder,
            n_heads=args.n_heads,
            head_dim=getattr(args, "head_dim", None),
            dim_token_emb=(
                args.dim_local_encoder if args.dim_token is None else args.dim_token
            ),
            dim_patch_emb=args.dim_global,
            dropout=args.dropout,
            vocab_size=args.vocab_size,
            patch_size=args.patch_size,
            sliding_window=getattr(args, "local_attention_window_len", None),
            use_rope=args.use_rope,
            cross_attn_encoder=False,
            cross_attn_decoder=args.cross_attn_decoder,
            cross_attn_k=args.cross_attn_k,
            cross_attn_nheads=args.cross_attn_nheads,
            cross_attn_all_layers_decoder=args.cross_attn_all_layers_decoder,
            cross_attn_init_by_pooling=False,
            patching_mode=args.patching_mode,
            max_seqlen=args.max_encoder_seq_length or 4096,
            attn_impl=getattr(args, "attn_impl", "sdpa"),
            rope_theta=getattr(args, "rope_theta", 10000.0),
            rope_use_fp32_in_outer_product=getattr(
                args, "rope_use_fp32_in_outer_product", False
            ),
            norm_eps=getattr(args, "norm_eps", 1e-5),
            eos_id=EOS_ID,
        )

        # Initialize model components
        self.local_encoder = LocalEncoder(encoder_args)
        self.global_transformer = BaseTransformer(global_args)
        self.local_decoder = LocalDecoder(decoder_args)

        # Tie encoder and decoder embeddings if specified
        if args.share_encoder_decoder_emb:
            self.local_decoder.tok_embeddings = self.local_encoder.tok_embeddings

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lengths: torch.Tensor,
    ):
        """
        Forward pass through the BLT model.

        Args:
            tokens: Input tokens [batch_size, seq_len]
            patch_lengths: Patch lengths [batch_size, num_patches]

        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        bs, seq_len = tokens.shape

        # Calculate number of BOE tokens based on patching mode
        nb_boe = int(self.patch_size - 1 if self.patching_mode == "static" else 0)

        # Prepare input tokens
        local_encoder_tokens, local_decoder_tokens = get_blt_input(
            tokens=tokens,
            nb_boe=nb_boe,
            boe_id=self.boe_id,
        )

        # Generate patch IDs from patch_lengths
        encoder_patch_ids = patch_ids_from_lengths(
            patch_lengths, local_encoder_tokens.shape[-1]
        )

        # Create cross-attention mask for encoder if needed
        cross_attn_mask_enc = None
        if self.cross_attn_encoder:
            cross_attn_mask_enc = cross_attn_mask(
                encoder_patch_ids,
                patch_lengths,
                local_encoder_tokens.shape[-1],
                patches_as_queries=True,
                cross_attn_k=self.cross_attn_k,
                window=self.cross_attn_window_encoder,
            )

        # Local encoder
        (h_encoder, h_cross), _ = self.local_encoder(
            tokens=local_encoder_tokens,
            embeds=None,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=encoder_patch_ids,
        )

        # Downsampling for global transformer
        if not self.cross_attn_encoder:
            h = downsample(
                h_encoder,
                patch_lengths.shape[1],
                patch_lengths=patch_lengths,
                patch_ids=encoder_patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
        else:
            h = h_cross

        # Global transformer
        # Create token IDs for global transformer with EOS positions
        global_tokens = torch.full_like(h[:, :, 0], self.boe_id, dtype=torch.long)
        rows, cols = torch.where(local_encoder_tokens == self.eos_id)
        if len(rows) > 0:
            eos_patch_ids = encoder_patch_ids[rows, cols]
            for i, (row, pid) in enumerate(zip(rows, eos_patch_ids)):
                if pid < global_tokens.shape[1]:
                    global_tokens[row, pid] = self.eos_id

        h, _ = self.global_transformer(
            h,
            mask="causal",
        )

        # Decoder input
        # Skip BOE tokens in encoder output
        dec_embeds = h_encoder[:, nb_boe : nb_boe + seq_len, :]

        # Generate decoder patch IDs
        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, nb_boe, seq_len
        )

        # Cross-attention mask for decoder if needed
        cross_attn_mask_dec = None
        if self.cross_attn_decoder:
            cross_attn_mask_dec = cross_attn_mask(
                decoder_patch_ids,
                patch_lengths,
                seq_len,
                patches_as_queries=False,
                cross_attn_k=self.cross_attn_k,
                window=self.cross_attn_window_decoder,
            )
        else:
            # If not using cross-attention, gather patch embeddings for each token
            h_reshaped = h.reshape(bs, -1, h.shape[-1])
            h = torch.gather(
                h_reshaped,
                1,
                decoder_patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1]),
            )

        # Local decoder
        output, _ = self.local_decoder(
            tokens=local_decoder_tokens,
            embeds=dec_embeds,
            patch_embeds=h,
            cross_mask=cross_attn_mask_dec,
        )

        return output

    def init_weights(self):
        """Initialize model weights."""
        self.local_encoder.init_weights()
        self.global_transformer.init_weights()
        self.local_decoder.init_weights()

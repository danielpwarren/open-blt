from typing import Optional

import torch
import torch.nn as nn

from blt.model.base_transformer import BaseTransformerArgs
from blt.model.common import cross_attn_mask, downsample, patch_ids_from_lengths
from blt.model.global_transformer import GlobalTransformer
from blt.model.local_models import LocalDecoder, LocalEncoder, LocalModelArgs
from blt.tokenizer.constants import BOE_ID, BOS_ID, EOS_ID, PAD_ID


class ByteLatentTransformerArgs(BaseTransformerArgs):
    """Arguments for the BLT model."""

    # Basic model configuration
    vocab_size: int = 260  # 256 + 4 special tokens
    n_heads_global: int = 10
    n_heads_local_encoder: int = 12
    n_heads_local_decoder: int = 12

    # Architecture and dimensions
    dim_token: Optional[int] = None
    dim_global: int = 1280
    dim_local_decoder: int = 768
    dim_local_encoder: int = 768
    n_layers_global: int = 24
    n_layers_local_decoder: int = 7
    n_layers_local_encoder: int = 1

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
    cross_attn_encoder: bool = True
    cross_attn_decoder: bool = True
    cross_attn_window_encoder: Optional[int] = None
    cross_attn_window_decoder: Optional[int] = None
    cross_attn_k: Optional[int] = 2
    cross_attn_nheads: Optional[int] = 10
    cross_attn_all_layers_decoder: bool = True
    cross_attn_all_layers_encoder: bool = True
    cross_attn_init_by_pooling: bool = True

    # Model behavior and optimization
    downsampling_by_pooling: Optional[str] = None
    use_rope: bool = True
    dropout: float = 0.0

    # Additional configurations
    share_encoder_decoder_emb: bool = True
    
    # Hash embedding configurations
    encoder_hash_byte_group_size: Optional[int] = None
    encoder_hash_byte_group_vocab: int = 50002
    encoder_hash_byte_group_nb_functions: int = 3


def get_blt_input(
    tokens: torch.Tensor,
    nb_boe: int,
    boe_id: int,
):
    """
    Prepare input tokens for the BLT model.
    
    Args:
        tokens: Input tokens [batch_size, seq_len]
        nb_boe: Number of BOE tokens to prepend
        boe_id: BOE token ID
        
    Returns:
        Tuple of (encoder_tokens, decoder_tokens)
    """
    batch_size, seq_len = tokens.shape
    local_encoder_tokens = tokens
    local_decoder_tokens = tokens

    if nb_boe > 0:
        padded_patch = tokens.new(batch_size, nb_boe).fill_(boe_id)
        local_encoder_tokens = torch.cat((padded_patch, local_encoder_tokens), dim=1)

    return local_encoder_tokens, local_decoder_tokens


def decoder_patch_ids_from_lengths(
    patch_lengths: torch.Tensor, nb_boe: int, seq_len: int
):
    """
    Generate patch IDs for the decoder from patch lengths.
    
    Args:
        patch_lengths: Patch lengths [batch_size, num_patches]
        nb_boe: Number of BOE tokens
        seq_len: Sequence length
        
    Returns:
        Decoder patch IDs [batch_size, seq_len]
    """
    first_patch_length = patch_lengths[:, 0].clone()
    decoder_patch_lengths = patch_lengths[:, 1:].clone()
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
            n_heads=args.n_heads_local_encoder,
            head_dim=getattr(args, "head_dim", None),
            dim_token_emb=256,  # Token embedding dimension
            dim_patch_emb=args.dim_local_encoder,  # Use local encoder dim instead of global dim
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
            n_heads=args.n_heads_global,
            head_dim=getattr(args, "head_dim", None),
            max_seqlen=args.max_encoder_seq_length or 4096,
            attn_impl=getattr(args, "attn_impl", "sdpa"),
            rope_theta=getattr(args, "rope_theta", 10000.0),
            rope_use_fp32_in_outer_product=getattr(
                args, "rope_use_fp32_in_outer_product", False
            ),
            norm_eps=getattr(args, "norm_eps", 1e-5),
            eos_id=EOS_ID,
            dim_token_emb=512,  # Patch embedding dimension
        )

        # Create LocalDecoder args
        decoder_args = LocalModelArgs(
            dim=args.dim_local_decoder,
            n_layers=args.n_layers_local_decoder,
            n_heads=args.n_heads_local_decoder,
            head_dim=getattr(args, "head_dim", None),
            dim_token_emb=256,  # Token embedding dimension
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
        self.global_transformer = GlobalTransformer(global_args)
        self.local_decoder = LocalDecoder(decoder_args)
        
        # Initialize hash token embeddings
        self.encoder_hash_tok_embedding = nn.ModuleList(
            [nn.Embedding(args.encoder_hash_byte_group_vocab, args.dim_local_encoder) 
             for _ in range(args.encoder_hash_byte_group_nb_functions)]
        )

        # Do not tie encoder and decoder embeddings
        # self.local_decoder.tok_embeddings = self.local_encoder.tok_embeddings

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
            tokens=global_tokens,
            embeds=h,
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
        
        # Initialize hash token embeddings if present
        if self.encoder_hash_tok_embedding is not None:
            for embedding in self.encoder_hash_tok_embedding:
                nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

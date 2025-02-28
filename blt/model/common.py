from typing import Optional

import torch
import torch.nn.functional as F


def patch_ids_from_lengths(patch_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Generate patch IDs from patch lengths.

    Args:
        patch_lengths: Patch lengths [batch_size, num_patches]
        seq_len: Sequence length

    Returns:
        Patch IDs [batch_size, seq_len]
    """
    bs, num_patches = patch_lengths.shape

    # Create cumulative sums of patch lengths
    cum_d = torch.cat(
        [
            torch.zeros(bs, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
            patch_lengths.cumsum(dim=-1),
        ],
        dim=-1,
    )

    # Assign patch IDs to each token position
    patch_ids = (cum_d.unsqueeze(-1) <= torch.arange(seq_len, device=cum_d.device)).sum(
        dim=-2
    ) - 1

    return patch_ids


def attention_flops_per_token(n_layers, seq_len, dim, causal=True):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(num_non_embed_params, n_layers, dim, seq_len):
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def tokens_to_seqlen(batch: torch.Tensor, eos_id: int):
    """
    0 0 0 1 0 0 0 1 0 0 0
    0 1 0 0 0 1 0 0 0 0 0
    -> 4 4 3 2 4 5
    """
    mask = batch == eos_id
    mask[:, -1] = True

    row, col = torch.where(mask)

    seqlens = (col[1:] - col[:-1]) + (row[1:] - row[:-1]) * mask.shape[1]
    return [int(col[0].item() + 1)] + seqlens.tolist()


def create_causal_mask(
    seq_len: int,
    attn_impl: str = "sdpa",
    causal_type: str = "causal",
    sliding_window: Optional[int] = None,
    tokens: Optional[torch.Tensor] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length
        attn_impl: Attention implementation ('sdpa', 'xformers', or 'flex_attention')
        causal_type: Type of causal mask ('causal' or 'local_block_causal')
        sliding_window: Window size for local attention
        tokens: Input tokens
        eos_id: End of sequence token ID

    Returns:
        Attention mask appropriate for the specified attention implementation
    """
    # For xformers, we need to return an AttentionBias object
    if attn_impl == "xformers":
        from xformers.ops.fmha.attn_bias import (
            BlockDiagonalCausalMask,
            LowerTriangularMask,
        )

        if sliding_window is not None and sliding_window < seq_len:
            # Create a block diagonal causal mask for sliding window attention
            # We create blocks of size 'sliding_window' where each token can attend
            # to itself and sliding_window-1 tokens before it
            return BlockDiagonalCausalMask.from_seqlens(
                [sliding_window] * ((seq_len + sliding_window - 1) // sliding_window)
            )
        else:
            # Standard causal mask
            return LowerTriangularMask()

    # Standard causal mask where each token can only attend to previous tokens
    if causal_type == "causal":
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

        if attn_impl == "sdpa":
            # For scaled dot product attention, convert to float and set invalid positions to -inf
            mask = mask.float()
            mask = mask.masked_fill(mask == 1, float("-inf"))

        # Apply sliding window if specified
        if sliding_window is not None:
            # Create a mask that only allows attention within the window
            window_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
            for i in range(seq_len):
                window_mask[i, max(0, i - sliding_window) : i + 1] = False

            # Combine with causal mask
            if attn_impl == "sdpa":
                mask = mask.masked_fill(window_mask, float("-inf"))
            else:
                mask = mask | window_mask

    # Local block causal mask (for patched approaches)
    elif causal_type == "local_block_causal":
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

        # Handle EOS tokens if provided
        if tokens is not None and eos_id is not None:
            # Create mask where tokens after EOS can't be attended to
            bs = tokens.shape[0]
            expanded_mask = mask.unsqueeze(0).expand(bs, seq_len, seq_len)

            # Find positions of EOS tokens
            eos_positions = (tokens == eos_id).nonzero(as_tuple=True)

            # Apply EOS masking
            if len(eos_positions[0]) > 0:
                for b, pos in zip(eos_positions[0], eos_positions[1]):
                    if pos < seq_len - 1:  # Not the last position
                        expanded_mask[b, :, pos + 1 :] = True

            mask = expanded_mask

        if attn_impl == "sdpa":
            if mask.dim() == 2:
                mask = mask.float().masked_fill(mask == 1, float("-inf"))
            else:
                mask = mask.float().masked_fill(mask == 1, float("-inf"))

        # Apply sliding window if specified
        if sliding_window is not None:
            window_mask = torch.ones_like(mask, dtype=torch.bool)

            if mask.dim() == 2:
                for i in range(seq_len):
                    window_mask[i, max(0, i - sliding_window) : i + 1] = False
            else:
                bs = mask.shape[0]
                for b in range(bs):
                    for i in range(seq_len):
                        window_mask[b, i, max(0, i - sliding_window) : i + 1] = False

            if attn_impl == "sdpa":
                mask = mask.masked_fill(window_mask, float("-inf"))
            else:
                mask = mask | window_mask
    else:
        raise ValueError(f"Unsupported causal_type: {causal_type}")

    return mask


def downsample(
    h: torch.Tensor,
    num_patches: int,
    patch_lengths: Optional[torch.Tensor] = None,
    patch_ids: Optional[torch.Tensor] = None,
    downsampling_by_pooling: Optional[str] = None,
    patch_size: float = 4,
) -> torch.Tensor:
    """
    Downsample token embeddings to patch embeddings.

    Args:
        h: Token embeddings [batch_size, seq_len, dim]
        num_patches: Number of patches
        patch_lengths: Patch lengths [batch_size, num_patches]
        patch_ids: Patch IDs [batch_size, seq_len]
        downsampling_by_pooling: Pooling method ('max', 'mean', or None)
        patch_size: Patch size

    Returns:
        Downsampled embeddings [batch_size, num_patches, dim]
    """
    bs, seq_len, dim = h.shape

    if downsampling_by_pooling is None or downsampling_by_pooling.lower() == "none":
        # No pooling, just use first token of each patch
        if patch_ids is None:
            # Static patches of fixed size
            indices = torch.arange(0, seq_len, int(patch_size), device=h.device)
            if len(indices) < num_patches:
                # Pad if needed
                pad_size = num_patches - len(indices)
                indices = F.pad(indices, (0, pad_size), value=indices[-1])
            else:
                # Truncate if needed
                indices = indices[:num_patches]

            # Gather embeddings at indices
            h_patches = h[:, indices, :]
        else:
            # Use first token of each patch based on patch_ids
            h_patches = torch.zeros(bs, num_patches, dim, device=h.device)
            for b in range(bs):
                for p in range(num_patches):
                    # Find positions where patch_ids == p
                    mask = patch_ids[b] == p
                    if mask.any():
                        # Use first position in the patch
                        first_pos = mask.nonzero()[0].item()
                        h_patches[b, p] = h[b, first_pos]
    else:
        # Pool embeddings within each patch
        h_patches = torch.zeros(bs, num_patches, dim, device=h.device)

        if patch_ids is None:
            # Static patches of fixed size
            for b in range(bs):
                for p in range(num_patches):
                    start_idx = p * int(patch_size)
                    end_idx = min(start_idx + int(patch_size), seq_len)

                    if start_idx < seq_len:
                        patch_embs = h[b, start_idx:end_idx]

                        if downsampling_by_pooling.lower() == "max":
                            h_patches[b, p] = torch.max(patch_embs, dim=0)[0]
                        elif downsampling_by_pooling.lower() == "mean":
                            h_patches[b, p] = torch.mean(patch_embs, dim=0)
                        else:
                            raise ValueError(
                                f"Unsupported pooling: {downsampling_by_pooling}"
                            )
        else:
            # Use patch_ids to determine which tokens belong to each patch
            for b in range(bs):
                for p in range(num_patches):
                    # Find positions where patch_ids == p
                    mask = patch_ids[b] == p
                    if mask.any():
                        patch_embs = h[b][mask]

                        if downsampling_by_pooling.lower() == "max":
                            h_patches[b, p] = torch.max(patch_embs, dim=0)[0]
                        elif downsampling_by_pooling.lower() == "mean":
                            h_patches[b, p] = torch.mean(patch_embs, dim=0)
                        else:
                            raise ValueError(
                                f"Unsupported pooling: {downsampling_by_pooling}"
                            )

    return h_patches


def cross_attn_mask(
    patch_ids: torch.Tensor,
    patch_lengths: torch.Tensor,
    seq_len: int,
    patches_as_queries: bool = False,
    cross_attn_k: int = 1,
    window: Optional[int] = None,
) -> torch.Tensor:
    """
    Create cross-attention mask.

    Args:
        patch_ids: Patch IDs [batch_size, seq_len]
        patch_lengths: Patch lengths [batch_size, num_patches]
        seq_len: Sequence length
        patches_as_queries: Whether patches are used as queries
        cross_attn_k: Cross-attention factor
        window: Window size for local attention

    Returns:
        Cross-attention mask
    """
    bs = patch_ids.shape[0]
    num_patches = patch_lengths.shape[1]

    # Create mask where tokens can only attend to their patch
    if patches_as_queries:
        # Patches attending to tokens
        mask = torch.zeros(
            bs,
            num_patches,
            seq_len,
            device=patch_ids.device,
            dtype=torch.bool,
        )
        for b in range(bs):
            for p in range(num_patches):
                mask[b, p] = patch_ids[b] == p
    else:
        # Tokens attending to patches
        mask = torch.zeros(
            bs,
            seq_len,
            num_patches,
            device=patch_ids.device,
            dtype=torch.bool,
        )
        for b in range(bs):
            for i in range(seq_len):
                mask[b, i, patch_ids[b, i]] = True

    # Apply window if specified
    if window is not None:
        if patches_as_queries:
            window_mask = torch.zeros_like(mask)
            for b in range(bs):
                for p in range(num_patches):
                    # Find token positions for this patch
                    patch_tokens = (patch_ids[b] == p).nonzero().squeeze(-1)
                    if len(patch_tokens) > 0:
                        # Allow attention to nearby tokens
                        for t in patch_tokens:
                            window_mask[
                                b, p, max(0, t - window) : min(seq_len, t + window + 1)
                            ] = True
            mask = mask & window_mask
        else:
            window_mask = torch.zeros_like(mask)
            for b in range(bs):
                for i in range(seq_len):
                    p = patch_ids[b, i].item()
                    window_mask[
                        b, i, max(0, p - window) : min(num_patches, p + window + 1)
                    ] = True
            mask = mask & window_mask

    # Repeat for cross_attn_k
    if cross_attn_k > 1:
        if patches_as_queries:
            mask = mask.repeat_interleave(cross_attn_k, dim=1)
        else:
            mask = mask.repeat_interleave(cross_attn_k, dim=2)

    # Convert to attention mask format
    attn_mask = torch.zeros(
        bs,
        1,  # Single head, will be broadcast
        mask.shape[1],
        mask.shape[2],
        device=mask.device,
        dtype=torch.float,
    )
    attn_mask = attn_mask.masked_fill(~mask.unsqueeze(1), float("-inf"))

    return attn_mask


def repeat_kv(
    x: torch.Tensor, n_rep: int, dim: int = 2, mode: str = "general"
) -> torch.Tensor:
    """
    Repeat key/value tensors.

    Args:
        x: Input tensor.
        n_rep: Number of repetitions.
        dim: Dimension along which to repeat.
        mode: "general" uses torch.repeat_interleave, "multiquery" assumes dim==2 and uses an expand/reshape trick.

    Returns:
        Repeated tensor.
    """
    if n_rep == 1:
        return x
    if mode == "general":
        shape = list(x.shape)
        shape[dim] *= n_rep
        return x.repeat_interleave(n_rep, dim=dim)
    elif mode == "multiquery":
        assert dim == 2, "multiquery mode only supports dim==2."
        bs, slen, n_kv_heads, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )
    else:
        raise ValueError(f"Unknown mode {mode}")

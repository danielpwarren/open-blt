import torch
from torch.nn.attention.flex_attention import create_block_mask
from xformers.ops import fmha


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
    mask[:, -1] = True  # virtual eos at the end of each row

    # 0 0 0 1 0 0 0 1 0 0 X
    # 0 1 0 0 0 1 0 0 0 0 X
    row, col = torch.where(mask)

    # row = 0, 0, 0, 1, 1, 1
    # col = 3, 7, 10, 1, 5, 10
    seqlens = (col[1:] - col[:-1]) + (row[1:] - row[:-1]) * mask.shape[1]
    # seqlens = (4, 3, -9, 4, 5) + (0, 0, 11, 0, 0) = (4, 3, 2, 4, 5)
    return [int(col[0].item() + 1)] + seqlens.tolist()


def create_causal_mask(
    seqlen,
    attn_impl: str,
    attn_bias_type: str | None,
    *,
    eos_id: int | None = None,
    tokens: torch.Tensor | None = None,
    sliding_window: int | None = None,
):
    if attn_impl == "xformers":
        if attn_bias_type is None:
            return fmha.attn_bias.LowerTriangularMask()
        elif attn_bias_type == "causal":
            assert sliding_window is None
            return fmha.attn_bias.LowerTriangularMask()
        elif attn_bias_type == "block_causal":
            assert sliding_window is None
            assert eos_id is not None
            assert tokens is not None
            return fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                q_seqlen=tokens_to_seqlen(tokens, eos_id)
            )
        elif attn_bias_type == "local_block_causal":
            assert sliding_window is not None
            assert eos_id is not None
            assert tokens is not None
            return fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                q_seqlen=tokens_to_seqlen(tokens, eos_id)
            ).make_local_attention(sliding_window)
        else:
            return fmha.attn_bias.LocalAttentionFromBottomRightMask(
                window_left=sliding_window - 1, window_right=0
            )
    elif attn_impl == "sdpa":
        if attn_bias_type == "causal":
            return "causal"
        else:
            raise ValueError(
                "SDPA attention being used, which doesn't have specialized attention implementations for block_causal and local_block_causal attention."
            )
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    elif attn_impl == "fmha":
        return None
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )

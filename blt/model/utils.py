# Not yet used in the codebase, but a general idea on how to estimate FLOPs
# which is how Meta's paper estimates the performance of the models vs traditional models
# This is ChatGPT estimated and not accurate for our current model

def attention_flops_per_token(n_layers, seq_len, dim, causal=True):
    """
    Estimate the FLOPs per token for attention.

    Args:
        n_layers (int): Number of transformer layers.
        seq_len (int): Sequence length.
        dim (int): Hidden dimension.
        causal (bool): If True, assume causal (autoregressive) attention (roughly half the cost).

    Returns:
        float: Estimated FLOPs per token for attention.
    """
    factor = 0.5 if causal else 1.0
    # The constant 3.5 is an approximate multiplier to account for additional operations.
    return 3.5 * (4 * n_layers * seq_len * dim * factor)


def get_num_flop_per_token(num_non_embed_params, n_layers, dim, seq_len):
    """
    Estimate the total FLOPs per token (including non-embedding parameters and attention).

    Args:
        num_non_embed_params (int): Total number of non-embedding parameters.
        n_layers (int): Number of transformer layers.
        dim (int): Hidden dimension.
        seq_len (int): Sequence length.

    Returns:
        float: Estimated total FLOPs per token.
    """
    return 6 * num_non_embed_params + attention_flops_per_token(n_layers, seq_len, dim, causal=True)

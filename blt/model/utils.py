def attention_flops_per_token(n_layers, seq_len, dim, causal=True):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(num_non_embed_params, n_layers, dim, seq_len):
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )

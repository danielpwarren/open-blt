import torch as pt
import torch.nn as nn
import torch.nn.functional as F


class EntropyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Create embedding: maps each token id to a vector of dimension config.dim.
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Compute the feed-forward dimension using the multiplier.
        dim_feedforward = int(config.dim * config.ffn_dim_multiplier)

        # Build a stack of transformer encoder layers with batch_first.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.dim,
                    nhead=config.n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.norm = nn.LayerNorm(config.dim)
        self.proj = nn.Linear(config.dim, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            pt.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                pt.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            pt.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_local_causal_mask(self, seq_len):
        """
        Create a local (sliding-window) causal mask.
        For each batch, tokens only attend to tokens in the window [max(0, i - sliding_window + 1), i].
        Returns a 2D tensor of shape (seq_len, seq_len), which will be broadcasted over the batch.
        """
        device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
        i = pt.arange(seq_len, device=device).unsqueeze(1)
        j = pt.arange(seq_len, device=device).unsqueeze(0)
        window = self.config.sliding_window
        # Allowed positions: j in [max(0, i-window+1), i]
        allowed = (j <= i) & (j >= i - window + 1)
        mask = (
            (~allowed)
            .float()
            .masked_fill(~allowed, float("-inf"))
            .masked_fill(allowed, 0.0)
        )
        return mask

    def forward(self, x):
        # x: [batch, seq_len] token IDs.
        x = self.embed(x)  # -> [batch, seq_len, dim]

        # Create a mask if requested by config.
        if (
            hasattr(self.config, "attn_bias_type")
            and self.config.attn_bias_type == "local_block_causal"
        ):
            if (
                not hasattr(self.config, "sliding_window")
                or self.config.sliding_window is None
            ):
                raise ValueError(
                    "sliding_window must be set when using attn_bias_type 'local_block_causal'"
                )
            mask = self._create_local_causal_mask(x.size(1))
        else:
            mask = None

        # Pass x through all transformer layers.
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        x = self.norm(x)
        return self.proj(x)  # -> [batch, seq_len, vocab_size]

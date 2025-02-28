import torch as pt
import torch.nn as nn
from omegaconf import OmegaConf

from blt.model.transformer import LMTransformer, LMTransformerArgs


class EntropyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        transformer_args_dict = OmegaConf.to_container(config.model, resolve=True)
        transformer_args_dict["max_seqlen"] = config.data.seq_len
        transformer_args_dict["sliding_window"] = transformer_args_dict.get(
            "sliding_window", None
        )

        transformer_args = LMTransformerArgs(**transformer_args_dict)
        self.transformer = LMTransformer(transformer_args)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            pt.nn.init.trunc_normal_(
                module.weight, mean=0.0, std=0.02, a=-3 * 0.02, b=3 * 0.02
            )
            if module.bias is not None:
                pt.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            pt.nn.init.trunc_normal_(
                module.weight, mean=0.0, std=0.02, a=-3 * 0.02, b=3 * 0.02
            )

    def forward(self, x):
        return self.transformer(x)

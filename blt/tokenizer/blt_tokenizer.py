from typing import List, Optional

from .abstract_tokenizer import Tokenizer
from .constants import BOE_ID, BOS_ID, BYTE_UNITS, EOS_ID, OFFSET, PAD_ID


class BLTTokenizer(Tokenizer):
    def __init__(
        self,
        *,
        vocab_size_unit_1: int = BYTE_UNITS,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.boe_id = BOE_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.offsetting_special_char = OFFSET
        self.n_words = vocab_size_unit_1 + self.offsetting_special_char

    def encode(
        self, text: str, add_bos: Optional[bool] = None, add_eos: Optional[bool] = None
    ) -> List[int]:
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        # Convert text to bytes and offset
        tokens = bytes(text, encoding="utf-8", errors="ignore")
        tokens = [int(unit) + self.offsetting_special_char for unit in tokens]

        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: List[int], cut_at_eos: bool = False) -> str:
        if cut_at_eos:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[: k + 1]
                    break
        return bytes(
            [
                tok - self.offsetting_special_char
                for tok in tokens
                if tok - self.offsetting_special_char >= 0
            ]
        ).decode("utf-8", errors="ignore")

    def get_token_offsets(self, text: str, tokens: Optional[List[int]] = None):
        raise NotImplementedError("Token offset calculation not implemented yet")

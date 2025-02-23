import abc

class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, text: str, add_bos: bool, add_eos: bool):
        pass

    @abc.abstractmethod
    def decode(self, tokens: list[int]):
        pass

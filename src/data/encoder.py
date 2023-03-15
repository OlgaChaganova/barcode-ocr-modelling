import torch


class Encoder(object):

    @classmethod
    def encode(cls, text: str) -> torch.tensor:
        text = list(map(int, list(text)))
        return torch.tensor(text)

    @classmethod
    def decode(cls, encoded_text: torch.tensor) -> str:
        text = encoded_text.tolist()
        text = list(map(str, text))
        return ''.join(text)

import torch


class Encoder(object):
    def __init__(self):
        ...

    @staticmethod
    def encode(text: str) -> torch.tensor:
        text = list(map(int, list(text)))
        return torch.tensor(text) + 1  # + 1 because blank is 0

    @staticmethod
    def decode(encoded_text: torch.tensor) -> str:
        text = (encoded_text - 1).tolist()
        text = list(map(str, text))
        return ''.join(text)
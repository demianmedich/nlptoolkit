# coding=utf-8
from typing import List
from typing import Mapping

from nlptoolkit.util import convert_to_unicode


class BaseTokenizer:
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path
        self._vocab = load_vocab(vocab_path)
        self._inv_vocab = {v: k for k, v in self._vocab.items()}

    @property
    def vocab(self) -> Mapping[str, int]:
        return self._vocab

    @property
    def inv_vocab(self) -> Mapping[int, str]:
        return self._inv_vocab

    def tokenize(self, sequence: str) -> List[str]:
        raise NotImplementedError()

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError()

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        raise NotImplementedError()

    def is_start_token(self, token):
        raise NotImplementedError()


def load_vocab(vocab_path: str) -> Mapping[str, int]:
    with open(vocab_path, mode="rt", encoding="utf-8") as f:
        vocab = {}
        idx = 0
        while True:
            line = f.readline()
            if not line:
                break
            line = convert_to_unicode(line).strip()
            if not line:
                raise ValueError("Vocab has invalid whitespace character.")

            vocab[line] = idx
            idx += 1
    return vocab

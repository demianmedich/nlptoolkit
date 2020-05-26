# coding=utf-8
from nlptoolkit.tokenizer import BaseTokenizer
from nlptoolkit.tokenizer import BertWordPieceTokenizer


def create_tokenizer(tokenizer_type: str, **kwargs) -> BaseTokenizer:
    if tokenizer_type == "BertWordPieceTokenizer":
        do_lower_case = kwargs.get("do_lower_case", False)
        tokenizer_vocab = kwargs["tokenizer_vocab"]
        tokenizer = BertWordPieceTokenizer(tokenizer_vocab,
                                           do_lower_case=do_lower_case)
    else:
        raise NotImplementedError()
    return tokenizer

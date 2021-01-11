# coding=utf-8
import argparse
import sys

from tokenizers import Tokenizer, normalizers, WordPieceTrainer
from tokenizers.models import WordPiece, WordLevel, BPE, Unigram
from tokenizers.normalizers import NFKC, StripAccents, Lowercase
from tokenizers.pre_tokenizers import Whitespace

from nlptoolkit.constants import UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, \
    CLS_TOKEN, SEP_TOKEN, MASK_TOKEN, NUM_TOKEN
from nlptoolkit.utils import get_all_filenames

SPECIAL_TOKENS = [
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, CLS_TOKEN,
    SEP_TOKEN, MASK_TOKEN, NUM_TOKEN,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--i", type=str, required=True,
                        help="input files or root directory. "
                             "if directory given, all sub-files gathered.")
    parser.add_argument("--output", "--o", type=str, required=True,
                        help="tokenizer path")
    parser.add_argument("--model", "--m", type=str, default="wordpiece",
                        choices=["bpe", "word", "wordpiece", "unigram"])
    parser.add_argument("--lower-case", action="store_true",
                        help="lowering all characters in a sequence")
    parser.add_argument("--vocab-size", type=int, default=32768,
                        help="vocab size")
    parser.add_argument("--reserved", type=int, default=1000,
                        help="number of reserved tokens include special tokens")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    special_tokens = list(SPECIAL_TOKENS)

    if args.reserved < len(special_tokens):
        raise AssertionError(
            f"number of reserved tokens should be more than number of f{len(special_tokens)}")
    for i in range(len(special_tokens), args.reserved):
        special_tokens.append(f"[unused{i:03d}]")

    all_filenames = get_all_filenames(args.input)
    # "C:\Users\demianmedich\data\wiki\20191120.en\pp_cased/"

    tokenizer = Tokenizer(get_model(args.model))
    tokenizer.normalizer = normalizers.Sequence([
        NFKC(), StripAccents(), Lowercase()
    ])

    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens)
    tokenizer.train(trainer, all_filenames)

    model_files = tokenizer.model.save()

    sys.exit(0)


def get_model(name: str):
    if name == "wordpiece":
        return WordPiece(unk_token=UNK_TOKEN)
    elif name == "bpe":
        return BPE(unk_token=UNK_TOKEN)
    elif name == "unigram":
        return Unigram()
    elif name == "word":
        return WordLevel(unk_token=UNK_TOKEN)
    else:
        raise AssertionError(f"{name} type model is not granted.")


if __name__ == '__main__':
    main()

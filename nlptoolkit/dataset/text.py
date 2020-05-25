# coding=utf-8
import random
from itertools import chain
from itertools import cycle
from typing import Iterable
from typing import List

import torch
from torch.utils.data import IterableDataset

from nlptoolkit.tokenizer import BaseTokenizer


class IterableTextLineDataset(IterableDataset):
    """Text line 단위로 하나의 정보가 구성되는 경우 사용.
    TensorFlow 의 tfrecord 포맷과 대응될 수 있도록 구조
    """

    def __init__(self,
                 corpus_files: List[str],
                 num_replicas: int = 1,
                 rank: int = 0,
                 infinite: bool = False,
                 shuffle_files: bool = True,
                 seed: int = 1):
        self.corpus_files = corpus_files
        self.num_replicas = num_replicas
        self.rank = rank
        self.infinite = infinite
        self.shuffle_files = shuffle_files
        self.seed = seed
        random.seed(self.seed)

        # multi-processing 환경일 경우, 동일 seed 를 줘야 동일한 순서로,
        # 중복된 데이터 없이 각 process 로 나눌 수 있다.
        if self.shuffle_files:
            self.corpus_files = random.sample(self.corpus_files,
                                              len(self.corpus_files))

        if self.num_replicas > 1:
            # replica 별로 다른 데이터가 들어갈 수 있도록 sharding
            self.corpus_files = [file for i, file in
                                 enumerate(self.corpus_files) if
                                 i % self.num_replicas == self.rank]

    def __iter__(self) -> Iterable:
        if self.infinite:
            streams = cycle(self.corpus_files)
        else:
            streams = self.corpus_files
        corpus_iterator = chain.from_iterable(map(self.lines, streams))
        return corpus_iterator

    def process_line(self, line: str):
        return line

    def lines(self, json_file_path: str):
        with open(json_file_path, mode="rt", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                yield self.process_line(line)


class StringToTensor:
    def __init__(self,
                 tokenizer: BaseTokenizer,
                 max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, str_list: List[str]):
        tokens_list = []
        for i, s in enumerate(str_list):
            tokens = self.tokenizer.tokenize(s)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            else:
                while len(tokens) < self.max_seq_len:
                    tokens.append(self.tokenizer.PAD_TOKEN)
            tokens_list.append(self.tokenizer.tokens_to_ids(tokens))
        return torch.tensor(tokens_list, dtype=torch.long)

# coding=utf-8
import json
from typing import List

from nlptoolkit.dataset import IterableTextLineDataset


class JsonTextLineDataset(IterableTextLineDataset):
    """Json format 으로 구성된 line 을 읽어들여서 parsing.

    TensorFlow 의 tfrecord 포맷과 대응되는 형태.
    """

    def __init__(self,
                 corpus_files: List[str],
                 features: List[str],
                 num_replicas: int = 1,
                 rank: int = 0,
                 infinite: bool = False,
                 shuffle_files: bool = True,
                 seed: int = 1):
        super(JsonTextLineDataset, self).__init__(corpus_files,
                                                  num_replicas=num_replicas,
                                                  rank=rank,
                                                  infinite=infinite,
                                                  shuffle_files=shuffle_files,
                                                  seed=seed)
        self.features = features

    def process_line(self, json_str: str):
        json_obj = json.loads(json_str)
        return {feature: json_obj[feature] for feature in self.features}

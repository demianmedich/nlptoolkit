# coding=utf-8
import itertools
import unittest

from torch.utils.data.dataloader import DataLoader

from nlptoolkit.dataset.text import IterableTextLineDataset, StringToIds
from nlptoolkit.tokenizer import BertWordPieceTokenizer


class DatasetTestCase(unittest.TestCase):
    def test_iterator(self):
        lst = [i for i in range(10)]
        lst = itertools.cycle(lst)
        for i in itertools.islice(iter(lst), 20):
            print(i)

    def test_json_dataset(self):
        corpus_files = [
            "C:/Users/demianmedich/data/wiki_sample_001.txt",
            "C:/Users/demianmedich/data/wiki_sample_002.txt",
            "C:/Users/demianmedich/data/medium_sample_001.txt",
        ]
        vocab_file = "../../vocab/bert_wordpiece_en_cased_vocab.txt"
        tokenizer = BertWordPieceTokenizer(vocab_file, do_lower_case=False)
        max_seq_len = 512
        # tokenizer = None
        ds = IterableTextLineDataset(corpus_files,
                                     num_replicas=2,
                                     rank=1,
                                     infinite=False,
                                     shuffle_files=False)
        collate_fn = StringToIds(tokenizer, max_seq_len)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

        for batch in itertools.islice(loader, 100):
            print(batch)


if __name__ == '__main__':
    unittest.main()

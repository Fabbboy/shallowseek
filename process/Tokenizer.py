from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
    normalizers,
    processors,
)
from typing import List

import tokenizers


class BPETokenizer:
    def __init__(self, vocab_size=30000, min_frequency=2):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.normalizer = normalizers.NFKC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel()

        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOS]"]

        self.trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
        )

    def train(self, data: List[str]):
        """
        Train the tokenizer directly on a list of text strings.

        Args:
            data (List[str]): The text data to train on.
        """
        # Train from memory using an iterator
        self.tokenizer.train_from_iterator(data, trainer=self.trainer)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text) -> tokenizers.Encoding:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def inner_tokenizer(self):
        return self.tokenizer

    def get_pad_token(self):
        return self.tokenizer.token_to_id("[PAD]")

    def get_eos_token(self):
        return self.tokenizer.token_to_id("[EOS]")

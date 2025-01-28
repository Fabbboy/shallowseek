from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
    normalizers,
    processors,
)


class BPETokenizer:
    def __init__(self, vocab_size=30000, min_frequency=2):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.normalizer = normalizers.NFKC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel()

        self.trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency
        )

    def train(self, files):
        self.tokenizer.train(files, self.trainer)

    def save(self, path):
        self.tokenizer.save(path)

    @classmethod
    def from_file(cls, path):
        tokenizer = cls()
        tokenizer.tokenizer = Tokenizer.from_file(path)
        return tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

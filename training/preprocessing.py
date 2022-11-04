from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_vocabulary(training_data_path, training_data_name):
    dataframe = pd.read_csv(os.path.join(BASE_DIR, training_data_path, training_data_name))
    train_iterator = list(zip(dataframe['text'], dataframe['label']))
    tokenizer = get_tokenizer("basic_english")

    def tokenizer_fn(data_iterator):
        for text, _ in data_iterator:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(tokenizer_fn(train_iterator), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab, tokenizer

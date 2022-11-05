from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os

from preprocessing.preprocessing import clean_text, tokenize_text, pad_tokens

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LSTMDataLoader:
    def __init__(self, data_path, dataset_name, vocab, tokenizer, batch_size, clean_text, seq_len):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.clean_text = clean_text,
        self.seq_len = seq_len

    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset_name)

        dataframe = pd.read_csv(data_path)
        print(f'Loaded {self.dataset_name} with {len(dataframe)} samples')

        if self.clean_text:
            print(f'Cleaning {self.dataset_name}')
            dataframe = clean_text(dataframe)
        else:
            print(f"Skipping cleaning {self.dataset_name}")

        print(f'Tokenizing {self.dataset_name}')
        dataframe = tokenize_text(dataframe, self.vocab, self.tokenizer)

        print(f'Padding {self.dataset_name}\n')
        dataframe = pad_tokens(dataframe=dataframe, vocab=self.vocab, max_len=self.seq_len)

        dataset = TensorDataset(
            torch.from_numpy(np.vstack(dataframe["text"].values)),
            torch.from_numpy(dataframe["label"].values)
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1
        )

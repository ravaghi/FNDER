from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def complete_batch(dataframe, batch_size):
    complete_buckets = []
    buckets = [bucket_df for _, bucket_df in dataframe.groupby('bucket')]

    for gr_id, bucket in enumerate(buckets):
        l = len(bucket)
        remainder = l % batch_size
        integer = l // batch_size

        if remainder != 0:
            bucket = pd.concat([bucket, pd.concat([bucket.iloc[:1]] * (batch_size - remainder))], ignore_index=True)
            integer += 1

        batch_ids = []
        for i in range(integer):
            batch_ids.extend([f'{i}_bucket{gr_id}'] * batch_size)

        bucket['batch_id'] = batch_ids
        complete_buckets.append(bucket)
    return pd.concat(complete_buckets, ignore_index=True)


def shuffle_batches(dataframe):
    batch_buckets = [df_new for _, df_new in dataframe.groupby('batch_id')]
    random.shuffle(batch_buckets)
    return pd.concat(batch_buckets).reset_index(drop=True)


def concater_collate(batch):
    (xx, yy, lengths, buckets) = zip(*batch)
    xx = torch.cat(xx, 0)
    yy = torch.tensor(yy)
    return xx, yy, list(lengths), list(buckets)


def process_dataframe(dataframe, vocab, tokenizer):
    dataframe["text"] = dataframe["text"].apply(lambda x: np.array(vocab(tokenizer(x))))
    dataframe['seq_len'] = dataframe['text'].apply(lambda x: len(x))

    percentiles = [i * 0.1 for i in range(10)] + [.95, .99, .995]
    buckets = np.quantile(dataframe['seq_len'], percentiles)
    bucket_labels = [i for i in range(len(buckets) - 1)]
    dataframe['bucket'] = pd.cut(dataframe['seq_len'], bins=buckets, labels=bucket_labels)

    dataframe["seq_len"] = dataframe["seq_len"].astype(int)

    return dataframe[['text', 'label', "seq_len", "bucket"]]


class DatasetCreator(Dataset):
    def __init__(self, dataframe, batch_size):
        dataframe = complete_batch(dataframe=dataframe, batch_size=batch_size)
        dataframe = shuffle_batches(dataframe=dataframe)
        self.dataframe = dataframe[['text', 'label', 'seq_len', 'bucket']]

    def __getitem__(self, index):
        X, Y, seq_len, bucket = self.dataframe.iloc[index, :]
        Y = torch.tensor(Y)
        X = torch.from_numpy(X)
        return X, Y, seq_len, bucket

    def __len__(self):
        return len(self.dataframe)


class ChordMixerDataLoader:
    def __init__(self, data_path, dataset_name, vocab, tokenizer, batch_size):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset_name)
        dataframe = pd.read_csv(data_path)
        dataframe = process_dataframe(dataframe, self.vocab, self.tokenizer)

        dataset = DatasetCreator(
            dataframe=dataframe,
            batch_size=self.batch_size
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=concater_collate,
            drop_last=False,
            num_workers=1
        )
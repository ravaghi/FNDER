from sklearn.utils.class_weight import compute_class_weight
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import random
import torch
import wandb
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_run(config):
    seed_everything(config.general.seed)

    wandb.init(project=config.wandb.project,
               entity=config.wandb.entity,
               config=OmegaConf.to_container(config, resolve=True),
               dir=BASE_DIR)

    print("-" * 30 + " config " + "-" * 30)
    print(OmegaConf.to_yaml(config))
    print("-" * 30 + " config " + "-" * 30)

    device = f'cuda:{config.general.device_id}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    return device


def get_class_weights(dataset_path, dataset_name):
    train_data_path = os.path.join(dataset_path, dataset_name)
    dataframe = pd.read_csv(train_data_path)
    return compute_class_weight('balanced', classes=[0, 1], y=dataframe["label"])


def process_data():
    dataset_path = os.path.join(os.path.dirname(__file__), 'data')

    processed_train_path = os.path.join(dataset_path, 'train.csv')
    processed_val_path = os.path.join(dataset_path, 'val.csv')
    processed_test_path = os.path.join(dataset_path, 'test.csv')

    if os.path.exists(processed_train_path) and \
            os.path.exists(processed_val_path) and \
            os.path.exists(processed_test_path):
        print('Data already exists')
        exit()
    else:
        print('Processing data')
        unprocessed_fake_data_path = os.path.join(dataset_path, 'Fake.csv')
        unprocessed_real_data_path = os.path.join(dataset_path, 'True.csv')

        fake = pd.read_csv(unprocessed_fake_data_path)
        real = pd.read_csv(unprocessed_real_data_path)

        fake['label'] = 0
        real['label'] = 1

        # Combine and shuffle
        combined = pd.concat([fake, real], axis=0)
        combined = combined.sample(frac=1).reset_index(drop=True)

        # Split dataset into train, val and test
        train = combined[:int(0.8 * len(combined))]
        val = combined[int(0.8 * len(combined)):int(0.9 * len(combined))]
        test = combined[int(0.9 * len(combined)):]

        # Save datasets
        train.to_csv(processed_train_path, index=False)
        val.to_csv(processed_val_path, index=False)
        test.to_csv(processed_test_path, index=False)

        # Delete old files
        os.remove(os.path.join(dataset_path, 'Fake.csv'))
        os.remove(os.path.join(dataset_path, 'True.csv'))

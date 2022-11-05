from sklearn.utils.class_weight import compute_class_weight
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import random
import torch
import wandb
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
               name=config.wandb.name,
               dir=BASE_DIR)

    print("-" * 30 + " config " + "-" * 30)
    print(OmegaConf.to_yaml(config))
    print("-" * 30 + " config " + "-" * 30)

    device = f'cuda:{config.general.device_id}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}\n')

    return device


def get_class_weights(dataset_path, dataset_name):
    train_data_path = os.path.join(dataset_path, dataset_name)
    dataframe = pd.read_csv(train_data_path)
    return compute_class_weight('balanced', classes=[0, 1], y=dataframe["label"])

import sys
import torch.nn as nn
import torch
import torch.optim as optim
import argparse
from module.data_module import preprocessor, analyser, feature_engineer, y_dataset
from module.categorical_module import cat_dataset
from module.scalar_module import sc_dataset
from model import MainModel
import torch.utils.data as data
import numpy as np


class ConcatDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, x_list, y):
        self.x_list = x_list
        self.y = y

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return [dataset.__getitem__(index) for dataset in self.x_list], self.y.__getitem__(index)

    def __len__(self):
        return len(self.y)


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


def get_data_loader(df, fg, args):
    categorical_dataset = cat_dataset.get_dataset(df, fg.categorical_columns)
    scalar_dataset = sc_dataset.get_dataset(df, fg.scalar_columns)
    label_dataset = y_dataset.get_dataset(df, fg.y_column)
    dataset = ConcatDataset([categorical_dataset, scalar_dataset], label_dataset)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=args.batch_size,
                                              collate_fn=my_collate,
                                              shuffle=True)

    return data_loader

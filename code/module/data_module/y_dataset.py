import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, df, column):
        self.sub_df = df[[column]]

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return torch.FloatTensor(self.sub_df.iloc[index, :])

    def __len__(self):
        return self.sub_df.shape[0]


def get_dataset(df, column):
    dataset = Dataset(df, column)
    return dataset

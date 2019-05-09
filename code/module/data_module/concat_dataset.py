import torch
import torch.utils.data as data

from module.categorical_module import cat_dataset
from module.data_module import y_dataset
from module.scalar_module import sc_dataset
from module.text_module import txt_dataset
from module.image_module import img_dataset


class ConcatDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, x_list, y):
        self.x_list = x_list
        self.y = y

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        if self.y is not None:
            return [dataset.__getitem__(index) for dataset in self.x_list], self.y.__getitem__(index)
        else:
            return [dataset.__getitem__(index) for dataset in self.x_list], None

    def __len__(self):
        return self.x_list[0].__len__()


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


def my_collate_test(batch):
    data = [item[0] for item in batch]
    return data


def get_data_loader(df, fg, args, vocab, handler, val=False, test=False):
    categorical_dataset = cat_dataset.get_dataset(df, fg.categorical_columns)
    scalar_dataset = sc_dataset.get_dataset(df, fg.scalar_columns)

    overview_dataset = txt_dataset.get_dataset(df, fg.text_column[0], vocab)
    tagline_dataset = txt_dataset.get_dataset(df, fg.text_column[1], vocab)
    title_dataset = txt_dataset.get_dataset(df, fg.text_column[2], vocab)

    poster_dataset = img_dataset.get_dataset(df, handler)
    dataset = None
    data_loader = None

    if not test:
        label_dataset = y_dataset.get_dataset(df, fg.y_column)
        dataset = ConcatDataset(
            [categorical_dataset, scalar_dataset, overview_dataset, tagline_dataset, title_dataset, poster_dataset],
            label_dataset)

    if test:
        dataset = ConcatDataset(
            [categorical_dataset, scalar_dataset, overview_dataset, tagline_dataset, title_dataset, poster_dataset],
            None)

    if not test:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=args.batch_size,
                                                  collate_fn=my_collate,
                                                  shuffle=not val)

    if test:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=args.batch_size,
                                                  collate_fn=my_collate_test,
                                                  shuffle=not val)

    return data_loader

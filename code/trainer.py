import sys
import torch.nn as nn
import torch
import torch.optim as optim
import argparse
from module.data_module import preprocessor, analyser, feature_engineer, y_dataset
from module.categorical_module import cat_dataset
from module.scalar_module import sc_dataset
from model import MainModel


def trainer(args, train_loader, test_loader, model):
    best_loss = sys.maxsize
    stop_count = 0
    criterion = nn.MSELoss()

    if args.gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        running_loss = 0.0
        for i, datas in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = datas

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            max_vals, max_indices = torch.max(outputs, 1)
            train_acc = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]

            if i % args.print_batch == args.print_batch - 1:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / args.print_batch), ' train_acc', train_acc)
                running_loss = 0.0
        loss = 999999999999999

        if loss < best_loss:
            best_loss = loss
        else:
            stop_count += 1
        if stop_count == args.early_stop:
            break
    print('Finished Training')
    return model


import torch.utils.data as data


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


def get_data_loader(df, fg, batch_size):
    categorical_dataset = cat_dataset.get_dataset(df, fg.categorical_columns)
    scalar_dataset = sc_dataset.get_dataset(df, fg.scalar_columns)
    label_dataset = y_dataset.get_dataset(df, fg.y_column)
    dataset = ConcatDataset([categorical_dataset, scalar_dataset], label_dataset)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=32,
                                              collate_fn=my_collate,
                                              shuffle=True)
    return data_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--print_batch", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--analysis", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--gpu", type=bool, default=torch.cuda.is_available())
    args = parser.parse_args()

    train_df, test_df = preprocessor.load_data()
    if args.analysis:
        analyser.analysis(train_df)
    else:
        preprocessor.preprocess(train_df)

    fg = feature_engineer.FeatureEngine()
    fg.run(train_df)

    if args.analysis:
        analyser.analysis_after_transform(train_df)

    data_loader = get_data_loader(train_df, fg, args.batch_size)
    model = MainModel(fg.categorical_dims, len(fg.scalar_columns))
    trainer(args, data_loader, None, model)


main()

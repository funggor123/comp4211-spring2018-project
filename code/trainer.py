import sys
import torch.nn as nn
import torch
import torch.optim as optim
import argparse
from module.data_module import preprocessor, analyser, feature_engineer
from module.categorical_module import cat_dataset
from module.scalar_module import sc_dataset
from torch.utils.data import ConcatDataset


def trainer(args, train_loader, test_loader, model, lr, epoch, print_batch, early_stop=100):
    best_loss = sys.maxsize
    stop_count = 0
    criterion = nn.MSELoss()

    if args.gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            if args.gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

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

            if i % print_batch == print_batch - 1:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_batch), ' train_acc', train_acc)
                running_loss = 0.0
        loss = 999999999999999

        if loss < best_loss:
            best_loss = loss
        else:
            stop_count += 1
        if stop_count == early_stop:
            break
    print('Finished Training')
    return model


def preparing_data_loader(df, fg):
    categorical_dataset = cat_dataset.get_dataset(df, fg.categorical_columns)
    scalar_dataset = sc_dataset.get_dataset(df, fg.scalar_columns)
    dataset = ConcatDataset([categorical_dataset, scalar_dataset])
    print(dataset.__getitem__(0))
    print("rdg")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=20)
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

    preparing_data_loader(train_df, fg)

main()
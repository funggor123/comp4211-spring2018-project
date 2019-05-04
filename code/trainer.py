import sys
import torch.nn as nn
import torch
import torch.optim as optim
import argparse
from module.data_module import preprocessor, analyser, feature_engineer, concat_dataset
from sklearn.model_selection import train_test_split
from model import MainModel
import numpy as np
from tensorboardX import SummaryWriter

writer_train = SummaryWriter('runs/train_0')
writer_vad = SummaryWriter('runs/vad_0')


def trainer(args, train_loader, test_loader, model):
    best_loss = sys.maxsize
    stop_count = 0
    criterion = nn.MSELoss()

    if args.gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epoch):
        running_loss = 0.0
        for i, datas in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = datas

            # forward + backward + optimize
            labels = torch.stack(labels).cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % args.print_batch == args.print_batch - 1:
                print('[%d, %5d] Training loss (Running): %.3f' %
                      (epoch + 1, i + 1, running_loss / args.print_batch))
                if epoch % args.sampling_epoch == args.sampling_epoch - 1:
                    print("|------Output Sampling---------|")
                    print(np.expm1(outputs.cpu().detach().numpy()))
                    print(np.expm1(labels.cpu().detach().numpy()))
                    print("--------------------------------")
                running_loss = 0.0

        if args.early_stop is not -1:
            vad_loss = validator(args, test_loader, model)
            train_loss = validator(args, train_loader, model)
            writer_train.add_scalar('MSE_loss_train', train_loss, epoch + 1)
            writer_vad.add_scalar('MSE_loss_validation', vad_loss, epoch + 1)
            if epoch % args.print_epoch == args.print_epoch - 1:
                print('[%d] Training loss (Total): %.3f' %
                      (epoch + 1, train_loss))
                print('[%d] Validation loss (Total): %.3f' %
                      (epoch + 1, vad_loss))

            if vad_loss < best_loss:
                best_loss = vad_loss
            else:
                stop_count += 1
            if stop_count == args.early_stop:
                break

    return model


def validator(args, test_loader, model):
    running_loss = 0
    criterion = nn.MSELoss()
    batch_count = 0

    if args.gpu:
        model = model.cuda()

    for i, data in enumerate(test_loader, 0):
        # get the inputs
        inputs, labels = data

        # forward + backward + optimize
        labels = torch.stack(labels).cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # print statistics
        running_loss += loss.item()
        batch_count += 1

    return running_loss / batch_count


def predictor():
    print("hi")


def tuner():
    print("hi")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--early_stop", type=int, default=300000)
    parser.add_argument("--print_batch", type=int, default=70)
    parser.add_argument("--print_epoch", type=int, default=5)
    parser.add_argument("--sampling_epoch", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=100000)
    parser.add_argument("--drop_prob", type=int, default=0.3)
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--test_size", type=float, default=0.2)
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

    if args.train:
        train_df, vad_df = train_test_split(train_df, test_size=args.test_size)
        train_data_loader = concat_dataset.get_data_loader(train_df, fg, args)
        val_data_loader = concat_dataset.get_data_loader(vad_df, fg, args)
        model = MainModel(fg.categorical_dims, len(fg.scalar_columns))
        model = trainer(args, train_data_loader, val_data_loader, model)


main()

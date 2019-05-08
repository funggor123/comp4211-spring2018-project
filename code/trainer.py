import sys
import torch.nn as nn
import torch
import torch.optim as optim
import argparse
from module.data_module import preprocessor, analyser, feature_engineer, concat_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from model import MainModel
import numpy as np
from tensorboardX import SummaryWriter
from module.text_module.txt_dataset import getVocab
from module.image_module.image_handler import ImageHandler

writer_train = SummaryWriter('logs/train_0')
writer_vad = SummaryWriter('logs/vad_0')


# https://www.kaggle.com/nakayamar/revenue-prediction-with-posters-using-cnn-keras
def trainer(args, train_loader, test_loader, model, fg=None):
    best_loss = sys.maxsize
    stop_count = 0
    criterion = nn.MSELoss()

    if args.gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        running_loss = 0.0
        model.train()
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
                    print(outputs.cpu().detach().numpy())
                    print(labels.cpu().detach().numpy())
                    print("----------Real------------------")
                    print(np.expm1(fg.y_scalar.inverse_transform(outputs.cpu().detach().numpy())))
                    print(np.expm1(fg.y_scalar.inverse_transform(labels.cpu().detach().numpy())))
                    print("--------------------------------")
                running_loss = 0.0

        if args.early_stop is not -1:

            writer_train.add_histogram("last_linear", model.linear_last.weight.grad, epoch + 1)
            writer_train.add_histogram("linear", model.linear1[0].weight.grad, epoch + 1)
            writer_train.add_histogram("linear2", model.linear2[0].weight.grad, epoch + 1)

            writer_train.add_histogram("attention_linear0", model.categorical_encoder.attention_linear[0].weight.grad,
                                       epoch + 1)
            writer_train.add_histogram("attention_linear1", model.categorical_encoder.attention_linear[1].weight.grad,
                                       epoch + 1)
            writer_train.add_histogram("attention_linear2", model.categorical_encoder.attention_linear[2].weight.grad,
                                       epoch + 1)
            writer_train.add_histogram("attention_linear3", model.categorical_encoder.attention_linear[3].weight.grad,
                                       epoch + 1)
            writer_train.add_histogram("attention_linear4", model.categorical_encoder.attention_linear[4].weight.grad,
                                       epoch + 1)
            writer_train.add_histogram("attention_linear5", model.categorical_encoder.attention_linear[5].weight.grad,
                                       epoch + 1)
            writer_train.add_histogram("attention_linear6", model.categorical_encoder.attention_linear[6].weight.grad,
                                       epoch + 1)

            writer_train.add_histogram("sc_encoder_linear", model.scalar_encoder.linear[0].weight.grad, epoch + 1)
            writer_train.add_histogram("ca_encoder_linear", model.categorical_encoder.encode_linear[0].weight.grad,
                                       epoch + 1)

            writer_train.add_histogram("title_encoder_linear", model.title_encoder.linear_hidden[0].weight.grad,
                                       epoch + 1)
            writer_train.add_histogram("title_encoder_linear", model.title_encoder.attention_linear.weight.grad,
                                       epoch + 1)

            writer_train.add_histogram("overview_encoder_linear", model.overview_encoder.linear_hidden[0].weight.grad,
                                       epoch + 1)
            writer_train.add_histogram("overview_encoder_linear", model.overview_encoder.attention_linear.weight.grad,
                                       epoch + 1)

            vad_loss = validator(args, test_loader, model)
            train_loss = validator(args, train_loader, model)
            writer_train.add_scalar('MSE_loss', train_loss, epoch + 1)
            writer_vad.add_scalar('MSE_loss', vad_loss, epoch + 1)
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

    model.eval()
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


def predictor(args, test_loader, model):
    results = []

    model.eval()
    if args.gpu:
        model = model.cuda()

    for i, data in enumerate(test_loader, 0):
        # get the inputs
        inputs = data
        outputs = model(inputs)
        results.append(np.expm1(outputs.cpu().detach().numpy()))

    sub1 = pd.read_csv('./sample_submission.csv')
    sub1['revenue'] = results
    sub_file_name = 'results_' + str(time.time()) + '.csv'
    sub1.to_csv(sub_file_name, index=False)
    print("submit file :" + sub_file_name)


# Drop Rate, Learning Rate
def tuner():
    print("hi")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=100000)
    parser.add_argument("--drop_prob", type=int, default=0.3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--tune", type=bool, default=False)
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--analysis", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--test", type=bool, default=True)
    parser.add_argument("--gpu", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--early_stop", type=int, default=300000)
    parser.add_argument("--print_batch", type=int, default=70)
    parser.add_argument("--print_epoch", type=int, default=5)
    parser.add_argument("--sampling_epoch", type=int, default=50)
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

    model = None
    vocab = getVocab()
    if args.train:
        handler = ImageHandler("train")
        handler.download_all_posters(train_df)
        train_df, vad_df = train_test_split(train_df, test_size=args.test_size, random_state=42)
        train_data_loader = concat_dataset.get_data_loader(train_df, fg, args, vocab, handler)
        val_data_loader = concat_dataset.get_data_loader(vad_df, fg, args, vocab, handler, test=True)
        model = MainModel(fg.categorical_dims, len(fg.scalar_columns), embedding_matrix=vocab.embeddings_matrix)
        model = trainer(args, train_data_loader, val_data_loader, model, fg)

    if args.test and model is not None:
        # FE
        handler = ImageHandler("test")
        handler.download_all_posters(test_df)
        test_data_loader = concat_dataset.get_data_loader(test_df, fg, args, vocab, handler, test=True)
        predictor(args, test_data_loader, model)


main()

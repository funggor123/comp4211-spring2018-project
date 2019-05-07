from module.text_module.model import BiLSTMAttention
from module.text_module.txt_dataset import get_dataloaders
import torch
from torch import nn
import torch.optim as optim
from module.data_module import preprocessor

mode_path = "./model/text/params.pkl"
is_gpu = torch.cuda.is_available()


def train(train_loader, test_loader, num_class, model, early_stop=100, load_model=False):
    best_loss = 999999
    stop_count = 0
    criterion = nn.CrossEntropyLoss()
    net = model
    if load_model:
        net = net.load_state_dict(torch.load(mode_path))
    if is_gpu:
        net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(150):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            if is_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            max_vals, max_indices = torch.max(outputs, 1)
            train_acc = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]

            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20), ' train_acc', train_acc)
                running_loss = 0.0
        loss = 999999999999999
        if loss < best_loss:
            best_loss = loss
        else:
            stop_count += 1
        if stop_count == early_stop:
            break
    torch.save(net.state_dict(), mode_path)
    print('Finished Training')
    return net


def main():
    train_df, test_df = preprocessor.load_data()
    preprocessor.preprocess(train_df)
    train_loader, embedding_matrix, num_of_vocab = get_dataloaders(32, train_df)
    model = BiLSTMAttention(num_class=10, embedding_matrix=embedding_matrix)
    net = train(train_loader, None, num_class=9, model=model)


main()

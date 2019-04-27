import torch
from torch import nn
import torchvision
import urllib.request
import preprocess
import os
from collections import Counter
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt

is_gpu = torch.cuda.is_available()
normalize = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])

poster_dir_path = "./data/poster/"
unk_poster_dir_path = "./data/unk_poster/"
poster_url = "http://image.tmdb.org/t/p/w185"
mode_path = "./model/"


def download_all_posters(df):
    genre = df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    directory = os.path.dirname(poster_dir_path)
    num_class = 0
    for ele in list(dict(Counter([i for j in genre for i in j])).keys()):
        num_class += 1
        if not os.path.exists(directory + "/" + ele):
            os.makedirs(directory + "/" + ele)
    directory = os.path.dirname(unk_poster_dir_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, url in enumerate(df["poster_path"]):
        if len(df["genres"][i]) == 0:
            download_poster_unk(url, i)
            print("Found UNK genres: ", i)
        print(i, "/" + str(len(df["poster_path"])))
        for genre in df["genres"][i]:
            if not os.path.exists(poster_dir_path +
                                  genre["name"] + "/" + str(i) + ".jpg"):
                download_poster(url, genre["name"], i)
    return num_class


def download_poster_unk(poster_path, ids):
    try:
        urllib.request.urlretrieve(poster_url + str(poster_path), unk_poster_dir_path +
                                   str(ids) + ".jpg")
    except IOError as e:
        print('404', e)
    except Exception as e:
        print('404', e)


def download_poster(poster_path, class_name, ids):
    try:
        urllib.request.urlretrieve(poster_url + str(poster_path), poster_dir_path +
                                   str(class_name) + "/" + str(ids) + ".jpg")
    except IOError as e:
        print('404', e)
    except Exception as e:
        print('404', e)


def input_transform():
    return Compose([
        Resize((276, 185)),
        ToTensor(),
        normalize
    ])


def load_dataset():
    train_dataset = torchvision.datasets.ImageFolder(
        root=poster_dir_path,
        transform=input_transform()
    )
    labels = train_dataset.classes
    train_dataset, test_dataset = cut_validation(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True
    )
    img, label = train_dataset.__getitem__(3)
    plt.imshow(img.numpy()[0], cmap='gray')
    print(img.shape)
    plt.show()
    return train_loader, test_loader, labels


def cut_validation(data_to_cut, ratio_of_train=0.9):
    print("Split ratio: ", ratio_of_train)
    train_size = int(ratio_of_train * len(data_to_cut))
    test_size = len(data_to_cut) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data_to_cut, [train_size, test_size])
    return train_dataset, test_dataset


def train(train_loader, test_loader, num_class, early_stop=100, load_model=False):
    best_loss = 999999
    stop_count = 0
    criterion = nn.CrossEntropyLoss()
    net = models.resnet18(pretrained=True)
    if load_model:
        net = net.load_state_dict(torch.load(mode_path))
    net.fc = nn.Linear(512, num_class)
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


def test(net, test_loader):
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if is_gpu:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    return 100 * correct / total


def predict(net, image_path, class_labels):
    image = image_loader(image_path)
    net.eval()
    if is_gpu:
        image = image.cuda()
    outputs = net(image)
    _, predicted = torch.max(outputs, 1)
    return class_labels[predicted.cpu().numpy()[0]]


def fill_genres(net, df, class_labels):
    for i, url in enumerate(df["poster_path"]):
        if len(df["genres"][i]) == 0:
            print(df['title'][i])
            genres = predict(net, unk_poster_dir_path + str(i) + ".jpg", class_labels)
            df["genres"][i] = {"name": genres}
    return df


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = input_transform()(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image  # assumes that you're using GPU


def testing():
    train_df, test_df = preprocess.load_data()
    preprocess.preprocess(train_df)
    num_class = download_all_posters(train_df)
    train_loader, test_loader, class_labels = load_dataset()
    net = train(train_loader, test_loader, num_class)
    df = fill_genres(net, train_df, class_labels)



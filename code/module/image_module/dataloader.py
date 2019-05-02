import torch
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

normalize = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])


def input_transform():
    return Compose([
        Resize((276, 185)),
        ToTensor(),
        normalize
    ])


def get_data_loaders(poster_dir_path, batch_size=32):
    train_dataset = torchvision.datasets.ImageFolder(
        root=poster_dir_path,
        transform=input_transform()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader

from module.image_module.image_handler import ImageHandler
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


def get_dataset(df, dir_name="train"):
    handler = ImageHandler(dir_name)
    handler.download_all_posters(df)
    train_dataset = torchvision.datasets.ImageFolder(
        root=handler.poster_dir_path,
        transform=input_transform()
    )
    return train_dataset

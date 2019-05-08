from module.image_module.image_handler import ImageHandler
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os
import numpy as np
import cv2
import torch.utils.data as data
import torch
from PIL import Image

normalize = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, df, column):
        self.sub_df = df[[column]]

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return input_transform()(self.sub_df.iloc[index, 0])

    def __len__(self):
        return self.sub_df.shape[0]


def input_transform():
    return Compose([
        Resize((276, 185)),
        ToTensor(),
        normalize
    ])


def image2memory(df, handler):

    images = []
    for i in df.id:
        path_to_img = handler.poster_dir_path + str(i) + ".jpeg"
        if not os.path.exists(path_to_img):
            image = Image.fromarray(np.zeros([276, 185, 3], dtype=np.uint8))
        else:
            image = cv2.imread(path_to_img)

        images.append(image)

    df['poster'] = images


def get_dataset(df, handler):
    image2memory(df, handler)
    dataset = Dataset(df, 'poster')
    return dataset

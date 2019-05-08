import torch
from torch import nn
import torchvision.models as models


# Encode Image Feature into image hidden vector
class ImageEncoder(nn.Module):
    def __init__(self, input_size=512, pre_trained_path=None):
        super(ImageEncoder, self).__init__()

        pre_train_net = models.resnet18(pretrained=True)
        if pre_trained_path is not None:
            pre_train_net = self.pre_train_net.load_state_dict(torch.load(pre_trained_path))

        self.output_size = int(input_size ** 0.25)
        pre_train_net.fc = nn.Linear(input_size, self.output_size)

        if torch.cuda.is_available():
            pre_train_net = pre_train_net.cuda()

        self.pre_train_net = pre_train_net

        print("------Image Features Encoder Detail------------")
        print("Res-net Output Dim:", self.output_size)
        print("------------------------------------------------")

    # Model Structure
    # Use Pre-Train CNN to process the image
    def forward(self, x):
        x = torch.stack(x).cuda()
        out = self.pre_train_net(x)
        return out

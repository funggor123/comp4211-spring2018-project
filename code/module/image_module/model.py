import torch
from torch import nn
import torchvision.models as models


# Encode Image Feature into image hidden vector
class ImageEncoder(nn.Module):
    def __init__(self, input_size=512):
        super(ImageEncoder, self).__init__()

        pre_train_net = models.resnet18(pretrained=True)
        fc_features = 512

        #self.output_size = int(fc_features ** 0.25)
        self.output_size = 64

        pre_train_net.fc = nn.Linear(fc_features, self.output_size)

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

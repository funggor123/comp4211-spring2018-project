from module.categorical_module.model import CategoricalEncoder
from module.scalar_module.model import ScalarEncoder
from torch import nn
import torch


class MainModel(nn.Module):
    def __init__(self, categorical_dims, scalar_dims):
        super(MainModel, self).__init__()

        self.categorical_encoder = CategoricalEncoder(categorical_dims)
        self.scalar_encoder = ScalarEncoder(scalar_dims)
        self.encoder_output_dim = self.categorical_encoder.encode_output_dim + self.scalar_encoder.encode_output_dim
        self.l1_out_dim = int(self.encoder_output_dim ** 0.75)

        # https://stackoverflow.com/questions/51052238/loss-increasing-with-batch-normalization-tf-keras
        self.linear = nn.Sequential(
            nn.Linear(self.encoder_output_dim, self.l1_out_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.linear_last = nn.Linear(self.l1_out_dim, 1)

        print("------------Main Model Detail----------")
        print("Encoder input dim :", self.encoder_output_dim)
        print("l1 output dim :", self.l1_out_dim)
        print("---------------------------------------")

    def forward(self, x, hidden=None):
        cat_h = self.categorical_encoder([item[0] for item in x])
        sc_h = self.scalar_encoder([item[1] for item in x])
        out = torch.cat((cat_h, sc_h), dim=1)
        out = self.linear(out)
        out = self.linear_last(out)
        return out


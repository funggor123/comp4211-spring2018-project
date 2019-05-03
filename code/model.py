from module.categorical_module.model import CategoricalEncoder
from module.scalar_module.model import ScalarEncoder
from torch import nn
import torch


class MainModel(nn.Module):
    def __init__(self, categorical_dims, scalar_dims):
        super(MainModel, self).__init__()

        self.categorical_encoder = CategoricalEncoder(categorical_dims)
        self.scalar_encoder = ScalarEncoder(scalar_dims)
        self.input_dims = self.categorical_encoder.output_dims + self.scalar_encoder.output_dims
        self.linear = nn.Sequential(
            nn.Linear(self.input_dims, int(self.input_dims ** 0.25)),
            nn.ReLU()
        )
        self.linear_last = nn.Linear(int(self.input_dims ** 0.25), 1)

    def forward(self, x, hidden=None):
        cat_h = self.categorical_encoder([item[0] for item in x])
        sc_h = self.scalar_encoder([item[1] for item in x])
        out = torch.cat(cat_h, sc_h)
        out = self.linear(out)
        out = self.linear_last(out)
        return out


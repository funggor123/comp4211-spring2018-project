from torch import nn
import torch


# Encode all Scalar Features into linear hidden vector
class ScalarEncoder(nn.Module):
    def __init__(self, encode_input_dim):
        super(ScalarEncoder, self).__init__()
        self.encode_output_dim = int(encode_input_dim ** 0.75)
        self.linear = nn.Sequential(
            nn.Linear(encode_input_dim, self.encode_output_dim),
            nn.ReLU6(),
            nn.BatchNorm1d(self.encode_output_dim),
            nn.Dropout(0.25)
        )
        print("------Scalar Network Detail------------")
        print("Encode Input dim :", encode_input_dim)
        print("Encode Output dim :", self.encode_output_dim)
        print("---------------------------------------")

    # Model Structure
    # linear
    def forward(self, x):
        x = torch.stack(x).cuda()
        scalar_h = self.linear(x)
        return scalar_h

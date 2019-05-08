from torch import nn
import torch


# Encode all Scalar Features into linear hidden vector
class ScalarEncoder(nn.Module):
    def __init__(self, encode_input_dim):
        super(ScalarEncoder, self).__init__()
        self.encode_output_dim = int(encode_input_dim ** 1.25)
        self.linear = nn.Sequential(
            nn.Linear(encode_input_dim, self.encode_output_dim),
            nn.RReLU(),
            nn.BatchNorm1d(self.encode_output_dim),
            #nn.Dropout(self.drop_rate)
        )
        print("------Scalar Features Encoder Detail------------")
        print("Linear Layer Input dim :", encode_input_dim)
        print("Linear Layer Output dim :", self.encode_output_dim)
        print("------------------------------------------------")

    # Model Structure
    # linear
    def forward(self, x):
        x = torch.stack(x).cuda()
        scalar_h = self.linear(x)
        return scalar_h

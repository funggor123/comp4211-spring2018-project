from torch import nn


# Encode all Scalar Features into linear hidden vector
class ScalarEncoder(nn.Module):
    def __init__(self, input_dims):
        super(ScalarEncoder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dims, int(sum(input_dims) ** 0.25)),
            nn.ReLU6
        )

    # Model Structure
    # linear
    def forward(self, x, hidden=None):
        scalar_h = self.linear(x)
        return scalar_h

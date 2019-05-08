from module.categorical_module.model import CategoricalEncoder
from module.scalar_module.model import ScalarEncoder
from module.text_module.model import BiLSTMAttention
from module.image_module.model import ImageEncoder
from torch import nn
import torch


class MainModel(nn.Module):
    def __init__(self, categorical_dims, scalar_dims, embedding_matrix=None):
        super(MainModel, self).__init__()

        self.categorical_encoder = CategoricalEncoder(categorical_dims)
        self.scalar_encoder = ScalarEncoder(scalar_dims)

        self.overview_encoder = BiLSTMAttention(embedding_matrix=embedding_matrix)
        self.tagline_encoder = BiLSTMAttention(embedding_matrix=embedding_matrix)
        self.title_encoder = BiLSTMAttention(embedding_matrix=embedding_matrix)

        self.image_encoder = ImageEncoder()

        self.encoder_output_dim = self.categorical_encoder.encode_output_dim + self.scalar_encoder.encode_output_dim + \
                                  self.overview_encoder.output_size + self.tagline_encoder.output_size + self.title_encoder.output_size + \
                                  self.image_encoder.output_size
        self.l1_out_dim = int(self.encoder_output_dim ** 1.25)
        self.l2_out_dim = int(self.l1_out_dim ** 0.75)

        # https://stackoverflow.com/questions/51052238/loss-increasing-with-batch-normalization-tf-keras
        self.linear1 = nn.Sequential(
            nn.Linear(self.encoder_output_dim, self.l1_out_dim),
            nn.RReLU(),
            nn.BatchNorm1d(self.l1_out_dim)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(self.l1_out_dim, self.l2_out_dim),
            nn.RReLU(),
        )

        self.linear_last = nn.Linear(self.l2_out_dim, 1)

        print("------------Main Model Detail----------")
        print("Encoder input dim :", self.encoder_output_dim)
        print("l1 output dim :", self.l1_out_dim)
        print("---------------------------------------")

    def forward(self, x, hidden=None):
        cat_h = self.categorical_encoder([item[0] for item in x])
        sc_h = self.scalar_encoder([item[1] for item in x])

        overview_h = self.overview_encoder([item[2] for item in x])
        tagline_h = self.overview_encoder([item[3] for item in x])
        title_h = self.title_encoder([item[4] for item in x])

        poster_h = self.image_encoder([item[5] for item in x])

        out = torch.cat((cat_h, sc_h, overview_h, tagline_h, title_h, poster_h), dim=1)
        out = self.linear(out)
        out = self.linear2(out)
        out = self.linear_last(out)
        return out

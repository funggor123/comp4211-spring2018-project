import torch.nn.functional as F
import torch
from torch import nn


# Encode all Categorical Features into a categorical hidden vector
class CategoricalEncoder(nn.Module):
    def __init__(self, input_dims):
        super(CategoricalEncoder, self).__init__()

        # https://datascience.stackexchange.com/questions/31109/ratio-between-embedded-vector-dimensions-and-vocabulary-size
        embedding_dims = [int(dim ** 0.25) for dim in input_dims]

        assert len(input_dims) == len(embedding_dims)

        # https://www.itread01.com/content/1543647544.html
        self.embeddings = nn.ModuleList([nn.Embedding(input_dims[i], embedding_dims[i]) for i in range(len(input_dims))])
        self.attention_linear = nn.ModuleList([nn.Linear(input_dims[i], 1) for i in range(len(input_dims))])
        self.encode_linear = nn.Sequential(
            nn.Linear(self.total_num_ebd_size, int(sum(embedding_dims) ** 0.25)),
            nn.ReLU6
        )

    # Model Structure
    # Cat ([Embedding -> Attention]) -> Linear
    def forward(self, x, hidden=None):
        assert len(x) == len(self.embeddings)
        feature_h = []

        for i, raw in enumerate(x):
            embedding = self.embeddings[i](raw)
            feature_h += self.attention(embedding, self.attention_linear[i])

        cat_feature_h = torch.cat(feature_h)
        categorical_h = self.encode_linear(cat_feature_h)
        return categorical_h

    @staticmethod
    def attention(x, attention_linear):
        x = F.tanh(x)
        e = attention_linear(x)
        a = F.softmax(e)
        out = torch.sum(torch.mul(x, a), 1)
        return out

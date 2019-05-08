import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils


# Encode all Categorical Features into a categorical hidden vector
class CategoricalEncoder(nn.Module):
    def __init__(self, input_dims):
        super(CategoricalEncoder, self).__init__()

        # https://datascience.stackexchange.com/questions/31109/ratio-between-embedded-vector-dimensions-and-vocabulary-size
        self.embedding_dims = [int(dim ** 0.25) for dim in input_dims]
        self.feature_linear_dim = [int(dim ** 1.25) for dim in self.embedding_dims]
        self.encode_input_dim = sum(self.feature_linear_dim)
        self.encode_output_dim = int(self.encode_input_dim ** 1.25)
        self.drop_rate = 0.25

        assert len(input_dims) == len(self.embedding_dims)

        # https://www.itread01.com/content/1543647544.html
        self.embeddings = nn.ModuleList(
            [nn.Embedding(input_dims[i], self.embedding_dims[i]) for i in range(len(input_dims))])
        self.attention_linear = nn.ModuleList([nn.Linear(self.embedding_dims[i], 1) for i in range(len(input_dims))])
        self.feature_linear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dims[i], self.feature_linear_dim[i]),
                nn.ELU(),
                nn.BatchNorm1d(self.feature_linear_dim[i]),
                #nn.Dropout(self.drop_rate)
            ) for i in range(len(input_dims))])

        self.encode_linear = nn.Sequential(
            nn.Linear(self.encode_input_dim, self.encode_output_dim),
            nn.ELU(),
            nn.BatchNorm1d(self.encode_output_dim),
            #nn.Dropout(self.drop_rate)
        )
        print("------Categorical Network Detail-------")
        print("Embedding dim :", self.embedding_dims)
        print("Feature Output dim :", self.feature_linear_dim)
        print("Encode Input dim :", self.encode_input_dim)
        print("Encode Output dim :", self.encode_output_dim)
        print("---------------------------------------")

    # Model Structure
    # Cat ([Embedding -> Attention]) -> Linear
    # https://nlp.stanford.edu/pubs/emnlp15_attn.pdf
    def forward(self, x, hidden=None):
        assert len(x[0]) == len(self.embeddings)
        feature_h = []

        for i in range(len(x[0])):
            raw = [item[i] for item in x]
            raw = rnn_utils.pad_sequence(raw, batch_first=True).cuda()
            embedding = self.embeddings[i](raw)
            feature_h.append(self.feature_linear[i](self.attention(embedding, self.attention_linear[i])))

        cat_feature_h = torch.cat(feature_h, dim=1)
        categorical_h = self.encode_linear(cat_feature_h)
        return categorical_h

    @staticmethod
    def attention(x, attention_linear):
        x = torch.tanh(x)
        e = attention_linear(x)
        a = F.softmax(e, dim=1)
        out = torch.sum(torch.mul(x, a), 1)
        return out

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
from torch import nn


class BiLSTMAttention(nn.Module):
    def __init__(self, num_class, embedding_dim=100, hidden_size=64, embedding_matrix=None):
        super(BiLSTMAttention, self).__init__()

        if torch.cuda.is_available():
            embedding_matrix_tensor = torch.FloatTensor(embedding_matrix).cuda()
        else:
            embedding_matrix_tensor = torch.FloatTensor(embedding_matrix)

        if embedding_matrix is not None:
            embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix_tensor)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.linear_last = nn.Linear(hidden_size, num_class)
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        ebd = self.embedding(x)
        out, hidden = self.lstm(ebd, hidden)
        out = self.attention(out)
        out = self.linear_last(out)
        return out

    def attention(self, x):
        hidden_states = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:]
        hidden_states = F.tanh(hidden_states)
        e = self.linear(hidden_states)
        a = F.softmax(e)
        out = torch.sum(torch.mul(hidden_states, a), 1)
        return out

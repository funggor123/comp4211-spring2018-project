import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils


# Encode the Text into a context vector
# https://www.aclweb.org/anthology/P16-2034
class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_dim=128, hidden_size=128, embedding_matrix=None):
        super(BiLSTMAttention, self).__init__()

        if torch.cuda.is_available():
            embedding_matrix_tensor = torch.FloatTensor(embedding_matrix).cuda()
        else:
            embedding_matrix_tensor = torch.FloatTensor(embedding_matrix)

        if embedding_matrix is not None:
            embedding_dim = embedding_matrix.shape[1]

        self.hidden_size = hidden_size
        self.output_size = int(self.hidden_size ** 1.25)

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix_tensor)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True)
        self.attention_linear = nn.Linear(hidden_size, 1)
        self.linear_hidden = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.RReLU(),
            nn.BatchNorm1d(self.output_size),
            # nn.Dropout(self.drop_rate)
        )

        print("------Text Features Encoder Detail------------")
        print("Embedding Layer Output Dim :", embedding_dim)
        print("LSTM Hidden Dim :", self.hidden_size)
        print("Linear Layer Output Dim :", self.output_size)
        print("------------------------------------------------")

    # Model Structure
    # Embedding -> LSTM -> Attention -> Linear
    def forward(self, x, hidden=None):
        x = rnn_utils.pad_sequence(x, batch_first=True).cuda()
        embedding = self.embedding(x)
        out, hidden = self.lstm(embedding, hidden)
        out = self.attention(out, self.attention_linear)
        context_vector = self.linear_hidden(out)
        return context_vector

    def attention(self, x, attention_linear):
        hidden_states = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:]
        hidden_states = torch.tanh(hidden_states)
        e = attention_linear(hidden_states)
        a = F.softmax(e, dim=1)
        out = torch.sum(torch.mul(hidden_states, a), 1)
        return out

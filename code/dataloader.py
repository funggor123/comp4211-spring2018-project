import pandas as pd
import torch
import torch.utils.data as data
from preprocess_text import preprocess
import gensim


UNK_INDEX = 0


class Vocab():
    def __init__(self):
        self.word2Vector = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                           binary=True, limit=100000)
        self.no_of_vocab = len(self.word2Vector.wv.vectors)
        self.embeddings_matrix = self.word2Vector.wv.vectors


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, sent_in, sent_out, vocab):
        self.X = sent_in
        self.y = sent_out
        self.vocab = vocab
        self.num_total_seqs = len(self.X)
        if self.y is not None: self.y = torch.LongTensor(self.y)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        X = self.tokenize(self.X[index])
        if self.y is not None:
            return torch.LongTensor(X), self.y[index]
        else:
            return torch.LongTensor(X)

    def __len__(self):
        return self.num_total_seqs

    def tokenize(self, sentence):
        return [self.vocab.word2Vector.vocab.get(word).index if word in self.vocab.word2Vector.vocab else
                self.vocab.word2Vector.vocab.get("0").index for word in sentence]


def get_dataloaders(batch_size, df):
    vocab = Vocab()
    train_data_sent_in, train_data_sent_out = preprocess(df)
    train = Dataset(train_data_sent_in, train_data_sent_out, vocab)
    data_loader_tr = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 collate_fn=collate_fn
                                                 )
    return data_loader_tr, vocab.embeddings_matrix, vocab.no_of_vocab


def collate_fn(data):
    batch_size = len(data)
    splited_batch = list(zip(*data))

    seqs = splited_batch[0]

    max_length = max([len(seq) for seq in seqs])
    padded_seqs_tensor = torch.LongTensor([])

    for i in range(batch_size):
        number_of_pad = max_length - len(seqs[i])
        cat_tensor = torch.cat([seqs[i], torch.zeros(number_of_pad).type_as(seqs[i])], dim=0)
        temp_tensor = torch.LongTensor(cat_tensor)
        padded_seqs_tensor = torch.cat([padded_seqs_tensor, temp_tensor.view(1, max_length)], dim=0)
        if i == 0:
            padded_seqs_tensor = padded_seqs_tensor.view(1, max_length)

    splited_batch = [torch.LongTensor(splited_batch[i]) for i in range(1, len(splited_batch))]
    splited_batch.insert(0, padded_seqs_tensor)
    return splited_batch

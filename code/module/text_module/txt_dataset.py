import torch
import torch.utils.data as data
import gensim
from module.text_module.preprocessor import Preprocessor

UNK_INDEX = 0


class Vocab:
    def __init__(self, vocab_size=100000):
        self.word2Vector = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                           binary=True, limit=vocab_size)
        self.no_of_vocab = len(self.word2Vector.wv.vectors)
        self.embeddings_matrix = self.word2Vector.wv.vectors


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, X, vocab):
        self.X = X
        self.vocab = vocab

    def __getitem__(self, index):
        x = self.tokenize(self.X[index])
        return torch.LongTensor(x)

    def __len__(self):
        return len(self.X)

    def tokenize(self, sentence):
        return [self.vocab.word2Vector.vocab.get(word).index if word in self.vocab.word2Vector.vocab else
                self.vocab.word2Vector.vocab.get("0").index for word in sentence]


def get_dataset(df, column, vocab):
    preprocessor = Preprocessor()
    train_data = preprocessor.preprocess(df, column)
    train_data = Dataset(train_data, vocab)
    return train_data


def getVocab():
    vocab = Vocab()
    return vocab

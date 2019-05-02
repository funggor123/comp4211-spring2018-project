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


def get_data_loaders(batch_size, df, column):
    vocab = Vocab()
    preprocessor = Preprocessor()
    train_data = preprocessor.preprocess(df, column)
    train_data = Dataset(train_data, vocab)
    data_loader_tr = torch.utils.data.DataLoader(dataset=train_data,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 collate_fn=collate_fn
                                                 )
    return data_loader_tr, vocab.embeddings_matrix, vocab.no_of_vocab


def collate_fn(input_data):
    batch_size = len(input_data)
    split_batch = list(zip(*input_data))

    seqs = split_batch[0]

    max_length = max([len(seq) for seq in seqs])
    padded_seqs_tensor = torch.LongTensor([])

    for i in range(batch_size):
        number_of_pad = max_length - len(seqs[i])
        cat_tensor = torch.cat([seqs[i], torch.zeros(number_of_pad).type_as(seqs[i])], dim=0)
        temp_tensor = torch.LongTensor(cat_tensor)
        padded_seqs_tensor = torch.cat([padded_seqs_tensor, temp_tensor.view(1, max_length)], dim=0)
        if i == 0:
            padded_seqs_tensor = padded_seqs_tensor.view(1, max_length)

    split_batch = [torch.LongTensor(split_batch[i]) for i in range(1, len(split_batch))]
    split_batch.insert(0, padded_seqs_tensor)
    return split_batch

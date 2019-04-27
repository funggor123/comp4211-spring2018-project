import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import torch
from random import randint

nltk.download('stopwords')
stop = stopwords.words('english')
sno = PorterStemmer()
wnl = WordNetLemmatizer()


def clean_str(string):
    string = lower_string(string)
    string = expand_contradiction(string)
    string = clean_sp_char(string)
    return string


def stem_and_lemma(data):
    # stem is too slow
    # data = [sno.stem(w) for w in data]
    data = [wnl.lemmatize(w) for w in data]
    return data


def lower_string(string):
    return string.strip().lower()


def stop_word_filtering(data):
    data = [w for w in data if not w in stop]
    return data


def expand_contradiction(string):
    # Expand the Contradiction
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, s)

    string = expand_contractions(string)
    return string


contractions_dict = {
    "won't": "were not",
    "you'll": "you will",
    "we're": "we are",
    "that's": "that is",
    "were't": "were not",
    "i'd": "i do not",
    "i'll": "i will",
    "there's": "there is",
    "they'll": "they will",
    "it's": "it is",
    "they're": "they are",
    "i've": "i have",
    "we'll": "we will",
    "she's": "she is",
    "could": "could have",
    "we've": "we have",
    "you'd": "you don't",
    "you're": "you are",
    "they've": "they have",
    "shouldn't": "should not",
    "he's": "he is ",
    "should ve": "should have",
    "could've": "could have",
    "couldn't've": "could not have",
    "did n't": "did not",
    "do n't": "do not",
    "had n't": "had not",
    "had n't've": "had not have",
    "has n't": "has not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "should've": "should have",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "there'd": "here would",
    "there'd've": "there would have",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll've": "they will have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll've": "we will have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd've": "you would have",
    "you'll've": "you will have",
    "you've": "you have",
    "n't": "not",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "isn't": "is not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "i'm": "i am",
}


def clean_sp_char(string):
    return re.sub(r"[^A-Za-z]", " ", string)


def preprocess(df, test=False):
    x = []
    y = []
    for i, overview in enumerate(df["overview"]):
        overview = clean_str(str(overview))
        overview = overview.split()
        x += [overview]
        y += [randint(0, 9)]
        # df["genres"][0]["name"]

    if test:
        return x, None
    else:
        assert len(x) == len(y)
        return x, y



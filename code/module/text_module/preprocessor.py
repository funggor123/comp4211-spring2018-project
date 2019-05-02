import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import unicodedata
nltk.download('stopwords')


class Preprocessor:

    def __init__(self):
        self.contractions_dict = {
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
        self.stopwords = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, df, column):
        x = []
        for row in df[column]:
            row = self.clean_str(str(row))
            row = row.split()
            x += [row]
        return x

    # Clean the string
    def clean_str(self, string):
        string = self.unicodeToAscii(string)
        string = self.lowercase(string)
        string = self.expand_contradiction(string)
        string = self.special_character_filtering(string)
        string = self.stopwords_filtering(string)
        string = self.root_extracting(string)
        return string

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    @staticmethod
    def unicodeToAscii(string):
        return ''.join(
            c for c in unicodedata.normalize('NFD', string)
            if unicodedata.category(c) != 'Mn'
        )

    # Extract to root
    def root_extracting(self, string):
        string = self.stemmer.stem(string)
        string = self.lemmatizer.lemmatize(string)
        return string

    # Convert to lowercase
    @staticmethod
    def lowercase(string):
        return string.strip().lower()

    # Filter the stopwords
    @staticmethod
    def stopwords_filtering(self, data):
        data = [w for w in data if w not in self.stopwords]
        return data

    # Expand the contradiction
    def expand_contradiction(self, string):
        # Expand the Contradiction
        contractions_re = re.compile('(%s)' % '|'.join(self.contractions_dict.keys()))

        def expand_contractions(s, contractions_dict):
            def replace(match):
                return contractions_dict[match.group(0)]

            return contractions_re.sub(replace, s)

        string = expand_contractions(string, self.contractions_dict)
        return string

    # Filter the special characters
    @staticmethod
    def special_character_filtering(string):
        string = re.sub(r"\[([^\]]+)\]", " ", string)
        string = re.sub(r"\(([^\)]+)\)", " ", string)
        string = re.sub(r"[^A-Za-z0-9,!?.;]", " ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string

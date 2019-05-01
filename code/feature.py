import numpy as np
from datetime import datetime
from cnn_genres import download_all_posters
from cnn_genres import fill_genres
from cnn_genres import train
from cnn_genres import load_dataset
from sklearn import preprocessing


def feature(df):
    feature_selection(df)
    fill_missing_and_extract(df)
    feature_transform(df)
    feature_encoding(df)


def feature_transform(df):
    log_transform(df, 'revenue')
    log_transform(df, 'budget')
    log_transform(df, 'popularity')
    log_transform(df, 'runtime')
    normalize(df, 'log_revenue')
    normalize(df, 'log_budget')
    normalize(df, 'log_popularity')
    normalize(df, 'log_runtime')


def normalize(df, column_name):
    df['norm_' + column_name] = preprocessing.scale(df[column_name])


def feature_selection(df):
    extract_date(df)

    # Drop constant value, unique value, uncorrelated value
    df.drop(['homepage', 'imdb_id', 'original_title', 'status'], axis=1)


def fill_missing_and_extract(df):
    # Fill Genres
    '''
    num_class = download_all_posters(df)
    train_loader, test_loader, class_labels = load_dataset()
    net = train(train_loader, test_loader, num_class)
    fill_genres(net, df, class_labels)
    '''

    df["belongs_to_collection"] = df["belongs_to_collection"].apply(lambda x: x[0]["name"] if x != {} else 'UNK')
    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    df['production_companies'] = df['production_companies'].apply(
        lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
    df['production_countries'] = df['production_countries'].apply(
        lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
    df['spoken_languages'] = df['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
    df['Keywords'] = df['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
    df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
    df['crew'] = df['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
    df['tagline'] = df['tagline'].fillna('UNK')
    df['overview'] = df['overview'].fillna('UNK')
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())

def feature_encoding(df):
    cat_to_label(df)


def log_transform(df, column_name):
    df['log_' + column_name] = np.log1p(df[column_name])


def cat_to_label(df):

    # Belongs to Collection
    # Embedding -> Dense
    le = preprocessing.LabelEncoder()
    le.fit(df["belongs_to_collection"].value_counts().index)
    df["belongs_to_collection"] = le.transform(df["belongs_to_collection"])

    # Genres
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    le.fit(list(set([i for j in df['genres'] for i in j])))
    df['genres'] = df['genres'].apply(lambda x: le.transform(x))

    # Production Company
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    le.fit(list(set([i for j in df['production_companies'] for i in j])))
    df['production_companies'] = df['production_companies'].apply(lambda x: le.transform(x))

    # Production Counties
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    le.fit(list(set([i for j in df['production_countries'] for i in j])))
    df['production_countries'] = df['production_countries'].apply(lambda x: le.transform(x))

    # Spoken Language
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    le.fit(list(set([i for j in df['spoken_languages'] for i in j])))
    df['spoken_languages'] = df['spoken_languages'].apply(lambda x: le.transform(x))

    # Keywords
    # LSTM -> Hidden Layer -> Attention

    # Cast
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    le.fit(list(set([i for j in df['cast'] for i in j])))
    df['cast'] = df['cast'].apply(lambda x: le.transform(x))

    # Crew
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    le.fit(list(set([i for j in df['crew'] for i in j])))
    df['crew'] = df['crew'].apply(lambda x: le.transform(x))


def date(x):
    x = str(x)
    year = x.split('/')[2]
    if int(year) < 19:
        return x[:-2] + '20' + year
    else:
        return x[:-2] + '19' + year


def extract_date(df):
    df['release_date'] = df['release_date'].fillna('1/1/90').apply(lambda x: date(x))
    df['release_date'] = df['release_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
    df['release_day'] = df['release_date'].apply(lambda x: x.weekday())
    df['release_month'] = df['release_date'].apply(lambda x: x.month)
    df['release_year'] = df['release_date'].apply(lambda x: x.year)

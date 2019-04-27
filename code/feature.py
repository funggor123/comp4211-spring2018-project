import numpy as np
from datetime import datetime
from cnn_genres import download_all_posters
from cnn_genres import fill_genres
from cnn_genres import train
from cnn_genres import load_dataset
from sklearn import preprocessing


def feature(df):
    feature_transform(df)
    feature_selection(df)
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


def fill_missing_value(df):
    # Fill Genres
    num_class = download_all_posters(df)
    train_loader, test_loader, class_labels = load_dataset()
    net = train(train_loader, test_loader, num_class)
    fill_genres(net, df, class_labels)

    #


def feature_encoding(df):
    cat_to_label(df)


def log_transform(df, column_name):
    df['log_' + column_name] = np.log1p(df[column_name])


def cat_to_label(df):
    # Encode belongs to collection, create collection for each (one value)
    # Embedding -> Dense
    le = preprocessing.LabelEncoder()
    collections = df["belongs_to_collection"].apply(lambda x: x[0]["name"] if x != {} else '?')
    le.fit(collections.value_counts().index)
    df["belongs_to_collection"] = le.transform(collections)

    # Encode original language to labels [original lang can be treated as the production place] (one value)
    # Embedding -> Dense
    le = preprocessing.LabelEncoder()
    origin_langs = df["original_language"].value_counts().index
    le.fit(origin_langs)
    df["original_language"] = le.transform(df["original_language"])

    # Encode Genres (A list of value)
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    genre = df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    le.fit(list(set([i for j in genre for i in j])))
    df['genres'] = genre.apply(lambda x: le.transform(x))

    # Production Company (A list of value)
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    genre = df['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    le.fit(list(set([i for j in genre for i in j])))
    df['production_companies'] = genre.apply(lambda x: le.transform(x))

    # Production Counties (A list of value)
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    genre = df['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    le.fit(list(set([i for j in genre for i in j])))
    df['production_countries'] = genre.apply(lambda x: le.transform(x))

    # Spoken Language (A list of value)
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    genre = df['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    le.fit(list(set([i for j in genre for i in j])))
    df['spoken_languages'] = genre.apply(lambda x: le.transform(x))

    # Keywords
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    keywords = df['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else [])

    # Cast (A list of value)
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    genre = df['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    le.fit(list(set([i for j in genre for i in j])))
    df['cast'] = genre.apply(lambda x: le.transform(x))

    # Crew (A list of value)
    # Embedding -> Convolution
    le = preprocessing.LabelEncoder()
    genre = df['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    le.fit(list(set([i for j in genre for i in j])))
    df['crew'] = genre.apply(lambda x: le.transform(x))


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

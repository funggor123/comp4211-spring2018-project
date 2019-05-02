import pandas as pd

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']


def load_data():
    return pd.read_csv('./train.csv'), pd.read_csv('./test.csv')


def preprocess(df):
    text_to_dict(df, dict_columns)


def text_to_dict(df, column_name):
    for column in column_name:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else eval(x))
    return df

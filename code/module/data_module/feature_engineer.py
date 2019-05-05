import numpy as np
from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# Engine to Perform Feature Engineering
class FeatureEngine:

    def __init__(self):
        self.categorical_dims = []
        self.categorical_columns = []

        self.scalar_columns = []
        self.y_column = 'norm_log_revenue'
        self.y_scalar = None

    def run(self, df):
        self.feature_selection(df)
        self.feature_cleaning(df)
        self.feature_encoding(df)
        self.feature_transform(df)

    def feature_transform(self, df):

        # Transform to Normal Dist
        self.log_transform(df, 'revenue')
        self.log_transform(df, 'budget')
        self.log_transform(df, 'popularity')
        self.log_transform(df, 'runtime')

        # Normalize the Feature
        self.normalize(df, 'log_revenue', add=False)
        self.normalize(df, 'log_budget')
        self.normalize(df, 'log_popularity')
        self.normalize(df, 'log_runtime')

        # Transform the Recursive Feature
        self.encode_recursive_feature(df, 'release_day', 6)
        self.encode_recursive_feature(df, 'release_month', 11)

    def encode_recursive_feature(self, df, column_name, max_limit):
        df['cos_' + column_name] = np.sin(2 * np.pi * df[column_name] / max_limit)
        df['sin_' + column_name] = np.cos(2 * np.pi * df[column_name] / max_limit)
        self.scalar_columns += ['cos_' + column_name]
        self.scalar_columns += ['sin_' + column_name]

    def normalize(self, df, column_name, add=True):
        # Create a minimum and maximum processor object
        scalar = preprocessing.MinMaxScaler()
        df['norm_' + column_name] = scalar.fit_transform(df[column_name].values.reshape(-1, 1))
        if add:
            self.scalar_columns += ['norm_' + column_name]
        else:
            self.y_scalar = scalar

    @staticmethod
    def log_transform(df, column_name):
        df['log_' + column_name] = np.log1p(df[column_name])

    def feature_selection(self, df):

        # Extract New Features
        self.extract_date(df)

        # Drop Unused Features
        # Drop constant value, unique value, uncorrelated value
        df.drop(['homepage', 'imdb_id', 'original_title', 'status'], axis=1, inplace=True)

    @staticmethod
    def correct_year(x):
        x = str(x)
        year = x.split('/')[2]
        if int(year) < 19:
            return x[:-2] + '20' + year
        else:
            return x[:-2] + '19' + year

    def extract_date(self, df):
        df['release_date'] = df['release_date'].fillna('1/1/90').apply(lambda x: self.correct_year(x))
        df['release_date'] = df['release_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
        df['release_day'] = df['release_date'].apply(lambda x: x.weekday())
        df['release_month'] = df['release_date'].apply(lambda x: x.month)
        df['release_year'] = df['release_date'].apply(lambda x: x.year)

    @staticmethod
    def feature_cleaning(df):

        # Fill Missing Value and Clean the data
        '''
        num_class = download_all_posters(df)
        train_loader, test_loader, class_labels = load_dataset()
        net = train(train_loader, test_loader, num_class)
        fill_genres(net, df, class_labels)
        '''

        df["belongs_to_collection"] = df["belongs_to_collection"].apply(lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
        df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
        df['production_companies'] = df["production_companies"].apply(
        lambda x: [x[i]["name"] for i in range(len(x))] if x != {} else ['UNK']).values
        df['production_countries'] = df['production_countries'].apply(
            lambda x: [i['name'] for i in x] if x != {} else ['!'])
        df['spoken_languages'] = df['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
        df['Keywords'] = df['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
        df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
        df['crew'] = df['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else ['UNK'])
        df['tagline'] = df['tagline'].fillna('UNK')
        df['overview'] = df['overview'].fillna('UNK')
        df['runtime'] = df['runtime'].fillna(df['runtime'].median())

    def feature_encoding(self, df):
        # Encode the categorical feature into index
        self.encode_categorical_to_index(df)

    def encode_categorical_to_index(self, df):

        # Belongs to Collection
        le = preprocessing.LabelEncoder()
        le.fit(['!'] + list(set([i for j in df['belongs_to_collection'] for i in j])))
        df["belongs_to_collection"] = df["belongs_to_collection"].apply(lambda x: le.transform(x))
        self.categorical_dims.append(len(le.classes_))
        self.categorical_columns += ["belongs_to_collection"]

        # Genres
        le = preprocessing.LabelEncoder()
        le.fit(['!'] +list(set([i for j in df['genres'] for i in j])))
        df['genres'] = df['genres'].apply(lambda x: le.transform(x))
        self.categorical_dims.append(len(le.classes_))
        self.categorical_columns += ["genres"]

        # Production Company
        le = preprocessing.LabelEncoder()

        le.fit(['!'] +list(set([i for j in df['production_companies'] for i in j])))
        df['production_companies'] = df['production_companies'].apply(lambda x: le.transform(x))
        self.categorical_dims.append(len(le.classes_))
        self.categorical_columns += ["production_companies"]

        # Production Counties
        le = preprocessing.LabelEncoder()
        le.fit(['!'] +list(set([i for j in df['production_countries'] for i in j])))
        df['production_countries'] = df['production_countries'].apply(lambda x: le.transform(x))
        self.categorical_dims.append(len(le.classes_))
        self.categorical_columns += ["production_countries"]

        # Spoken Language
        le = preprocessing.LabelEncoder()
        le.fit(['!'] + list(set([i for j in df['spoken_languages'] for i in j])))
        df['spoken_languages'] = df['spoken_languages'].apply(lambda x: le.transform(x))
        self.categorical_dims.append(len(le.classes_))
        self.categorical_columns += ['spoken_languages']

        # Cast
        le = preprocessing.LabelEncoder()
        le.fit(['!'] + list(set([i for j in df['cast'] for i in j])))
        df['cast'] = df['cast'].apply(lambda x: le.transform(x))
        self.categorical_dims.append(len(le.classes_))
        self.categorical_columns += ['cast']

        # Crew
        le = preprocessing.LabelEncoder()
        le.fit(['!'] + list(set([i for j in df['crew'] for i in j])))
        df['crew'] = df['crew'].apply(lambda x: le.transform(x))
        self.categorical_dims.append(len(le.classes_))
        self.categorical_columns += ['crew']


import numpy as np
from datetime import datetime
from sklearn import preprocessing


# Engine to Perform Feature Engineering and Data PreProcessing
class FeatureEngine:

    def __init__(self):
        self.categorical_dims = []
        self.categorical_columns = []
        self.cat_encoder = []

        self.scalar_columns = []
        self.scalar_scalar = []

        self.y_column = 'norm_log_revenue'
        self.y_scalar = None

        self.text_column = ['overview', 'tagline', 'title']

    def transform_and_fit(self, df):
        self.feature_selection(df)
        self.feature_cleaning(df)
        self.feature_encoding(df)
        self.feature_transform(df)

    def transform(self, df):
        self.feature_selection(df)
        self.feature_cleaning(df)
        self.feature_encoding(df, fit=True)
        self.feature_transform(df, fit=True, test=True)

    def feature_transform(self, df, fit=False, test=False):

        # Transform to Normal Dist
        if not test:
            self.log_transform(df, 'revenue')
        self.log_transform(df, 'budget')
        self.log_transform(df, 'popularity')
        self.log_transform(df, 'runtime')

        # Normalize the Feature
        if not test:
            self.normalize(df, 'log_revenue', target=True, fit=fit)
        self.normalize(df, 'log_budget', fit=fit)
        self.normalize(df, 'log_popularity', fit=fit)
        self.normalize(df, 'log_runtime', fit=fit)

        self.normalize(df, 'production_countries_count', fit=fit)
        self.normalize(df, 'production_companies_count', fit=fit)
        self.normalize(df, 'cast_count', fit=fit)
        self.normalize(df, 'crew_count', fit=fit)
        self.normalize(df, 'num_Keywords', fit=fit)

        self.normalize(df, 'popularity_mean_year', fit=fit)
        self.normalize(df, 'budget_runtime_ratio', fit=fit)
        self.normalize(df, 'budget_popularity_ratio', fit=fit)
        self.normalize(df, 'budget_year_ratio', fit=fit)
        self.normalize(df, 'releaseYear_popularity_ratio', fit=fit)
        self.normalize(df, 'releaseYear_popularity_ratio2', fit=fit)
        self.normalize(df, 'inflationBudget', fit=fit)

        # Transform the Recursive Feature
        self.encode_recursive_feature(df, 'release_day', 6, fit=fit)
        self.encode_recursive_feature(df, 'release_month', 11, fit=fit)

    def encode_recursive_feature(self, df, column_name, max_limit, fit):
        df['cos_' + column_name] = np.sin(2 * np.pi * df[column_name] / max_limit)
        df['sin_' + column_name] = np.cos(2 * np.pi * df[column_name] / max_limit)

        self.normalize(df, 'sin_' + column_name, fit=fit)
        self.normalize(df, 'cos_' + column_name, fit=fit)

    def normalize(self, df, column_name, fit, target=False):

        if not fit:
            scalar = preprocessing.StandardScaler()
            df['norm_' + column_name] = scalar.fit_transform(df[column_name].values.reshape(-1, 1))
            if target:
                self.y_scalar = scalar
            else:
                self.scalar_scalar += [scalar]
                self.scalar_columns += ['norm_' + column_name]
        else:
            if target:
                df['norm_' + column_name] = self.y_scalar.transform(df[column_name].values.reshape(-1, 1))
            else:
                i = self.scalar_columns.index('norm_' + column_name)
                df['norm_' + column_name] = self.scalar_scalar[i].transform(df[column_name].values.reshape(-1, 1))

    @staticmethod
    def log_transform(df, column_name):
        df['log_' + column_name] = np.log1p(df[column_name])

    def feature_selection(self, df):

        # Extract New Features
        self.extract_date(df)

        # Fill Missing Runtime
        df['runtime'] = df['runtime'].fillna(df['runtime'].median())

        # Extract New Features

        df['popularity_mean_year'] = df['popularity'] / df.groupby("release_year")["popularity"].transform(
            'mean')
        df['budget_runtime_ratio'] = df['budget'] / (df['runtime'] + 0.1)
        df['budget_popularity_ratio'] = df['budget'] / (df['popularity'] + 0.1)
        df['budget_year_ratio'] = df['budget'] / (df['release_year'] * df['release_year'])

        df['releaseYear_popularity_ratio'] = df['release_year'] / (df['popularity'] + 0.1)
        df['releaseYear_popularity_ratio2'] = df['popularity'] / df['release_year']

        df['production_countries_count'] = df['production_countries'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['production_companies_count'] = df['production_companies'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['cast_count'] = df['cast'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['crew_count'] = df['crew'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        df['inflationBudget'] = df['budget'] + df['budget'] * 1.8 / 100 * (
                2019 - df['release_year'])  # Inflation simple formula

        # Drop Unused Features
        # Drop constant value, unique value, uncorrelated value
        df.drop(['homepage', 'status'], axis=1, inplace=True)

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

        df['belongs_to_collection'].fillna('!')
        df['genres'].fillna('!')
        df['production_companies'].fillna('!')
        df['production_countries'].fillna('!')
        df['spoken_languages'].fillna('!')
        df['cast'].fillna('!')
        df['crew'].fillna('!')
        df['tagline'] = df['tagline'].fillna('0')
        df['overview'] = df['overview'].fillna('0')

        df["belongs_to_collection"] = df["belongs_to_collection"].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else ["!"])
        df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else ["!"])
        df['production_companies'] = df["production_companies"].apply(
            lambda x: [x[i]["name"] for i in range(len(x))] if isinstance(x, list) else ["!"]).values
        df['production_countries'] = df['production_countries'].apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else ["!"])
        df['spoken_languages'] = df['spoken_languages'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else ["!"])
        df['Keywords'] = df['Keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else ["!"])
        df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else ["!"])
        df['crew'] = df['crew'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else ["!"])

    def feature_encoding(self, df, fit=False):
        # Encode the categorical feature into index
        self.encode_categorical_to_index(df, fit=fit)

    def encode_categorical_to_index(self, df, fit):

        if not fit:
            # Belongs to Collection
            le = preprocessing.LabelEncoder()
            le.fit(['!'] + list(set([i for j in df['belongs_to_collection'] for i in j])))
            df["belongs_to_collection"] = df["belongs_to_collection"].apply(lambda x: le.transform(x))
            self.categorical_dims.append(len(le.classes_))
            self.categorical_columns += ["belongs_to_collection"]
            self.cat_encoder += [le]

            # Genres
            le = preprocessing.LabelEncoder()
            le.fit(['!'] + list(set([i for j in df['genres'] for i in j])))
            df['genres'] = df['genres'].apply(lambda x: le.transform(x))
            self.categorical_dims.append(len(le.classes_))
            self.categorical_columns += ["genres"]
            self.cat_encoder += [le]

            # Production Company
            le = preprocessing.LabelEncoder()

            le.fit(['!'] + list(set([i for j in df['production_companies'] for i in j])))
            df['production_companies'] = df['production_companies'].apply(lambda x: le.transform(x))
            self.categorical_dims.append(len(le.classes_))
            self.categorical_columns += ["production_companies"]
            self.cat_encoder += [le]

            # Production Counties
            le = preprocessing.LabelEncoder()
            le.fit(['!'] + list(set([i for j in df['production_countries'] for i in j])))

            df['production_countries'] = df['production_countries'].apply(lambda x: le.transform(x))
            self.categorical_dims.append(len(le.classes_))
            self.categorical_columns += ["production_countries"]
            self.cat_encoder += [le]

            # Spoken Language
            le = preprocessing.LabelEncoder()
            le.fit(['!'] + list(set([i for j in df['spoken_languages'] for i in j])))
            df['spoken_languages'] = df['spoken_languages'].apply(lambda x: le.transform(x))
            self.categorical_dims.append(len(le.classes_))
            self.categorical_columns += ['spoken_languages']
            self.cat_encoder += [le]

            # Cast
            le = preprocessing.LabelEncoder()
            le.fit(['!'] + list(set([i for j in df['cast'] for i in j])))
            df['cast'] = df['cast'].apply(lambda x: le.transform(x))
            self.categorical_dims.append(len(le.classes_))
            self.categorical_columns += ['cast']
            self.cat_encoder += [le]

            # Crew
            le = preprocessing.LabelEncoder()
            le.fit(['!'] + list(set([i for j in df['crew'] for i in j])))
            df['crew'] = df['crew'].apply(lambda x: le.transform(x))
            self.categorical_dims.append(len(le.classes_))
            self.categorical_columns += ['crew']
            self.cat_encoder += [le]
        else:
            # Belongs to Collection
            i = self.categorical_columns.index("belongs_to_collection")
            le = self.cat_encoder[i]
            df["belongs_to_collection"] = df["belongs_to_collection"].apply(lambda x: [i for i in x if i in le.classes_])
            df["belongs_to_collection"] = df["belongs_to_collection"].apply(lambda x: le.transform(x))

            # Genres
            i = self.categorical_columns.index("genres")
            le = self.cat_encoder[i]
            df['genres'] = df['genres'].apply(lambda x: le.transform(x))

            # Production Company
            i = self.categorical_columns.index("production_companies")
            le = self.cat_encoder[i]
            df['production_companies'] = df['production_companies'].apply(
                lambda x: [i for i in x if i in le.classes_])
            df['production_companies'] = df['production_companies'].apply(lambda x: le.transform(x))

            # Production Counties
            i = self.categorical_columns.index("production_countries")
            le = self.cat_encoder[i]
            df['production_countries'] = df['production_countries'].apply(
                lambda x: [i for i in x if i in le.classes_])
            df['production_countries'] = df['production_countries'].apply(lambda x: le.transform(x))

            # Spoken Language
            i = self.categorical_columns.index("spoken_languages")
            le = self.cat_encoder[i]
            df['spoken_languages'] = df['spoken_languages'].apply(
                lambda x: [i for i in x if i in le.classes_])
            df['spoken_languages'] = df['spoken_languages'].apply(lambda x: le.transform(x))

            # Cast
            i = self.categorical_columns.index("cast")
            le = self.cat_encoder[i]
            df['cast'] = df['cast'].apply(
                lambda x: [i for i in x if i in le.classes_])
            df['cast'] = df['cast'].apply(lambda x: le.transform(x))

            # Crew
            i = self.categorical_columns.index("crew")
            le = self.cat_encoder[i]
            df['crew'] = df['crew'].apply(
                lambda x: [i for i in x if i in le.classes_])
            df['crew'] = df['crew'].apply(lambda x: le.transform(x))

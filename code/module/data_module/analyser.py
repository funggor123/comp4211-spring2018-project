import seaborn as sns
import matplotlib.pyplot as plt
from module.data_module import preprocessor
import numpy as np
from wordcloud import WordCloud
from collections import Counter

entire = True


def analysis(df):
    if entire:
        # Overall Statistics
        show_df_shape(df)
        show_columns_type(df)
        show_columns_descriptive_statistics(df)
        show_top_data(df, 10)
        show_columns_nan_counts(df)
        preprocessor.preprocess(df)

        # Columns Statistics
        show_title_dff_orgin_title(df)
        show_top_collection_that_films_belongs_to(df, 15)
        show_number_of_collections(df)
        show_tagline_nan_counts(df)
        show_taglines_word_cloud(df)
        show_keywords_word_cloud(df)
        show_the_most_famous_production_company(df, 10)
        show_production_counties(df)
        show_the_number_of_spoken_languages_in_a_film(df, 10)
        show_top_spoken_language_in_a_film(df, 10)
        show_most_common_genre(df)
        show_number_of_genre(df)
        show_revenue_graph(df)
        show_revenue_describe(df)
        show_budget(df)
        show_revenue_vs_budget(df)
        show_popularity(df)
        show_popularity_vs_revenue(df)
        show_home_page(df)
        show_crew(df)
        show_cast(df)
        show_runtime(df)
        correlation_matrix(df)


def show_title_dff_orgin_title(df):
    line()
    print("[Original Title Difference from title]")
    line()
    count = 5
    for i, ele in enumerate(df["original_title"]):
        if ele != df["title"][i]:
            print(ele, " -- ", df["title"][i])
            count -= 1
            if count == 0:
                break


def analysis_after_transform(df):
    if entire:
        show_days(df)
        show_days_vs_revenue(df)
        show_month_vs_revenue(df)
        show_year_vs_revenue(df)
        show_revenue_vs_budget_norm(df)
        show_popularity_vs_budget_norm(df)
        show_runtime_vs_revenue_norm(df)
        show_columns_nan_counts(df)


def show_runtime(df):
    line()
    print("[Runtime]")
    line()
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.distplot(np.log1p(df['runtime'].fillna(0)))

    plt.subplot(1, 2, 2)
    sns.scatterplot(np.log1p(df['runtime'].fillna(0)), np.log1p(df['revenue']))
    plt.show()
    print("Graph")


def show_runtime_vs_revenue_norm(df):
    line()
    print("[Runtime vs revenue norm]")
    line()
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(df['norm_log_runtime'].fillna(0), df['norm_log_revenue'])
    plt.show()
    print("Graph")


def show_columns_nan_counts(df):
    line()
    print("[show_columns_nan_counts]")
    line()
    print(df.isna().sum().sort_values(ascending=False))
    missing = df.isna().sum().sort_values(ascending=False)
    sns.barplot(missing[:15], missing[:15].index)
    plt.show()
    print("Graph")


def show_days(df):
    line()
    print("[Days]")
    line()
    day = df['release_day'].value_counts().sort_index()
    sns.barplot(day.index, day)
    plt.gca().set_xticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                              rotation='45')
    plt.ylabel('No of releases')
    plt.show()
    print("Graph")


def show_month_vs_revenue(df):
    line()
    print("[Month vs revenue]")
    line()
    plt.figure(figsize=(10, 15))
    sns.catplot(x='release_month', y='revenue', data=df)
    month_lst = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                 'August', 'September', 'October', 'November', 'December']
    plt.gca().set_xticklabels(month_lst, rotation='90')
    plt.show()
    print("Graph")


def show_year_vs_revenue(df):
    plt.figure(figsize=(15, 8))
    yearly = df.groupby(df['release_year'])['revenue'].agg('mean')
    plt.plot(yearly.index, yearly)
    plt.xlabel('year')
    plt.ylabel("Revenue")
    plt.show()
    print("Graph")


def show_days_vs_revenue(df):
    line()
    print("[Days vs revenue]")
    line()
    sns.catplot(x='release_day', y='revenue', data=df)
    plt.gca().set_xticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                              rotation='90')
    plt.show()
    print("Graph")


def line():
    print("-----------------------")


def show_columns_descriptive_statistics(df):
    line()
    print("[show_columns_descriptive_statistics]")
    line()
    print(df.describe(include='all'))


def show_most_common_genre(df):
    line()
    print("[The most common genre]")
    line()
    genre = df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    count = Counter([i for j in genre for i in j]).most_common(15)
    sns.barplot([val[1] for val in count], [val[0] for val in count])
    plt.show()
    print("Graph")


def show_number_of_genre(df):
    line()
    print("[The number of genre]")
    line()
    genres = df['genres'].apply(lambda x: len(x))
    plt.hist(genres, bins=10, color='g')
    plt.show()
    print("Graph")


def show_df_shape(df):
    line()
    print("[show_df_shape]")
    line()
    print(df.shape)


def show_columns_type(df):
    line()
    print("[show_columns_type]")
    line()
    print(df.info())


def show_top_data(df, number_of_top):
    line()
    print("[show_top_data]")
    line()
    print(df.head(number_of_top))


def show_revenue(df):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('skewed data')
    sns.distplot(df['revenue'])
    plt.subplot(1, 2, 2)
    plt.title('log transformation')
    sns.distplot(np.log(df['revenue']))
    plt.show()


def show_top_collection_that_films_belongs_to(df, top):
    line()
    print("[Top collection that films belongs to]")
    line()
    collections = df["belongs_to_collection"].apply(lambda x: x[0]["name"] if x != {} else '?').value_counts()[
                  1:top]
    plt.style.use('seaborn')
    sns.barplot(collections, collections.index)
    plt.show()
    print("Graph")


def show_number_of_collections(df):
    counts_collections = df["belongs_to_collection"].apply(lambda x: len(x))
    plt.hist(counts_collections, bins=10, color='g')
    plt.title('number of collection')
    plt.show()
    print("Graph")


def show_top_spoken_language_in_a_film(df, top):
    line()
    print("[Top Spoken languages in a film]")
    line()
    lang = df['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    count = Counter([i for j in lang for i in j]).most_common(top)
    sns.barplot([val[1] for val in count], [val[0] for val in count])
    plt.show()
    print("Graph")


def show_the_number_of_spoken_languages_in_a_film(df, top):
    line()
    print("[The number of languages spoken in a film]")
    line()
    print(df["spoken_languages"].apply(lambda x: len(x) if x != {} else 0).value_counts()[0:top])


def show_budget(df):
    line()
    print("[The budget]")
    line()
    plt.subplots(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['budget'] + 1, bins=10, color='g')
    plt.title('budget skewed data')
    plt.subplot(1, 2, 2)
    plt.hist(np.log(df['budget'] + 1), bins=10, color='g')
    plt.title('budget log transformation')
    plt.show()
    print("Graph")
    '''
    for i, ele in enumerate(df['budget']):
        if ele == 0:
            print(df['title'][i])
    '''


def show_revenue_vs_budget(df):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(df['budget'], df['revenue'])
    plt.subplot(1, 2, 2)
    sns.scatterplot(np.log1p(df['budget']), np.log1p(df['revenue']))
    plt.show()
    print("Graph")

    x1 = np.array(df["budget"])
    y1 = np.array(df["revenue"])
    fig = plt.figure(1, figsize=(9, 5))
    plt.plot([0, 400000000], [0, 400000000], c="green")
    plt.scatter(x1, y1, c=['blue'], marker='o')
    plt.grid()
    plt.xlabel("budget", fontsize=10)
    plt.ylabel("revenue", fontsize=10)
    plt.title("Link between revenue and budget", fontsize=10)
    plt.show()
    print("Graph")


def show_revenue_vs_budget_norm(df):
    line()
    print("[Reven vs bugdet norm]")
    line()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    sns.scatterplot(df['norm_log_budget'], df['norm_log_revenue'])
    plt.show()
    print("Graph")


def show_popularity_vs_budget_norm(df):
    line()
    print("[Reven vs bugdet norm]")
    line()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    sns.scatterplot(df['norm_log_popularity'], df['norm_log_revenue'])
    plt.show()
    print("Graph")


def show_production_counties(df):
    line()
    print("[Production countries]")
    line()
    countries = df['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values
    count = Counter([j for i in countries for j in i]).most_common(10)
    sns.barplot([val[1] for val in count], [val[0] for val in count])
    plt.show()
    print("Graph")


def show_the_most_famous_production_company(df, top):
    line()
    print("[The most famous production companies and the number films released by each]")
    line()
    x = df["production_companies"].apply(
        lambda x: [x[i]["name"] for i in range(len(x))] if x != {} else []).values
    print(Counter([i for j in x for i in j]).most_common(top))


def show_tagline_nan_counts(df):
    line()
    print("[Tagline Nan counts (0=Nan)]")
    line()
    print(df["tagline"].apply(lambda x: 1 if x is not np.nan else 0).value_counts())


def show_keywords_word_cloud(df):
    line()
    print("[show_keywords_word_cloud]")
    line()
    keywords = df['Keywords'].apply(lambda x: ' '.join(i['name'] for i in x) if x != {} else '')
    plt.figure(figsize=(10, 10))
    data = ' '.join(words for words in keywords)
    wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                          width=1200, height=1000).generate(data)
    plt.imshow(wordcloud)
    plt.title('Keywords')
    plt.axis("off")
    plt.show()
    print("Graph")


def show_taglines_word_cloud(df):
    line()
    print("[show_taglines_word_cloud]")
    line()
    plt.figure(figsize=(10, 10))
    taglines = ' '.join(df["tagline"].apply(lambda x: x if x is not np.nan else ''))

    wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                          width=1200, height=1000).generate(taglines)
    plt.imshow(wordcloud)
    plt.title('Taglines')
    plt.axis("off")
    plt.show()
    print("Graph")


def show_revenue_graph(df):
    line()
    print("[show revenue graph]")
    line()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('revenue skewed data')
    sns.distplot(df['revenue'])
    plt.subplot(1, 2, 2)
    plt.title('revenue log transformation')
    sns.distplot(np.log(df['revenue']))
    plt.show()
    print("Graph")

    plt.subplots(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['revenue'], bins=10, color='g')
    plt.title('revenue skewed data')
    plt.subplot(1, 2, 2)
    plt.hist(np.log(df['revenue']), bins=10, color='g')
    plt.title('revenue log transformation')
    plt.show()
    print("Graph")


def show_revenue_describe(df):
    line()
    print("[show revenue describe]")
    line()
    print(df['revenue'].describe())


def show_popularity(df):
    line()
    print("[Popularity]")
    line()
    plt.hist(df['popularity'], bins=30, color='violet')
    plt.title("popularity")
    plt.show()
    print("Graph")


def correlation_matrix(df):
    line()
    print("[Correlation Matrix]")
    line()
    col = ['revenue', 'budget', 'popularity', 'runtime']
    plt.subplots(figsize=(10, 8))
    corr = df[col].corr()
    sns.heatmap(corr, xticklabels=col, yticklabels=col, linewidths=.5, cmap="Reds")
    plt.show()
    print("Graph")


def show_popularity_vs_revenue(df):
    line()
    print("[Popularity vs revenue]")
    line()
    sns.scatterplot(df['popularity'], df['revenue'], color='violet')
    plt.show()
    print("Graph")


def show_home_page(df):
    line()
    print("[Home_page]")
    line()
    print(df['homepage'].value_counts().sort_values(ascending=False)[:5])


def show_crew(df):
    line()
    print("[Crews]")
    line()
    crew = df['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    print(Counter([i for j in crew for i in j]).most_common(15))


def show_cast(df):
    line()
    print("[Cast]")
    line()
    cast = df['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    print(Counter([i for j in cast for i in j]).most_common(15))

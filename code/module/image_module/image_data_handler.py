import urllib.request
import os
from collections import Counter

poster_dir_path = "./data/poster/"
unk_poster_dir_path = "./data/unk_poster/"
poster_url = "http://image.tmdb.org/t/p/w185"
mode_path = "./model/"


def download_all_posters(df):
    genre = df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
    directory = os.path.dirname(poster_dir_path)
    num_class = 0
    for ele in list(dict(Counter([i for j in genre for i in j])).keys()):
        num_class += 1
        if not os.path.exists(directory + "/" + ele):
            os.makedirs(directory + "/" + ele)
    directory = os.path.dirname(unk_poster_dir_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, url in enumerate(df["poster_path"]):
        if len(df["genres"][i]) == 0:
            download_poster_unk(url, i)
            print("Found UNK genres: ", i)
        print(i, "/" + str(len(df["poster_path"])))
        for genre in df["genres"][i]:
            if not os.path.exists(poster_dir_path +
                                  genre["name"] + "/" + str(i) + ".jpg"):
                download_poster(url, genre["name"], i)
    return num_class


def download_poster_unk(poster_path, ids):
    try:
        urllib.request.urlretrieve(poster_url + str(poster_path), unk_poster_dir_path +
                                   str(ids) + ".jpg")
    except IOError as e:
        print('404', e)
    except Exception as e:
        print('404', e)


def download_poster(poster_path, class_name, ids):
    try:
        urllib.request.urlretrieve(poster_url + str(poster_path), poster_dir_path +
                                   str(class_name) + "/" + str(ids) + ".jpg")
    except IOError as e:
        print('404', e)
    except Exception as e:
        print('404', e)

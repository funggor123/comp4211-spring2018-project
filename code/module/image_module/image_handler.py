import urllib.request
import os
from collections import Counter


class ImageHandler:

    def __init__(self, dir_prefix):

        self.poster_dir_path = "./" + dir_prefix + "data/poster/"
        self.unk_poster_dir_path = "./data/" + dir_prefix + "unk_poster/"
        self.poster_url = "http://image.tmdb.org/t/p/w185"

    def download_all_posters(self, df):
        genre = df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
        directory = os.path.dirname(self.poster_dir_path)
        num_class = 0
        for ele in list(dict(Counter([i for j in genre for i in j])).keys()):
            num_class += 1
            if not os.path.exists(directory + "/" + ele):
                os.makedirs(directory + "/" + ele)
        directory = os.path.dirname(self.unk_poster_dir_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        for i, url in enumerate(df["poster_path"]):
            if len(df["genres"][i]) == 0:
                self.download_poster_unk(url, i)
                print("Found UNK genres: ", i)
            print(i, "/" + str(len(df["poster_path"])))
            for genre in df["genres"][i]:
                if not os.path.exists(self.poster_dir_path +
                                      genre["name"] + "/" + str(i) + ".jpg"):
                    self.download_poster(url, genre["name"], i)
        return num_class

    def download_poster_unk(self, poster_path, ids):
        try:
            urllib.request.urlretrieve(self.poster_url + str(poster_path), self.unk_poster_dir_path +
                                       str(ids) + ".jpg")
        except IOError as e:
            print('404', e)
        except Exception as e:
            print('404', e)

    def download_poster(self, poster_path, class_name, ids):
        try:
            urllib.request.urlretrieve(self.poster_url + str(poster_path), self.poster_dir_path +
                                       str(class_name) + "/" + str(ids) + ".jpg")
        except IOError as e:
            print('404', e)
        except Exception as e:
            print('404', e)

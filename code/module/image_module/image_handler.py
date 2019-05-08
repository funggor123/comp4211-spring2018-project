import urllib.request
import os

class ImageHandler:

    def __init__(self, dir_prefix):

        self.poster_dir_path = "./poster/" + dir_prefix + "/"
        self.poster_url = "http://image.tmdb.org/t/p/w185"

    def download_all_posters(self, df):
        directory = os.path.dirname(self.poster_dir_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        print("Download Image")

        for i, url in enumerate(df["poster_path"]):
            print(i, "/" + str(len(df["poster_path"])))
            if not os.path.exists(self.poster_dir_path + str(i) + ".jpg"):
                self.download_poster(url, i)

        print("Download Finish")

    def download_poster(self, poster_path, ids):
        try:
            urllib.request.urlretrieve(self.poster_url + str(poster_path), self.poster_dir_path + str(ids) + ".jpg")
        except IOError as e:
            print('404', e)
        except Exception as e:
            print('404', e)

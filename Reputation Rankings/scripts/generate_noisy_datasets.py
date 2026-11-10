"""
Generate noisy datasets by adding spamming users.

For each dataset and each spamming ratio, a CSV is saved to OUT_DIR:
    <out_dir>/<dataset>_spam_<ratio in %>.csv     

"""
import os

import numpy as np
import pandas as pd
import utils

SEED = 42
RATIOS = [0.1, 0.3, 0.5]
OUT_DIR = "../data/noisy"


def load_datasets():
    books = pd.read_csv('../data/book-crossing/Ratings.csv', sep=';')
    books = books[["User-ID", "ISBN", "Rating"]]

    movielens = pd.read_csv("../data/ml-100k/u.data", sep='\t', names=["userId", "movieId", "rating", "timestamp"])
    movielens = movielens[["userId", "movieId", "rating"]]

    datasets = {"book-crossing": books, "movielens": movielens}
    return {name: df for name, df in datasets.items()}



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, data in load_datasets().items():
        for ratio in RATIOS:
            print(f"Generating {name}_spam_{int(ratio * 100)} dataset ...")
            noisy = utils.add_spammers(data, ratio)
            path = os.path.join(OUT_DIR, f"{name}_spam_{int(ratio * 100)}.csv")
            noisy.to_csv(path, index=False)
            print(f"Finish!")


if __name__ == "__main__":
    main()
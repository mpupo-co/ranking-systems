import numpy as np
import pandas as pd

def split_ds_sellers(ratings, n_sellers):
    """
    Randomly distribute item-ratings across sellers (Algorithm 2)

    Parameters
    ----------
    ratings : pandas DataFrame - Original ratings, containing 
        columns: user, item, rating.
    n_sellers : int - Number of sellers

    Returns
    -------
    split_ratings : pandas DataFrame - columns: user, seller, item, rating
    """

    SEED = 42
    rng = np.random.default_rng(SEED) # create a random number generator

    items = ratings["item"].unique() 
    sellerID = np.arange(1, n_sellers+1)

    # Randomly assign a random set of seller to each item.
    item_sellers = {}
    for item in items:
        item_sellers[item]=rng.choice(sellerID, 
                                      size=rng.integers(1,n_sellers+1), # random numberof sellers per item 
                                      replace=False) # avoids a seller to be picked twice for the same item

    # Assign each rating to one of its item's sellers
    sellers=[]
    for item in ratings["item"]:
        seller = rng.choice(item_sellers[item]) # randomly pick a seller for the curr (item, rating)
        sellers.append(seller)

    # Add seller column
    split_ratings = ratings.copy()
    split_ratings["seller"] = sellers

    # Arrange columns
    split_ratings = split_ratings[["user","seller","item","rating"]]

    return split_ratings

def load_ml1m():
    """
    Load movielens 1M dataset into a dataframe and normalizes ratings
    
    Returns
    -------
    ratings : pandas DataFrame - columns: user, item, rating
    """
    ratings = pd.read_csv("data/ml-1m/ratings.dat", sep="::", names=["user","item", "rating","timestamp"], engine="python")
    
    # Normalize dividing by the max rating
    max_rate = ratings["rating"].max()
    ratings["rating"] = ratings["rating"]/max_rate

    # drop timestamp
    return ratings[["user", "item","rating"]]
    




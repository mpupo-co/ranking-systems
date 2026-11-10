import pandas as pd
import numpy as np

def push_nuke(ratings):
    """
    Simulates a push-nuke attack where 1 malicious seller pushes
    the ranking of an item to max value, and nukes a set of 10
    random items (with uniform distribution).

    Parameters
    ----------
    ratings : pandas DataFrame - Original ratings, containing 
        columns: user, seller, item, rating. Ratings within ]0,1]

    Returns
    -------
    noisy : pandas Dataframe - data after attack
    """

     # Calculate relevance scores for each seller-item as the mean of ratings
    rs = ratings.groupby(["seller", "item"])["rating"].mean().reset_index()

    SEED = 42
    rng = np.random.default_rng(SEED) # create a random number generator
    
    sellers = ratings["seller"].unique()

    # Randomly select a "malicious" seller
    hacker = rng.choice(sellers)

    # Get the item belonging to malicious seller (hacker)
    hacker_items = rs.loc[rs["seller"] == hacker, "item"].to_numpy()

    # Randomly select the item to push
    target_item = rng.choice(hacker_items)

    # Exclude the target item and rand select 10 to nuke
    remaining_items = hacker_items[hacker_items != target_item]
    nuked_items = rng.choice(remaining_items, size=10, replace=False)

    #Attack
    noisy = ratings.copy()
    noisy.loc[(noisy["seller"] == hacker) & (noisy["item"] == target_item),"rating"] = 1 # max norm rating 
    noisy.loc[(noisy["seller"] == hacker) & (noisy["item"].isin(nuked_items)),"rating"]  = 1/5 # min norm rating 

    return noisy

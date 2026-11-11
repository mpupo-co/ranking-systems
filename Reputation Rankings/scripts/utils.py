import pandas as pd
import numpy as np
from scipy.stats import kendalltau

def load_ml100k():
    """
    Load movielens 100k dataset into a dataframe and normalizes ratings
    
    Returns
    -------
    ratings : pandas DataFrame - columns: user, item, rating
    """
    ratings = pd.read_csv("data/ml-100k/u.data", sep='\t', names=["user", "item", "rating", "timestamp"])
    
    # Normalize dividing by the max rating
    max_rate = ratings["rating"].max()
    ratings["rating"] = ratings["rating"]/max_rate

    # drop timestamp
    return ratings[["user", "item","rating"]]

def load_amazonMusic():
    """
    Load Amazon Music dataset into a dataframe and normalizes ratings
    
    Returns
    -------
    ratings : pandas DataFrame - columns: user, item, rating
    """

    url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Digital_Music.csv.gz"
    ratings = pd.read_csv(url, compression="gzip", header=0, names=["user", "item", "rating", "timestamp"])
    
    # Normalize dividing by the max rating
    max_rate = ratings["rating"].max()
    ratings["rating"] = ratings["rating"]/max_rate

    # drop timestamp
    return ratings[["user", "item","rating"]]
                          
def kendall(r1, r2):
    """
    Kendall tau between two ranking Series, aligned on common items.
    """

    common_idx = r1.index.intersection(r2.index)
    return kendalltau(r1.loc[common_idx], r2.loc[common_idx])[0]

def average_ranking(data):
    """
    Computes the aggregated average ranking: mean rating per item.

    Parameters
    ----------
    ratings : pandas DataFrame - Original ratings, containing 
            columns: user, item, rating.
    
    Returns
    ----------
    ranking: pandas Series
    """
    
    ranking = data.groupby("item")["rating"].mean().sort_values(ascending=False)
    return ranking

def reputation_ranking(ratings, distance="l1", aggregate="avg", lam=0.1, max_iter=30, tolerance=1e-8):
    """
    Generic reputation-based ranking on a bipartite rating network (Li et al., Eq. 3.1-3.12).
    L1-AVG, L2-AVG, L1-MAX, L2-MAX, L1-MIN and L2-MIN.

    Parameters
    ----------
    ratings: pandas DataFrame, one row per rating (normalized). Columns: user, item, rating.
    distance: "l1" -> |R_ij - r_j|; "l2" -> (R_ij - r_j)^2
    aggregate: "avg", "max" or "min". How each user's distances are aggregated.
    lam : float in (0, 1). Decay constant lambda.
    max_iter: int. Maximum number of iterations.
    tolerance: float. Max difference to converge

    Returns
    -------
    r : Series (n_items,). ranking scores of objects, indexed by item
    """

    coef = lam if distance == "l1" else lam / 2
    agg_func = {"avg": "mean", "max": "max", "min": "min"}[aggregate]

    # Initialize user reputations and item rankings
    c = pd.Series(1.0, index=ratings["user"].unique())
    r = pd.Series(ratings["rating"].mean(), index=ratings["item"].unique())

    # Number of ratings per item
    item_counts = ratings.groupby("item")["rating"].count()

    for i in range(max_iter):
        c_old = c.copy()

        # Update item rankings
        weights = ratings["user"].map(c)

        weighted_sum = ((ratings["rating"] * weights).groupby(ratings["item"]).sum())

        if aggregate == "avg":
            weight_sum = weights.groupby(ratings["item"]).sum()
            r = weighted_sum / weight_sum

        else:
            # MAX / MIN: weighted sum divided by number of ratings
            r = weighted_sum / item_counts

        # Compute distances between ratings and updated item rankings
        diff = ratings["rating"] - ratings["item"].map(r)

        if distance == "l1":
            distances = diff.abs()
        else:
            distances = diff.pow(2)

        # Update user reputations: c = 1 - coef * AGG(distances)
        user_distance = distances.groupby(ratings["user"]).agg(agg_func)
        c = 1 - coef * user_distance

        # Check convergence
        error = (c - c_old).abs().max()

        if error < tolerance:
            break

    return r.sort_values(ascending=False)

def add_spammers(ratings, spam_ratio, rating_min=0.2, rating_max=1.0, seed=42):
    """
    Add spamming users to a ratings dataset (Li et al., robustness experiment).

    Three equally sized groups of spammers:
        random : uniform random ratings in [rating_min, rating_max]
        max    : always give the maximal rating
        min    : always give the minimal rating
    Each spammer rates a random set of items, whose size is the average
    number of items rated per user in the original data.

    Parameters
    ----------
    ratings : pandas DataFrame - Normalized ratings, columns: user, item, rating.
    spam_ratio : float - Number of spammers as a fraction of the original users
    rating_min, rating_max : float - Minimal / maximal rating of the normalized scale.

    Returns
    -------
    pandas DataFrame - Original ratings plus the spammers' ratings.
    """
    rng = np.random.default_rng(seed)

    items = ratings["item"].unique()
    n_spammers = int(round(spam_ratio * ratings["user"].nunique()))

    n_per_type = n_spammers // 3          # same proportion for each type

    # number of items rated by spammed users 
    n_items = int(round(ratings.groupby("user")["item"].count().mean()))

    rows = []
    for spam_type in ["random", "max", "min"]:
        for k in range(n_per_type):
            user = f"spam_{spam_type}_{k}"
            chosen = rng.choice(items, size=n_items, replace=False) # set of items to spam

            if spam_type == "random":
                values = rng.uniform(rating_min, rating_max, size=n_items)
            elif spam_type == "max":
                values = np.full(n_items, rating_max)
            else:
                values = np.full(n_items, rating_min)

            rows.append(pd.DataFrame({"user": user, "item": chosen, "rating": values}))

    if not rows:
        return ratings.copy()

    return pd.concat([ratings] + rows, ignore_index=True)

def bribe(ratings):
    bribe_item = ratings.groupby("item")
    return 0
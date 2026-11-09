import pandas as pd
import numpy as np


# Federated Strategies for marketplace rankings
def n_aa(ratings):
    """
    Compute marketplace item ranking, averaging seller scores.

    Parameters
    ----------
    ratings : pandas DataFrame - Original ratings, containing 
        columns: user, seller, item, rating.

    Returns
    -------
    ranking : pandas Series - Items ordered by their final relevance scores.
    """

    # Calculate relevance scores for each seller-item as the mean of ratings
    rs = ratings.groupby(["seller", "item"])["rating"].mean()

    ranking = rs.groupby("item").mean()
    return ranking.sort_values(ascending=False)


def w_aa(ratings):
    """
    Compute marketplace item ranking, averaging seller scores, weighted 
    on number of ratings given by users.

    Parameters
    ----------
    ratings : pandas DataFrame - Original ratings, containing 
        columns: user, seller, item, rating.

    Returns
    -------
    ranking : pandas Series - Items ordered by their final relevance scores.
    """

    # Calculate relevance scores for each seller-item
    rs = ratings.groupby(["seller", "item"])["rating"].mean()

    # Count distinct users who rated each seller-item pair
    rating_counts = ratings.groupby(["seller", "item"])["user"].nunique()

    # Calculate weighted relevance for each seller-item pair
    weighted_relevance = rs * rating_counts

    # Calculate weighted average for each item
    ranking = weighted_relevance.groupby("item").sum() / \
        rating_counts.groupby("item").sum()

    return ranking.sort_values(ascending=False)

def compute_reputations(rs, lambda_M=.2):
    """
    Parameters
    ----------
    rs: pandas DataFrame - seller relevance scores, containing
        columns: seller, item, relevance.
    lambda_M: float - reputation penalty.

    Returns
    ----------
    reputations : pandas DataFrame - Contains columns: seller,
        item, reputation.
    """

    reputations = []

    # For each seller
    sellers = rs["seller"].unique()

    for seller in sellers:
        seller_items = rs[rs["seller"] == seller]["item"]

        # For each item in the seller's catalogue
        for item in seller_items:
            relevance_row = rs.loc[
                (rs["seller"] == seller) & (rs["item"] == item)
            ]
            relevance_score = relevance_row["relevance"].iloc[0]

            sum_diff = 0
            count = 0

            # Compare with other sellers
            for other_seller in sellers:
                if other_seller != seller:
                    other_relevance_score = rs.loc[
                        (rs["seller"] == other_seller) & (rs["item"] == item),
                        "relevance"
                    ]

                    # Check if the other seller offers this item
                    if not other_relevance_score.empty:
                        sum_diff += abs(
                            relevance_score - other_relevance_score.iloc[0]
                        )
                        count += 1

            # Calculate reputation
            if sum_diff > 0:
                reputation = 1 - lambda_M * (sum_diff / count)
            else:
                reputation = lambda_M

            reputations.append({
                "seller": seller,
                "item": item,
                "reputation": reputation
            })

    return pd.DataFrame(reputations)
        

def federated_reputation_ranking(ratings, lambda_M=.2):
    """
    Compute the final marketplace item ranking, aggregating seller scores
    based on rating items agreement.

    Parameters
    ----------
    ratings : pandas DataFrame - Original ratings, containing 
        columns: user, seller, item, rating.
    lambda_M : float - Reputation penalty parameter.

    Returns
    ----------
    ranking : pandas DataFrame - Items ordered by their final relevance scores.
    """

    # Calculate relevance scores for each seller-item as the mean of ratings
    rs = ratings.groupby(["seller", "item"])["rating"].mean().reset_index()
    rs = rs.rename(columns={"rating": "relevance"})

    # Calculate reputations for each seller-item pair
    reputations = compute_reputations(rs, lambda_M)

    # Count distinct users who rated each seller-item pair
    rating_counts = ratings.groupby(["seller", "item"])["user"].nunique().reset_index(name="rating_count")

    # Combine relevance scores, reputations and rating counts
    rs = rs.merge(reputations, on=["seller", "item"])
    rs = rs.merge(rating_counts, on=["seller", "item"])

    # Calculate the weight of each seller-item pair: |Ui_Sk| * ri_Sk
    rs["weight"] = rs["reputation"] * rs["rating_count"]

    # Calculate weighted relevance scores: ci_k * |Ui_Sk| * ri_Sk
    rs["weighted_relevance"] = rs["relevance"] * rs["weight"]

    # Sum weighted relevance scores for each item
    weighted_sums = rs.groupby("item")["weighted_relevance"].sum()

    # Sum weights for each item
    total_weights = rs.groupby("item")["weight"].sum()

    # Calculate final relevance scores
    ranking = weighted_sums / total_weights

    return ranking.sort_values(ascending=False)

# Reputation-based ranking (non-federated approach)
def reputation_ranking(ratings, max_iter=30, tol=1e-8, lamb=0.1):
    """
    Reputation-based item ranking using the equations from ref. paper saude et al. 2021.

    Parameters
    ----------
    ratings : pandas DataFrame - Normalized ratings [0,1], containing 
        columns: user, item, rating.
    
    Returns
    ----------
    r: Series with items' ranking

    """

    # Initialize reputations to 1, c
    c = pd.Series(1.0, index=ratings["user"].unique())
    
    # Initilize rankings to average, r
    r=pd.Series(ratings["rating"].mean(), index=ratings["item"].unique())
  
    
    for i in range(max_iter):
        c_old = c.copy()

        # Compute updated ranking
        # R_ui * cu
        ratings["w_rating"] = ratings["rating"] * ratings["user"].map(c)
        ratings["w_reputation"]  =  ratings["user"].map(c)
        # sum(R_ui * cu)/sum(cu)
        r = ratings.groupby("item")["w_rating"].sum()/ratings.groupby("item")["w_reputation"].sum()
        
        # Compute updated user reputation
        # |R_ui - r_i|
        ratings["distance"] = (ratings["rating"] - ratings["item"].map(r)).abs()
        # 1 - lambda/|Iu| * sum(|R_ui - r_i|)
        c = 1 - lamb * ratings.groupby("user")["distance"].mean()
     
        # Check convergence
        error = (c - c_old).abs().max()
        if error < tol:
            break
    
    #print (f"number of iterations: {i+1}")
    return r.sort_values(ascending=False)


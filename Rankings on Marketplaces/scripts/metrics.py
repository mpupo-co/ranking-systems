import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

def kendall_tau(r1, r2):
    """Kendall tau between two ranking Series, aligned on common items."""
    common_idx = r1.index.intersection(r2.index)
    return kendalltau(r1.loc[common_idx], r2.loc[common_idx])[0]


def plot_kendall(kendall_res):
    """
    Plot Kendall tau as function of number of seller for each federated
    algortihm

    Parameters
    ----------
    kendal_res : pandas DataFrame -  columns: Num Sellers, 
        Algorithm, kendall tau.
    """

    # Add the 1-seller baseline for each algorithm
    baseline = pd.DataFrame([
        {
            "Num Sellers": 1,
            "Algorithm": algorithm,
            "kendall tau": 1.0
        }
        for algorithm in kendall_res["Algorithm"].unique()
    ])

    plot_data = pd.concat([kendall_res, baseline], ignore_index=True)

    # Sort results
    plot_data = plot_data.sort_values(by=["Algorithm", "Num Sellers"])

    # Plot
    plt.figure(figsize=(6,5))

    sns.lineplot(
        data=plot_data,
        x="Num Sellers",
        y="kendall tau",
        hue="Algorithm",
        marker="o"
    )

    plt.xlabel("Number of Sellers")
    plt.ylabel("Kendall's Tau")
    plt.title("Ranking Effectiveness - Kendall's τ")
    plt.xticks(sorted(plot_data["Num Sellers"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.show()


def mean_abs_error(ratings, mktpl_ranking):
    """
    Compute average mean absolute error between the overall relevance 
    score obtained by the marketplace and the relevance score provided 
    by each seller.

    Parameters
    ----------
    ratings : pandas DataFrame - Original ratings, containing 
        columns: user, seller, item, rating.
    mktpl_ranking: pandas Series - relevance scores given by the marketplace

    Returns
    -------
    mae : float - mean absolute error
    """

    # Calculate relevance scores for each seller-item (= average item rating)
    rs = ratings.groupby(["seller", "item"])["rating"].mean()

    # Match marketplace relevance scores to each seller-item pair
    marketplace_rs = rs.index.get_level_values("item").map(mktpl_ranking)

    # Calculate absolute errors. |ri - ri_Sk|
    error = (rs - marketplace_rs).abs()

    # Calculate mean absolute errors per seller. |ri - ri_Sk|/|Ĩ|
    mae_sk = error.groupby('seller').mean()
  
    # Calculate mean absolute errors across sellers. sum(mae_sk)/s, s=num sellers
    return mae_sk.mean()
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

def kendall_tau(r1, r2):
    """Kendall tau between two ranking Series, aligned on common items."""
    common_idx = r1.index.intersection(r2.index)
    return kendalltau(r1.loc[common_idx], r2.loc[common_idx])[0]

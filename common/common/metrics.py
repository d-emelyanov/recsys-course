import pandas as pd
import numpy as np


def precision_at_k_(k: int, recs: list, real: list):
    TP = len(list(set(recs[:k]).intersection(set(real))))
    return TP / k


def precision_at_k(k: int, recs: pd.Series, real: pd.Series):
    res = []
    for i in range(recs.shape[0]):
        recs_ = recs.iloc[i]
        real_ = real.iloc[i]
        res.append(precision_at_k_(
            k, recs_, real_
        ))
    return pd.Series(res, dtype='float64')


def average_precision_at_k(k: int, recs: pd.Series, real: pd.Series):
    res = []
    for i in range(recs.shape[0]):
        recs_ = recs.iloc[i]
        real_ = real.iloc[i]
        n_rel = len(list(set(recs_[:k]).intersection(set(real_))))
        if n_rel > 0:
            res_ = 0
            for j, r in enumerate(recs_[:k]):
                TP = len(list(set(recs_[:j+1]).intersection(set(real_))))
                res_ += TP / (j+1) * int(r in real_)
            res.append(res_ / n_rel)
        else:
            res.append(0.0)
    return pd.Series(res, dtype='float64')


def map_at_k(k: int, recs: pd.Series, real: pd.Series):
    return average_precision_at_k(k, recs, real).mean()


def aggregate_diversity():
    pass


def intra_list_diversity():
    pass


def inter_user_diversity():
    pass


def mean_inverse_user_frequency():
    pass


def unexpectedness():
    pass


def serendipity():
    pass
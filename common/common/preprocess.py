import logging
import pandas as pd
import numpy as np
from transliterate import translit


def onehot(df, cols, prefix=False):
    for c in cols:
        if prefix:
            prefix_ = {
                'prefix': c
            }
        else:
            prefix_ = {}
        df = pd.concat([
            df,
            pd.get_dummies(df[c], **prefix_)
        ], axis=1)
    return df


def category(ser):
    if ser.dtype == 'category':
        return ser
    else:
        return ser.astype('category')


def preprocess_items(data):
    logging.info('Preprocessing items')
    data['content_type'] = category(
        data['content_type']
    )
    data['title'] = data['title'].str.lower()
    data['title_orig'] = data['title_orig'].fillna('None')

    data.loc[data['release_year'].isna(), 'release_year'] = 2020.
    data.loc[data['release_year'] < 1920, 'release_year_cat'] = 'inf_1920'
    data.loc[data['release_year'] >= 2020, 'release_year_cat'] = '2020_inf'
    for i in range (1920, 2020, 10):
        data.loc[
            (data['release_year'] >= i) & (data['release_year'] < i+10),
            'release_year_cat'
        ] = f'{i}-{i+10}'
    data['release_year_cat'] = category(data['release_year_cat'])

    data['genres'] = category(data['genres'])
    data.loc[data.countries.isna(), 'countries'] = 'Россия'
    data['countries'] = data['countries'].str.lower()
    data['countries'] = data['countries'].apply(lambda x: ', '.join(sorted(list(set(x.split(', '))))))
    data['countries'] = category(data['countries'])

    # TODO: there is a smarter way to do it
    data['for_kids'] = data['for_kids'].fillna(-1)
    data['for_kids'] = data['for_kids'].astype('int')

    data['for_kids_rating'] = data['age_rating'].map(
        lambda x: -1 if pd.isna(x) else 1 if int(x) <= 12 else 0
    )
    data.loc[data.age_rating.isna(), 'age_rating'] = 0
    data['age_rating'] = category(data['age_rating'])

    data['studios'] = data['studios'].fillna('Unknown')
    data['studios'] = data['studios'].str.lower()
    data['studios'] = data['studios'].apply(lambda x: ', '.join(sorted(list(set(x.split(', '))))))
    data['studios'] = category(data['studios'])

    data['directors'] = data['directors'].fillna('Unknown')
    data['directors'] = data['directors'].str.lower()
    data['directors'] = category(data['directors'])

    data['actors'] = data['actors'].fillna('Unknown')
    data['actors'] = category(data['actors'])

    data['keywords'] = data['keywords'].fillna('Unknown')
    data['keywords'] = category(data['keywords'])

    data['description'] = data['description'].fillna('-')

    return data


def preprocess_unused_items(data, items):

    class UnusedItems:
        def __init__(self, ids: set, r: int):
            self.ids = ids
            self.r = r

        def __call__(self, x):
            l = list(set(list(x)) ^ self.ids)
            return np.random.choice(l, min([len(l), self.r]))

    items = set(data['item_id'].unique().tolist()).intersection(
        set(items['item_id'].unique().tolist())
    )
    unused = (
        data
        .groupby('user_id')['item_id']
        .apply(UnusedItems(
            ids=items,
            r=10
        ))
        .reset_index()
        .explode('item_id')
        .dropna()
    )
    return unused

def preprocess_users(data):
    logging.info('Preprocessing users')

    data['age'] = data['age'].fillna('age_unknown').astype(str)
    data['income'] = data['income'].fillna('income_unknown').astype(str)
    data['sex'] = data['sex'].fillna('unknown').astype(str)
    data['kids_flg'] = data['kids_flg'].fillna(-1).astype('int8')

    data['sex'] = data['sex'].map(lambda x: translit(x.lower(), 'ru', reversed=True))

    return data


def preprocess_interactions(data):

    data['watched_pct'] = data['watched_pct'].astype(pd.Int8Dtype())
    data['watched_pct'] = data['watched_pct'].fillna(0)
    data['last_watch_dt'] = pd.to_datetime(data['last_watch_dt'])

    return data
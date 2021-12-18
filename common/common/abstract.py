import pandas as pd
from abc import (
    ABC,
    abstractmethod,
    abstractclassmethod,
    abstractproperty
)


class BaseRecommender(ABC):

    @abstractproperty
    @property
    def params(self):
        pass

    @abstractclassmethod
    def from_args(cls, params):
        pass

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def recommend(self, n, data):
        pass

    def add_user_features(self, data):
        self.user_features = data

    def add_item_features(self, data):
        self.item_features = data

    def get_full_df(self, data, user_col, item_col):
        data = pd.merge(
            left=data,
            right=self.user_features,
            on=[user_col],
            how='left'
        )
        data = pd.merge(
            left=data,
            right=self.item_features,
            on=[item_col],
            how='left'
        )
        return data

from abc import ABC, abstractmethod, abstractclassmethod


class BaseRecommender(ABC):

    @abstractclassmethod
    def from_args(cls, params):
        pass

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def recommend(self, n, data):
        pass

from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty


class BaseRecommender(ABC):

    @abstractproperty
    @property
    def params(self):
        pass

    @abstractclassmethod
    def from_args(cls, params):
        pass

    @abstractmethod
    def add_user_features(self, data):
        pass

    @abstractmethod
    def add_item_features(self, data):
        pass

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def recommend(self, n, data):
        pass

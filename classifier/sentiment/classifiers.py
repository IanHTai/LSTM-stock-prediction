from abc import ABC, abstractmethod
class Classifier(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, epochs, ndarray, earlyStopping=False):
        pass

    @abstractmethod
    def test(self, ndarray, expected):
        pass

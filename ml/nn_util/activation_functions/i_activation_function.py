from abc import ABC, abstractmethod


class IActivationFunction(ABC):

    @abstractmethod
    def eval(self, x):
        pass

    @abstractmethod
    def eval_deriv(self, x):
        pass

    @abstractmethod
    def __str__(self):
        pass
from abc import ABC


class IActivationFunction(ABC):

    def eval(self, x):
        pass

    def eval_deriv(self, x):
        pass
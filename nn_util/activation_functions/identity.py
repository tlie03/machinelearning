from nn_util.activation_functions.i_activation_function import IActivationFunction


class Identity(IActivationFunction):

    def __init__(self):
        super().__init__()
        self.name = 'identity'

    def eval(self, x):
        return x

    def eval_deriv(self, x):
        return 1

    def __str__(self):
        return self.name
from nn_util import NeuralNetwork, ILabelLoss


class Discriminator:

    def __init__(self, model: NeuralNetwork, loss: ILabelLoss):
        self.model = model
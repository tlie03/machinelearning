from nn_util import ILayer


class Discriminator:

    def __init__(self, model: ILayer, loss: ILoss):
        self.model = model
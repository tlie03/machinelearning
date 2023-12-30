from nn_util import Layer, ReLu, Identity, NeuralNetwork


nn = NeuralNetwork([
    Layer(2, 2, Identity()),
    Layer(2, 2, Identity())
])

if __name__ == '__main__':
    print(nn)
    print(nn.forward([2, 2]))

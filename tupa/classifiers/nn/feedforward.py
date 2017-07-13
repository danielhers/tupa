from tupa.config import MLP_NN
from .neural_network import NeuralNetwork


class MLP(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(MLP, self).__init__(MLP_NN, *args, **kwargs)

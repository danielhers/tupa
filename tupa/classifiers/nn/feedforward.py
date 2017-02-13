from nn.neural_network import NeuralNetwork
from tupa.config import MLP_NN


class MLP(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(MLP, self).__init__(MLP_NN, *args, **kwargs)

from nn.neural_network import NeuralNetwork
from parsing.config import MLP_NN


class MLP(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(MLP, self).__init__(MLP_NN, *args, **kwargs)

    def evaluate(self, features, train=False):
        self.init_cg()
        return self.evaluate_mlp(features, train)

import numpy as np


def randomize_orthonormal(*parameters, activation=None):  # Saxe et al., 2014 (https://arxiv.org/abs/1312.6120)
    for param in parameters:
        shape = param.shape()
        if len(shape) == 2 and shape[0] == shape[1] > 1:
            init, _, _ = np.linalg.svd(np.random.randn(*shape))
            if str(activation) == "relu":
                init *= np.sqrt(2)
            param.set_value(init)

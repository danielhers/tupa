import re

import torch.nn as nn


def randomize_orthonormal(**parameters):  # Saxe et al., 2014 (https://arxiv.org/abs/1312.6120)
    for name, param in parameters.items():
        if "bias" in name or re.match("b\d+", name):
            nn.init.constant(param, 0.0)
        elif "weight" in name or re.match("W\d+", name):
            nn.init.orthogonal(param)

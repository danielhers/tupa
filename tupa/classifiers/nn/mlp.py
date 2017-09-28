import dynet as dy


def init(model, layers, input_dim, layer_dim, output_dim, init, num_labels=None, **kwargs):
    i_dim = [input_dim] + (layers-1) * [layer_dim] + ([output_dim] if num_labels else [])
    o_dim = (layers-1) * [layer_dim] + [output_dim] + ([num_labels] if num_labels else [])
    return {key(prefix, i, **kwargs): model.add_parameters(dims[i], init=init)
            for prefix, dims in (("W", list(zip(o_dim, i_dim))), ("b", o_dim)) for i, dim in enumerate(dims)}


def evaluate(params, inputs, layers, dropout, activation, train=False, **kwargs):
    x = dy.concatenate(list(inputs))
    for i in range(layers):
        if train and dropout:
            x = dy.dropout(x, dropout)
        W, b = [dy.parameter(params[key(prefix, i, **kwargs)]) for prefix in ("W", "b")]
        x = activation(W * x + b)
    return x


def key(prefix, i, offset=0, suffix1="", suffix2=()):
    return (prefix + suffix1, i + offset) + suffix2

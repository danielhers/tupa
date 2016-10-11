import time

import numpy as np
from dynet import *

from nn.neural_network import NeuralNetwork
from parsing import config


class FeedforwardNeuralNetwork(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super(FeedforwardNeuralNetwork, self).__init__(*args, model_type=config.FEEDFORWARD_NN, **kwargs)
        self._samples = defaultdict(list)
        self._inputs = {}

    def init_model(self):
        if self.model is not None:
            return
        if config.Config().args.verbose:
            print("Input: " + self.feature_params)
        self.model = Model()
        encoded = []
        for suffix, param in self.feature_params.items():
            x = self._inputs[suffix] = vecInput(param.num)
            # x = Input(shape=(param.num,), dtype="float32" if param.numeric else "int32", name=suffix)
            if not param.numeric:  # index feature
                p = self.model.add_lookup_parameters(suffix, (param.size, param.dim))
                x = lookup(p, x)
                # x = Embedding(output_dim=param.dim, input_dim=param.size, init=self._init,
                #               weights=param.init, input_length=param.num,
                #               W_regularizer=self._regularizer())(x)
            encoded.append(x)
        x = concatenate(encoded)
        # if self._dropout:
        #     x = Dropout(float(self._dropout))(x)
        pW = self.model.add_parameters((x.dim, self._layer_dim))
        pb = self.model.add_parameters((1, self._layer_dim))
        W = parameter(pW)
        b = parameter(pb)
        x = self._activation(W * x + b)
        for _ in range(self._layers - 2):
            pW = self.model.add_parameters((self._layer_dim, self._layer_dim))
            pb = self.model.add_parameters((1, self._layer_dim))
            W = parameter(pW)
            b = parameter(pb)
            x = self._activation(W * x + b)
            # x = Dense(self._layer_dim, activation=self._activation, init=self._init,
            #           W_regularizer=self._regularizer(), b_regularizer=self._regularizer())(x)
        pW = self.model.add_parameters((self._layer_dim, self.max_num_labels))
        pb = self.model.add_parameters((1, self.max_num_labels))
        W = parameter(pW)
        b = parameter(pb)
        x = softmax(W * x + b)
        # out = Dense(self.max_num_labels, activation="softmax", init=self._init, name="out",
        #             W_regularizer=self._regularizer(), b_regularizer=self._regularizer())(x)

    def score(self, features):
        """
        Calculate score for each label
        :param features: extracted feature values, of size input_size
        :return: array with score for each label
        """
        super(FeedforwardNeuralNetwork, self).score(features)
        if not self.is_frozen and self._iteration == 0:  # not fit yet
            return np.zeros(self.num_labels)
        self.init_model()
        scores = self.model.predict_on_batch({k: np.array(v)[None] for k, v in features.items()})
        return scores[0, :self.num_labels]

    def update(self, features, pred, true, importance=1):
        """
        Update classifier weights according to predicted and true labels
        :param features: extracted feature values, of size input_size
        :param pred: label predicted by the classifier (non-negative integer less than num_labels)
        :param true: true label (non-negative integer less than num_labels)
        :param importance: add this many samples with the same features
        """
        super(FeedforwardNeuralNetwork, self).update(features, pred, true, importance)
        for _ in range(int(importance)):
            for name, value in features.items():
                self._samples[name].append(value)
            self._samples["out"].append(true)

    def finish(self, train=False):
        """
        Mark the current item as finished.  Fit the model if reached the batch size.
        :param train: fit the model if batch size reached?
        """
        self._item_index += 1
        if train and self._batch_size is not None and self._item_index >= self._batch_size:
            self.finalize(freeze=False)

    def finalize(self, freeze=True):
        """
        Fit this model on collected samples, and return a frozen model
        :return new FeedforwardNeuralNetwork object with the same weights, after fitting
        """
        super(FeedforwardNeuralNetwork, self).finalize()
        if self._samples:
            started = time.time()
            print("Fitting model...", flush=True)
            x = {}
            y = None
            for name, values in self._samples.items():
                if name == "out":
                    y = values if self._loss == "sparse_categorical_crossentropy" else \
                        np_utils.to_categorical(values, nb_classes=self.max_num_labels)
                else:
                    x[name] = np.array(values)
            self.init_model()
            callbacks = [EarlyStopping(patience=1, verbose=1)]
            if self.filename and config.Config().args.saveeveryepoch:
                # noinspection PyTypeChecker
                callbacks.append(ModelCheckpoint(self.filename + ".{epoch:02d}-{val_loss:.2f}.h5",
                                                 verbose=1, save_best_only=True, save_weights_only=True))
            log = self.model.fit(x, y, batch_size=self._minibatch_size, nb_epoch=self._nb_epochs,
                                 validation_split=config.Config().args.validationsplit, verbose=2, callbacks=callbacks)
            config.Config().log(log.history)
            self._samples = defaultdict(list)
            self._item_index = 0
            self._iteration += 1
            print("Done (%.3fs)." % (time.time() - started))
        if freeze:
            print("Labels: %d" % self.num_labels)
            print("Features: %d" % sum(f.num * (f.dim or 1) for f in self.feature_params.values()))
            return FeedforwardNeuralNetwork(self.filename, list(self.labels), model=self.model)
        return None

import time

import numpy as np
from collections import defaultdict

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, merge, Flatten, Dropout, Embedding, BatchNormalization
from keras.models import Model
from keras.utils import np_utils

from nn.neural_network import NeuralNetwork
from parsing.config import Config, FEEDFORWARD_NN


class FeedforwardNeuralNetwork(NeuralNetwork):

    def __init__(self, *args, model=None, **kwargs):
        super(FeedforwardNeuralNetwork, self).__init__(FEEDFORWARD_NN, *args, model=model, **kwargs)
        self._samples = defaultdict(list)

    def init_model(self):
        if self.model is not None:
            return
        if Config().args.verbose:
            print("Input: " + self.input_params)
        inputs = []
        encoded = []
        frozen_layers = []
        for suffix, param in self.input_params.items():
            if param.dim:
                x = Input(shape=(param.num,), dtype="float32" if param.numeric else "int32", name=suffix)
                inputs.append(x)
                if not param.numeric:  # index feature
                    l = Embedding(output_dim=param.dim, input_dim=param.size, init=self._init,
                                  weights=param.init, input_length=param.num,
                                  W_regularizer=self._regularizer())
                    if not param.updated:
                        frozen_layers.append(l)
                    x = l(x)
                    x = Flatten()(x)
                encoded.append(x)
        x = merge(encoded, mode="concat")
        if self._dropout:
            x = Dropout(float(self._dropout))(x)
        for _ in range(self._layers):
            x = Dense(self._layer_dim, activation=self._activation, init=self._init,
                      W_regularizer=self._regularizer(), b_regularizer=self._regularizer())(x)
            if self._dropout:
                x = Dropout(float(self._dropout))(x)
        out = Dense(self.max_num_labels, activation="softmax", init=self._init, name="out",
                    W_regularizer=self._regularizer(), b_regularizer=self._regularizer())(x)
        self.model = Model(input=inputs, output=[out])
        for l in frozen_layers:
            l.trainable = False
        self.compile()

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

    def finished_item(self, train=False):
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
            if self.filename and Config().args.save_every_epoch:
                # noinspection PyTypeChecker
                callbacks.append(ModelCheckpoint(self.filename + ".{epoch:02d}-{val_loss:.2f}.h5",
                                                 verbose=1, save_best_only=True, save_weights_only=True))
            log = self.model.fit(x, y, batch_size=self._minibatch_size, nb_epoch=self._nb_epochs,
                                 validation_split=Config().args.validation_split, verbose=2, callbacks=callbacks)
            Config().log(log.history)
            self._samples = defaultdict(list)
            self._item_index = 0
            self._iteration += 1
            print("Done (%.3fs)." % (time.time() - started))
        if freeze:
            print("Labels: %d" % self.num_labels)
            print("Features: %d" % sum(f.num * (f.dim or 1) for f in self.input_params.values()))
            return FeedforwardNeuralNetwork(self.filename, list(self.labels), model=self.model,
                                            input_params=self.input_params)
        return None

from collections import defaultdict

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from keras import backend as K, Sequential
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate
from keras.models import Model

from utils.build_rainbow import build_rainbow
from utils.plot_images import plot_images
from utils.show_array import show_array


def get_triples_indices(grouped, n):
    num_classes = len(grouped)
    positive_labels = np.random.randint(0, num_classes, size=n)
    negative_labels = (np.random.randint(1, num_classes, size=n) + positive_labels) % num_classes
    triples_indices = []
    for positive_label, negative_label in zip(positive_labels, negative_labels):
        negative = np.random.choice(grouped[negative_label])
        positive_group = grouped[positive_label]
        m = len(positive_group)
        anchor_j = np.random.randint(0, m)
        anchor = positive_group[anchor_j]
        positive_j = (np.random.randint(1, m) + anchor_j) % m
        positive = positive_group[positive_j]
        triples_indices.append([anchor, positive, negative])
    return np.asarray(triples_indices)


def get_triples_data(x, grouped, n):
    indices = get_triples_indices(grouped, n)
    return x[indices[:, 0]], x[indices[:, 1]], x[indices[:, 2]]


def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.mean(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.mean(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)


def tri_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[:1], y_pred[2:3], y_pred[3:]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


def build_model(input_shape):
    base_input = Input(input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(base_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(2, activation='linear')(x)
    #     x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # force the embedding onto the surface of an n-sphere
    embedding_model = Model(base_input, x, name='embedding')

    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]
    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))
    triplet_model.compile(loss=None, optimizer='adam')

    return embedding_model, triplet_model


def simple_net(input_shape, output_shape):
    embedding_model = Sequential()

    embedding_model.add(Input(input_shape))
    embedding_model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    embedding_model.add(Conv2D(64, (3, 3), activation='relu'))
    embedding_model.add(MaxPooling2D(pool_size=(2, 2)))
    embedding_model.add(Dropout(0.25))
    embedding_model.add(Flatten())
    embedding_model.add(Dense(128, activation='relu'))
    embedding_model.add(Dropout(0.5))
    embedding_model.add(Dense(output_shape, activation='softmax'))

    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]

    triplet_model = Model(inputs=inputs, outputs=outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))
    triplet_model.compile(optimizer='adam', loss=None)

    return embedding_model, triplet_model


class Plotter(keras.callbacks.Callback):
    def __init__(self, embedding_model, x, images, plot_size):
        self.embedding_model = embedding_model
        self.x = x
        self.images = images
        self.plot_size = plot_size

    def on_epoch_end(self, epoch, logs={}):
        clear_output(wait=True)
        xy = self.embedding_model.predict(self.x[:self.plot_size])
        # show_array(255 - plot_images(self.images[:self.plot_size].squeeze(), xy))

        print(xy.shape)
        print(xy)

        plt.scatter(xy[0], xy[1])
        plt.show()


def triplet_generator(x, y, batch_size):
    grouped = defaultdict(list)
    for i, label in enumerate(y):
        grouped[label].append(i)

    while True:
        x_anchor, x_positive, x_negative = get_triples_data(x, grouped, batch_size)
        yield ({'anchor_input': x_anchor,
                'positive_input': x_positive,
                'negative_input': x_negative},
               None)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = x_train.reshape(-1, 28, 28, 1).astype(np.float32) / 255.
    y = y_train.astype(np.int32)

    # build colored versions
    colors = build_rainbow(len(np.unique(y)))
    colored_x = np.asarray([colors[cur_y] * cur_x for cur_x, cur_y in zip(x, y)])

    batch_size = 32
    steps_per_epoch = 32
    epochs = 100
    plot_size = 1024

    # embedding_model, triplet_model = build_model((28, 28, 1))
    embedding_model, triplet_model = simple_net((28, 28, 1), 2)
    plotter = Plotter(embedding_model, x, colored_x, plot_size)

    try:
        history = triplet_model.fit_generator(triplet_generator(x, y, batch_size),
                                              steps_per_epoch=steps_per_epoch,
                                              epochs=epochs,
                                              verbose=0,
                                              callbacks=[plotter])
    except KeyboardInterrupt:
        pass

    plt.plot(history.history['loss'], label='loss')
    plt.legend()
    plt.show()

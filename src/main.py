import os

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Activation, BatchNormalization, Input, concatenate, Flatten, merge, ZeroPadding2D, \
    Convolution2D, Dropout, MaxPooling2D, Conv2D
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.datasets import mnist
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets


def triplet_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[:1], y_pred[2:3], y_pred[3:]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


def simple_net(input_shape, output_shape):
    # input = Input(input_shape)
    # x = Flatten()(input)
    # out = Dense(output_shape, activation='relu')(x)

    # model = Model(input, out)
    # model.summary()

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))

    return model


def get_resnet(input_shape, output_shape):
    input = Input(input_shape)

    model = ResNet50(input_tensor=input, classes=output_shape, weights=None, include_top=False)

    return model


def get_vgg(input_shape, output_shape):
    input = Input(input_shape)

    model = VGG16(input_shape=(28, 28, 3), weights=None, include_top=False)

    return model


def generate_triplets(x_train, y_train, count=100):
    i = 0
    dlist = []
    clist = []

    while True:
        i += 1
        idx = np.random.randint(0, y_train.shape[0])
        anchor, cls = x_train[[idx]], y_train[[idx]]
        pos = x_train[y_train == cls][[i % (y_train == cls).sum()]]
        neg = x_train[y_train != cls][[i % (y_train == cls).sum()]]
        dlist.append([anchor, pos, neg])
        clist.append(cls)
        if i == count:
            return dlist, clist


def tensorboard():
    # Create randomly initialized embedding weights which will be trained.
    N = 10000  # Number of items (vocab size).
    D = 200  # Dimensionality of the embedding.

    sess = tf.Session()

    embedding_var = tf.Variable(tf.random_normal([N, D]), name='word_embedding')

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    saver = tf.train.Saver()
    saver.save(sess, '../logs/model.ckpt', 0)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)


if __name__ == '__main__':
    path = '/Users/Philippe/Programmation/research-purposes/data/a0001-jmac_DSC1459.dng'

    train_length = 100
    test_length = 100
    in_dims = (28, 28, 1)
    out_dims = 3
    LOG_DIR = '../logs'

    #  dataset loading
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')[::10]
    x_test = x_test.astype('float32')[::10]
    y_train = y_train[::10]
    y_test = y_test[::10]
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], y_train.shape, 'train samples')
    print(x_test.shape[0], y_test.shape, 'test samples')

    #  train triplets
    triplets_train, cls_train = generate_triplets(x_train=x_train, y_train=y_train)
    triplets_array = np.asarray(triplets_train)
    x_triplet_train = {
        'input_1': triplets_array[:, 0].reshape(train_length, 28, 28, 1),
        'input_2': triplets_array[:, 1].reshape(train_length, 28, 28, 1),
        'input_3': triplets_array[:, 2].reshape(train_length, 28, 28, 1)
    }

    # test triplets (pos and neg can be useless)
    triplets_test, cls_test = generate_triplets(count=test_length, x_train=x_test, y_train=y_test)
    triplets_array = np.asarray(triplets_test)
    x_triplet_test = {
        'input_1': triplets_array[:, 0].reshape(test_length, 28, 28, 1),
        'input_2': triplets_array[:, 1].reshape(test_length, 28, 28, 1),
        'input_3': triplets_array[:, 2].reshape(test_length, 28, 28, 1)
    }

    #  input tensors
    anc_in = Input(shape=in_dims)
    pos_in = Input(shape=in_dims)
    neg_in = Input(shape=in_dims)

    #  shared network
    base_network = simple_net(in_dims, out_dims)
    print(base_network.summary())

    anc_out = base_network(anc_in)
    pos_out = base_network(pos_in)
    neg_out = base_network(neg_in)

    merged_vector = concatenate([anc_out, pos_out, neg_out])

    model = Model(inputs=[anc_in, pos_in, neg_in], outputs=merged_vector)
    model.compile(optimizer='adam',
                  loss=triplet_loss)
    model.fit(x_triplet_train, np.ones(len(x_triplet_train['input_1'])), steps_per_epoch=64, epochs=1)

    x_train_encoded = model.predict(x_triplet_train)
    x_test_encoded = model.predict(x_triplet_test)

    # keep anchor encoding only
    x_train_encoded_anchor = x_train_encoded[:, [0, 1, 2]]
    x_test_encoded_anchor = x_test_encoded[:, [0, 1, 2]]

    print(x_test_encoded_anchor.shape)

    # reshape raw (not from triplets) mnist data to 2D for KNN
    x_train_2d = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test_2d = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    #  todo : warning : not the same x_train !
    #  take instead x_triplet_train['input_1']reshape(train_length,28²)
    #  knn without encoding
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(x_train_2d, y_train)
    score = neigh.score(x_test_2d, y_test)
    print(score)

    print(x_train_encoded_anchor.shape)

    #  knn with encoding
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(x_train_encoded_anchor, np.ravel(cls_train))
    score = neigh.score(x_test_encoded_anchor, cls_test)
    print(score)

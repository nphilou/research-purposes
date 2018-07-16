import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Activation, BatchNormalization, Input, concatenate, Flatten, merge, ZeroPadding2D, \
    Convolution2D, Dropout, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.datasets import mnist


def triplet_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[:1], y_pred[2:3], y_pred[3:]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


def create_base_network(input_shape, output_shape):
    """
    Base network to be shared.
    """

    input = Input(input_shape)
    x = Flatten()(input)
    out = Dense(output_shape, activation='relu')(x)

    model = Model(input, out)
    model.summary()

    return model


def get_resnet(input_shape, output_shape):
    input = Input(input_shape)

    model = ResNet50(input_tensor=input, classes=output_shape, weights=None, include_top=False)

    return model


def get_vgg(input_shape, output_shape):
    input = Input(input_shape)

    model = VGG16(input_tensor=input, classes=output_shape, weights=None, include_top=False)

    return model


def generate_triplets():
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

    i = 0

    dlist = []

    while True:
        i += 1
        idx = np.random.randint(0, y_train.shape[0])
        anchor, cls = x_train[[idx]], y_train[[idx]]
        pos = x_train[y_train == cls][[i % (y_train == cls).sum()]]
        neg = x_train[y_train != cls][[i % (y_train == cls).sum()]]
        dlist.append([anchor, pos, neg])
        if i == 10:
            return dlist


if __name__ == '__main__':
    path = '/Users/Philippe/Programmation/research-purposes/data/a0001-jmac_DSC1459.dng'
    # with rawpy.imread(path) as raw:
    #     rgb = raw.postprocess()

    # plt.imshow(rgb)
    # plt.show()
    # print(rgb.shape)

    # imageio.imsave('default.tiff', rgb)

    # anchor_in = Input(shape=1)
    # pos_in = Input(shape=1)
    # neg_in = Input(shape=1)

    in_dims = (28, 28, 1)
    out_dims = 2

    # Create the 3 inputs
    anc_in = Input(shape=in_dims)
    pos_in = Input(shape=in_dims)
    neg_in = Input(shape=in_dims)

    # base_network = create_base_network(in_dims, out_dims)

    base_network = get_vgg(in_dims, out_dims)

    print(base_network.summary())

    anchor_out = base_network(anc_in)
    pos_out = base_network(pos_in)
    neg_out = base_network(neg_in)

    print("[INFO] model - shape: %s" % str(anchor_out.get_shape()))
    print("[INFO] model - shape: %s" % str(pos_out.get_shape()))
    print("[INFO] model - shape: %s" % str(neg_out.get_shape()))

    print(anchor_out.shape)

    merged_vector = concatenate([anchor_out, pos_out, neg_out])

    print(merged_vector.shape)

    # Define the trainable model
    model = Model(inputs=[anc_in, pos_in, neg_in], outputs=merged_vector)
    model.compile(optimizer='adam',
                  loss=triplet_loss)

    # model.fit(x=[anchor_train, pos_train, neg_train], y=[1], batch_size=256, epochs=10, verbose=2)

    # for i in generate_triplets():
    # print(len(i[0][0][0]))  # anchor
    # print(len(i[0][1][0]))  # pos
    # print(len(i[0][2][0]))  # neg
    # print(len(i[1]))  # cls

    # x = i[0]
    # break

    triplets = generate_triplets()

    print(triplets[0].__len__())

    data = np.asarray(triplets)

    print(data.shape)

    print(data[1, 0].shape)

    plt.imshow(data[1, 0].reshape(28, 28), cmap='gray')
    plt.show()

    plt.imshow(data[1, 1].reshape(28, 28), cmap='gray')
    plt.show()

    plt.imshow(data[1, 2].reshape(28, 28), cmap='gray')
    plt.show()

    X_te = {
        'input_1': data[:, 0].reshape(10, 28, 28, 1),
        'input_2': data[:, 1].reshape(10, 28, 28, 1),
        'input_3': data[:, 2].reshape(10, 28, 28, 1)
    }

    # np.squeeze

    print(data.shape)
    print(type(np.asarray(data)))

    model.fit(X_te, np.ones(len(X_te['input_1'])), steps_per_epoch=512, epochs=1)

    x_test_encoded = model.predict(X_te, batch_size=5)

    print(x_test_encoded)

    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=np.ones(len(X_te['input_1'])))
    plt.colorbar()
    # plt.show()

    # disjonctive loss
    # siamese network
    # downsampling

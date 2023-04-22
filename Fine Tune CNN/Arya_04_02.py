# Arya, Shubham
# 1001_650_536
# 2023_04_16
# Assignment_04_02

import pytest
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import keras
import Arya_04_01 as CNN
# import tensorflow.keras as keras

def test_evaluate():
    tf.keras.utils.set_random_seed(123)
    # get dataset
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    X_train = tf.reshape(X_train, (60000, 28, 28, 1))
    X_test = tf.reshape(X_test, (10000, 28, 28, 1))
    X_train = X_train[:100]
    X_test = X_test[:100]
    Y_train = Y_train[:100]
    Y_test = Y_test[:100]
    Y_train = tf.keras.utils.to_categorical(Y_train)
    Y_test = tf.keras.utils.to_categorical(Y_test)
    # normalize dataset
    X_train = X_train / 255
    X_test = X_test / 255
    my_cnn = CNN.CNN()
    model = my_cnn.model

    # add layers to this model 
    my_cnn.add_input_layer(shape=(28,28,1), name="input layer")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, padding="same", strides=1, activation="relu", name="layer_1")
    my_cnn.append_flatten_layer(name="layer_2")
    my_cnn.append_dense_layer(num_nodes=10, activation="relu", name="layer_3")

    # set optimizer, metric and loss function
    my_cnn.set_optimizer("SGD")
    my_cnn.set_metric("accuracy")
    my_cnn.set_loss_function("hinge")

    # compile model
    model.compile(optimizer=my_cnn.optimizer, loss=my_cnn.loss_function, metrics=my_cnn.metric)
    
    # train
    my_cnn.train(X_train=X_train, y_train=Y_train, batch_size=20, num_epochs=5)

    # evaluate
    loss, metric = my_cnn.evaluate(X=X_test, y=Y_test)

    # test evaluation results
    assert (loss == 1.0041534900665283)
    assert (metric == 0.8059999942779541)
    print("Passed test_evaluate")
    
def test_train():
    tf.keras.utils.set_random_seed(123)
    # get dataset
    (X_train, Y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = tf.reshape(X_train, (60000, 28, 28, 1))
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    Y_train = tf.keras.utils.to_categorical(Y_train)
    # normalize dataset
    X_train = X_train / 255
    my_cnn = CNN.CNN()
    model = my_cnn.model

    # add layers to this model 
    my_cnn.add_input_layer(shape=(28,28,1), name="input layer")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, padding="same", strides=1, activation="relu", name="layer_1")
    my_cnn.append_flatten_layer(name="layer_2")
    my_cnn.append_dense_layer(num_nodes=10, activation="relu", name="layer_3")

    # set optimizer, metric and loss function
    my_cnn.set_optimizer("SGD")
    my_cnn.set_metric("accuracy")
    my_cnn.set_loss_function("hinge")

    # compile model
    model.compile(optimizer=my_cnn.optimizer, loss=my_cnn.loss_function, metrics=my_cnn.metric)
    
    # get train history 
    history = my_cnn.train(X_train=X_train, y_train=Y_train, batch_size=20, num_epochs=5)
    assert (history == [1.0280656814575195, 1.0089970827102661, 1.0048623085021973, 1.0029592514038086, 1.00175142288208])
    print("Passed the test_train()")
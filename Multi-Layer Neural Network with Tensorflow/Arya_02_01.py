# Arya, Shubham
# 1001_650_536
# 2023_03_19
# Assignment_02_01

import tensorflow as tf
import numpy as np

def initialize_weights(layers, columns, seed):
    W = []
    for layer in layers:
        np.random.seed(seed)
        w = tf.Variable(np.random.randn(columns,layer), dtype=tf.float32)
        W.append(w)
        columns = layer + 1
    return W

def weights_to_tensors(weights):
    weights_tf = []
    for weight in weights:
        w = tf.Variable(weight)
        weights_tf.append(w)
    return weights_tf
    
def forward_pass(X, W, activations):
    inputs = tf.cast(X, tf.float32)
    for i in range(len(W)):
        net = tf.matmul(tf.transpose(tf.cast(W[i], tf.float32)),tf.transpose(inputs))
        out = activation_function(activations[i], net)
        if i + 1 <= len(W): # if it is the last layer in W, then dont convert output to next layers input
            inputs = current_layer_output_to_next_layer_input(out)
    return out

def current_layer_output_to_next_layer_input(out):
    out = tf.transpose(out)
    ones = tf.ones(shape=(tf.shape(out)[0], 1), dtype=out.dtype)
    out = tf.cast(tf.concat([ones, out], axis=1), tf.float64)
    return tf.cast(out, tf.float32)

def activation_function(activation, net):
    if activation == "sigmoid":
        return sigmoid(net)
    elif activation == "relu":
        return relu(net)
    else: # linear activation
        return net

def sigmoid(net):
    return tf.divide(1.0, 1.0 + tf.exp(-net))

def relu(net):
    return tf.maximum(0, net)

def loss_function(loss, Y_true, Y_pred):
    if loss == "mse":
        return mse(Y_true, Y_pred)
    elif loss == "cross_entropy":
        return cross_entropy(Y_true, Y_pred)
    else: # svm
        return svm(Y_true, Y_pred)
    
def mse(Y_true, Y_pred):
    Y_true = tf.cast(Y_true, tf.float64)
    Y_pred = tf.transpose(tf.cast(Y_pred, tf.float64)) # to have the same dimensions
    return tf.reduce_mean(tf.square(Y_true - Y_pred))

def cross_entropy(Y_true, Y_pred):
    Y_pred = tf.transpose(Y_pred)
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y_true, logits=Y_pred)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    return cross_entropy_loss

def svm(Y_true, Y_pred):
    Y_pred = tf.transpose(Y_pred)
    correct_scores = tf.reduce_sum(Y_true * Y_pred, axis=1)
    margins = tf.maximum(0.0, Y_pred - tf.expand_dims(correct_scores, axis=1))
    loss = tf.reduce_mean(tf.reduce_sum(margins, axis=1))
    loss_fn = tf.keras.losses.Hinge() # above code gives answer that is slighlty off.
    loss = loss_fn(Y_true, Y_pred)
    return loss

def update_weights(W, gradient, alpha):
    for (i, grad) in enumerate(gradient):
        W[i].assign_sub(alpha*grad)
    return W

def convert_to_tensorflow_variables(X, Y):
    X = tf.Variable(X, dtype=tf.float32)
    ones = tf.ones(shape=(tf.shape(X)[0], 1), dtype=X.dtype)
    X = tf.concat([ones, X], axis=1)
    Y = tf.Variable(Y, dtype=tf.float32)
    return X,Y

# This is a helper function Jason provided me
def split_data(X_train, Y_train, split_range=[0.2, 0.7]):
    start = int(split_range[0] * X_train.shape[0])
    end = int(split_range[1] * X_train.shape[0])
    return np.concatenate((X_train[:start], X_train[end:])), np.concatenate(
        (Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end]

# This is a helper function Jason provided me
def generate_batches(X, y, batch_size=32):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    if X.shape[0] % batch_size != 0:
        yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]

def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",validation_split=[0.8,1.0],weights=None,seed=2):

    X_train, Y_train, X_val, Y_val = split_data(X_train, Y_train, split_range=validation_split)
    
    # transforming training set into tensorflow variables
    X,Y = convert_to_tensorflow_variables(X_train, Y_train)
    
    # transforming validation set into tensorflow variables
    X_v, Y_v = convert_to_tensorflow_variables(X_val, Y_val)
    
    columns = tf.shape(X)[1].numpy()
    W = initialize_weights(layers, columns, seed) if weights == None else weights_to_tensors(weights)

    errors = []

    for epoch in range(epochs):                             # for each epoch
        for (X,Y) in generate_batches(X, Y, batch_size):    # for each batch in training batch set
            with tf.GradientTape() as tape:
                Y_pred = forward_pass(X, W, activations)    # find predicted values with the weights and batch
                error = loss_function(loss, Y, Y_pred)      # calculate loss with predicted values and actual values

            grads = tape.gradient(error, W)                 # finds gradient of error wrt to weights
            W = update_weights(W, grads, alpha)             # update weights 
            
        Y_val_pred = forward_pass(X_v, W, activations)      # predicted values with validation set
        error = loss_function(loss, Y_v, Y_val_pred)        # loss with validation set
        errors.append(error.numpy())                        # error is a tensor type. error.numpy() gives numpy array
    
    output = forward_pass(X_v,W,activations).numpy()        # output with validation set
    
    weights = []
    for w in W:
        weights.append(w.numpy())                           # weights are converted to np array from tf array

    return [weights, errors, np.transpose(output)]

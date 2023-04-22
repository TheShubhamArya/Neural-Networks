# Arya, Shubham
# 1001_650_536
# 2023_02_26
# Assignment_01_01

import numpy as np

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    mses = []
    column = np.array(X_train).shape[0] + 1                    # number of weight column needed in first layer
    weights = intialize_weights(layers, column, seed)          # initialize weights with seed for layers
    dW = intialize_weights(layers,column,seed,withZeros=True)  # set weight change to zeros initially
    
    for epoch in range(epochs):                                # goes over all samples epoch times
        for (X,Y) in zip(X_train.T,Y_train.T):                 # goes through every train sample
            X = add_1s_to_first_col_of_input(X)                # train sample altered to account for bias in W
            for layer in range(len(layers)):                   # goes through every layer
                for i in range(len(weights[layer])):           # goes through every node in layer
                    for j in range(len(weights[layer][i])):    # goes through every weight in a node

                        weights[layer][i][j] += h              # adjust Wij by + h
                        Y_h1 = feedforward(weights, X)         # find output after Wij is changed
                        mse_h1 = mean_square_error(Y, Y_h1)    # calculating mse with Wij + h

                        weights[layer][i][j] -= 2*h            # adjust Wij by - h (- 2h to account for +h earlier)
                        Y_h2 = feedforward(weights, X)         # find output after Wij is changed
                        mse_h2 = mean_square_error(Y, Y_h2)    # calculate mse with Wij - h

                        dw = (mse_h1 - mse_h2)/(2*h)           # calculate the change in mse w.r.t. Wij
                        dW[layer][i][j] = dw                   # store this dw to change weights later
                        
                        weights[layer][i][j] += h              # bring the weight chnage back to original
                        
            weights = update_weights(weights, alpha, dW)       # updates weights with dW

        output = predict(weights, X_test)                      # predict outputs for test data
        mse = mean_square_error(Y_test, output)                # calculate mse with actual predictions
        mses.append(mse)

    output = predict(weights, X_test)                          # final output prediction for X_test
    return [weights, mses, output]

def predict(W, X_test):
    X = X_test
    ones = np.ones((1,X_test.shape[1]))
    X = np.vstack((ones,X))
    for (_,w) in enumerate(W):
        net = np.dot(w,X)
        X = np.array(sigmoid_function(net))
        ones = np.ones((1,X.shape[1]))
        X = np.vstack((ones,X))
    X = np.delete(X, 0, 0) # delete the row with ones
    return X

def feedforward(weights, X):
    for (_,W) in enumerate(weights):
        net = np.dot(X, W.T)
        X = np.array(sigmoid_function(net))
        X = add_1s_to_first_col_of_input(X)
    X = np.delete(X, 0, 0)
    return X

def update_weights(weights, alpha, dW):
    for layer in range(len(weights)):
        for i  in range(len(weights[layer])):
            for j in range(len(weights[layer][i])):
                weights[layer][i][j] -= alpha*dW[layer][i][j]
    return weights

# initializes weights for each layer. W.shape = nodes x (column + 1)
def intialize_weights(layers,column,seed,withZeros = False):
    weights = [None] * len(layers)
    for (i,layer) in enumerate(layers):
        np.random.seed(seed=seed)
        weight = np.random.randn(layer,column)
        if withZeros:
            weight = np.zeros((layer,column))
        weights[i]=weight
        column = layer + 1
    return weights

# calculates and returns the mean square error.
def mean_square_error(actual, prediction):
    return (np.square(actual - prediction)).mean()

# sigmoid activation function that takes net value from a node
def sigmoid_function(net):
    return 1/(1 + np.exp(-net))

def add_1s_to_first_col_of_input(X):
    return np.hstack((1,X))

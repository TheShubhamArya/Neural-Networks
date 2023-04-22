# Arya, Shubham
# 1001_650_536
# 2023_04_02
# Assignment_03_01

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def confusion_matrix(y_true, y_pred, n_classes=10):
    # Compute the confusion matrix for a set of predictions
    matrix = np.zeros((n_classes, n_classes))
    for i in range(len(y_pred)):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])
        matrix[true_label, pred_label] += 1
    return matrix

def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):
    tf.keras.utils.set_random_seed(5368) # do not remove this line

    model = tf.keras.models.Sequential()
    # All layers that have weights should have L2 regularization with a regularization strength of 0.0001 (only use kernel regularizer)
    regularizer = tf.keras.regularizers.l2(0.0001)
    # - Convolutional layer with 8 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizer, input_shape=X_train[0].shape))
    # - Convolutional layer with 16 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizer))
    # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    # - Convolutional layer with 32 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizer))
    # - Convolutional layer with 64 filters, kernel size 3 by 3 , stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizer))
    # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    # - Flatten layer
    model.add(Flatten())
    # - Dense layer with 512 units and ReLU activation
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizer))
    # - Dense layer with 10 units with linear activation
    model.add(Dense(10, activation='linear', kernel_regularizer=regularizer))
    # - a softmax layer
    model.add(tf.keras.layers.Activation("softmax"))

    # The neural network should be trained using the Adam optimizer with default parameters. The loss function should be categorical cross-entropy.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # The number of epochs should be given by the 'epochs' parameter. The batch size should be given by the 'batch_size' parameter.
    training_history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    y_pred = predict(model, X_test)

    # You should save the keras model to a file called 'model.h5' (do not submit this file). When we test run your program we will check "model.h5"
    save(model)

    # You should compute the confusion matrix on the test set and return it as a numpy array.
    Y_test_max = np.argmax(Y_test, axis=1)
    matrix = confusion_matrix(Y_test_max, y_pred, Y_test.shape[1])

    # You should plot the confusion matrix using the matplotlib function matshow (as heat map) and save it to 'confusion_matrix.png'
    save_confusion_matrix_image(matrix)

    return model, training_history, matrix, y_pred  


# returns the predicted values given a model and dataset
def predict(model, X_test):
    y_pred = model.predict(X_test)
    return np.array(np.argmax(y_pred, axis=1))

# saves the model as .h5
def save(model):
    model.save('model.h5')

# saves confusion matrix image to the current folder
def save_confusion_matrix_image(confusion_matrix):
    _, px = plt.subplots(figsize=(15, 15))
    px.matshow(confusion_matrix, cmap="cool", alpha=0.5)
    # To show the numbers on the matrix tiles
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            px.text(i,j,str(int(confusion_matrix[i, j])), va='center', ha='center', size='large')

    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.ylabel('True', fontsize=17)
    plt.xlabel('Prediction', fontsize=17)
    plt.title('Confusion Matrix', fontsize=25)
    # this is to separate the tiles with a white line
    for i in range(confusion_matrix.shape[0]-1):
        plt.axhline(i+0.5, color='white', linewidth=1)
        plt.axvline(i+0.5, color='white', linewidth=1)
    plt.savefig('confusion_matrix.png')
    
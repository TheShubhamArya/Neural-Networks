import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):
    tf.keras.utils.set_random_seed(5368) # do not remove this line

    

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # training_history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model = create_model()
    model.summary()
    return model

def create_model():
    model = tf.keras.models.Sequential()
    regularizer = tf.keras.regularizers.l2(0.0001)
    # model.add(tf.keras.layers.ZeroPadding2D(padding=(8, 8)))
    model.add(Conv2D(filters=8, kernel_size=(7,7), strides=(5,5), padding='same', activation='relu', kernel_regularizer=regularizer, input_shape=(256,256,3)))
    model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizer))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(4,4 ), padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), padding='same', activation='relu', kernel_regularizer=regularizer))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizer))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizer))
    model.add(Dense(10, activation='linear', kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Activation("softmax"))
    return model

model = create_model()
"""
W layer 1 = 3x5x3x10 => 460 parameters
Output layer 1 => Nonex30x28x10
"""

# model = tf.keras.models.Sequential([
    # Dense(16, input_shape=(20,20,3), activation='relu'), # (*3 + 1)*16, output=None*20*20*16
    # Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'), # 3x3x16x32, params = 145*32, output=Nonex20x20x32
    # MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'), # Nonex10x10x32
    # Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'), # 5x5x32x64, params=51264, output= None*6*6*64
    # Flatten(),
    # Dense(2, activation='softmax')
#     Conv2D(filters=16, kernel_size=(4,4), strides=(2,2), padding="same", activation="relu",input_shape=(28,28,3))
# ])
# 3x3x3x16, parans 28*16, output = (28 - 4)/2 + 1 None*13*13*16
model.summary()
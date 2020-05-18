import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda
from keras.models import Model, Sequential
from keras.initializers import Initializer
import keras.backend as K
from keras import regularizers
from keras.utils import plot_model
from keras.optimizers import Adam
def get_siamese_model(input_shape=(105, 105, 3)):
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(12e-4)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (7, 7), activation='relu', bias_initializer='random_normal', kernel_initializer='random_normal',
                     kernel_regularizer=regularizers.l2(12e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (4, 4), activation='relu', bias_initializer='random_normal', kernel_initializer='random_normal',
                     kernel_regularizer=regularizers.l2(12e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (4, 4), activation='relu', bias_initializer='random_normal',
                     kernel_initializer='random_normal',
                     kernel_regularizer=regularizers.l2(12e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-3), kernel_initializer='random_normal', bias_initializer='random_normal'))
    left_inp = Input(input_shape)
    right_inp = Input(input_shape)
    encoded_1 = model(left_inp)
    encoded_2 = model(right_inp)

    L1_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))

    L1_distance = L1_layer([encoded_1, encoded_2])

    prediction = Dense(1, activation='sigmoid', bias_initializer='random_normal')(L1_distance)

    siamese_net = Model(inputs = [left_inp, right_inp], outputs = prediction)
    return siamese_net

model = get_siamese_model()